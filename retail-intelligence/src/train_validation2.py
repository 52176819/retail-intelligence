
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_log_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Trouver le dossier principal du projet
ROOT = Path(__file__).resolve().parents[1]

# Emplacement de la base créée par ingest.py
DB_PATH = ROOT / "data" / "processed" / "retail.duckdb"

# Colonnes numériques utilisées par le modèle
NUMERIC_COLS = [
    "onpromotion",
    "is_holiday",
    "dow",
    "dom",
    "month",
    "lag_14",
    "lag_21",
    "lag_28",
    "roll_mean_7",
    "roll_mean_27",
    "roll_std_27",
    "promo_mean_7",
]

# Colonnes catégorielles
CATEGORICAL_COLS = [
    "store_nbr",
    "family",
    "store_type",
    "store_cluster",
]

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def main():
    # Ouvrir la base DuckDB
    con = duckdb.connect(str(DB_PATH), read_only=True)

    try:
        print(
            con.execute(
                "DESCRIBE SELECT * FROM features.sales_features"
            ).fetchdf()
        )

        # Lire les colonnes nécessaires
        columns = ", ".join(["date", "sales"] + FEATURE_COLS)

        query = f"""
            SELECT {columns}
            FROM features.sales_features
            ORDER BY date
        """

        df = con.execute(query).fetchdf()

    finally:
        con.close()

    # Vérifier les variables retardées
    print("\n--- Vérification des variables retardées ---")

    print("Colonnes disponibles :")
    print([c for c in df.columns if c.startswith("lag_")])

    for col in ["lag_14", "lag_21", "lag_28"]:
        if col in df.columns:
            print(
                f"{col} : {df[col].isna().sum()} valeurs manquantes "
                f"sur {len(df)} lignes"
            )
        else:
            print(f"ATTENTION : {col} est absente !")

    # Préparer les dates
    df["date"] = pd.to_datetime(df["date"])

    print("\nPremière date :", df["date"].min())
    print("Dernière date :", df["date"].max())

    # Vérifier que les ventes sont valides pour le RMSLE
    if df["sales"].isna().any():
        raise ValueError("La colonne sales contient des valeurs manquantes.")

    if (df["sales"] < 0).any():
        raise ValueError(
            "La colonne sales contient des valeurs négatives : "
            "le RMSLE standard ne peut pas être calculé."
        )

    # Cible et variables explicatives
    y = df["sales"]
    X = df[FEATURE_COLS]
    # Période de validation : les 14 dernières dates calendaires
    last_date = df["date"].max()

    validation_end = last_date
    validation_start = validation_end - pd.Timedelta(days=13)

    train_mask = df["date"] < validation_start

    valid_mask = (
        (df["date"] >= validation_start)
        & (df["date"] <= validation_end)
)

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    X_valid = X.loc[valid_mask]
    y_valid = y.loc[valid_mask]

    print(
        "\nPériode de vérification :",
        validation_start.date(),
        "au",
        validation_end.date(),
    )
    print("Lignes entraînement     :", len(X_train))
    print("Lignes validation       :", len(X_valid))

    if X_valid.empty:
        raise ValueError("La période de validation ne contient aucune ligne.")

    # Limiter l'entraînement à 150 000 lignes
    if len(X_train) > 150_000:
        sampled = X_train.sample(
            n=150_000, random_state=42
        ).index

        X_train = X_train.loc[sampled]
        y_train = y_train.loc[sampled]

        print("Entraînement limité à   :", len(X_train), "lignes")

    # Traitement des variables numériques
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    # Traitement des variables catégorielles
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Prétraitement
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLS),
        ]
    )

    # Random Forest
    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=14,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )

    # Transformer les ventes en log(1 + ventes)
    y_train_log = np.log1p(y_train)

    print("\nEntraînement du modèle...")
    model.fit(X_train, y_train_log)

    # Prédictions Random Forest
    predictions_log = model.predict(X_valid)
    predictions = np.expm1(predictions_log)
    predictions = np.maximum(predictions, 0)

    # RMSLE du Random Forest
    rmsle = np.sqrt(
        mean_squared_log_error(y_valid, predictions)
    )

    print(f"\nRMSLE Random Forest : {rmsle:.5f}")

    # Comparer quelques prédictions aux ventes réelles
    comparison = pd.DataFrame(
        {
            "date": df.loc[valid_mask, "date"].values,
            "ventes_reelles": y_valid.values,
            "ventes_predit": predictions,
        }
    )

    print("\nQuelques prédictions comparées aux ventes réelles :")
    print(comparison.head(10).to_string(index=False))

    # Modèle de référence : ventes d'il y a 14 jours
    print("\n--- Comparaison des modèles ---")

    baseline_predictions = df.loc[
        valid_mask, "lag_14"
    ].to_numpy(dtype=float)

    # Le RMSLE exige des prédictions non négatives et sans NaN.
    # On évalue les deux modèles sur les mêmes lignes.
    baseline_valid = np.isfinite(baseline_predictions)

    if not baseline_valid.all():
        print(
            "Attention : certaines lignes de validation ont un lag_14 "
            "manquant ou non fini."
        )

    if baseline_valid.any():
        baseline_y = y_valid.to_numpy()[baseline_valid]
        baseline_pred = np.maximum(
            baseline_predictions[baseline_valid], 0
        )
        rf_pred = predictions[baseline_valid]

        baseline_rmsle = np.sqrt(
            mean_squared_log_error(baseline_y, baseline_pred)
        )

        rf_rmsle_comparable = np.sqrt(
            mean_squared_log_error(baseline_y, rf_pred)
        )

        print(f"RMSLE Random Forest : {rf_rmsle_comparable:.5f}")
        print(f"RMSLE lag_14        : {baseline_rmsle:.5f}")

        if rf_rmsle_comparable < baseline_rmsle:
            print("Résultat : Random Forest est meilleur.")
        elif rf_rmsle_comparable > baseline_rmsle:
            print("Résultat : le modèle lag_14 est meilleur.")
        else:
            print("Résultat : les deux modèles ont le même score.")
    else:
        print(
            "Comparaison impossible : aucune valeur valide de lag_14 "
            "dans la période de validation."
        )


if __name__ == "__main__":
    main()
