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

# Colonnes que le modèle utilisera comme indices
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

CATEGORICAL_COLS = [
    "store_nbr",
    "family",
    "store_type",
    "store_cluster",
]

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def main():
    # Ouvrir la base DuckDB déjà créée par ingest.py
    con = duckdb.connect(str(DB_PATH), read_only=True)

    # Lire les colonnes nécessaires depuis la table de features
    columns = ", ".join(["date", "sales"] + FEATURE_COLS)
    query = f"""
        SELECT {columns}
        FROM features.sales_features
        ORDER BY date
    """
    df = con.execute(query).fetchdf()

    # Fermer la base après avoir récupéré les données
    con.close()

    # S'assurer que date est bien reconnue comme une date
    df["date"] = pd.to_datetime(df["date"])

    # La cible est la valeur que le modèle doit apprendre à prévoir
    y = df["sales"]

    # Les indices fournis au modèle (on ne met pas sales ici)
    X = df[FEATURE_COLS]

    # Garder les 16 dates les plus récentes pour vérifier le modèle
    last_date = df["date"].max()
    validation_start = last_date - pd.Timedelta(days=14)

    train_mask = df["date"] < validation_start
    valid_mask = df["date"] >= validation_start

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    X_valid = X.loc[valid_mask]
    y_valid = y.loc[valid_mask]

    print("Dernière date disponible :", last_date.date())
    print("Début validation         :", validation_start.date())
    print("Lignes entraînement      :", len(X_train))
    print("Lignes validation        :", len(X_valid))

    # Garder au maximum 150 000 lignes d'entraînement pour ce premier essai
    if len(X_train) > 150_000:
        sampled = X_train.sample(n=150_000, random_state=42).index
        X_train = X_train.loc[sampled]
        y_train = y_train.loc[sampled]
        print("Entraînement limité à    :", len(X_train), "lignes")

    # Pour les colonnes numériques :
    # remplacer les valeurs manquantes par la médiane
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    # Pour les colonnes texte/catégories :
    # remplacer les valeurs manquantes et transformer les catégories en nombres
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Appliquer les traitements appropriés aux deux types de colonnes
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLS),
        ]
    )

    # Le modèle apprend les ventes en échelle logarithmique
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

    # log1p transforme les ventes : log(1 + ventes)
    y_train_log = np.log1p(y_train)

    print("Entraînement du modèle...")
    model.fit(X_train, y_train_log)

    # Demander au modèle de prédire les 14 dernières dates
    predictions_log = model.predict(X_valid)

    # Revenir de l'échelle logarithmique aux ventes normales
    predictions = np.expm1(predictions_log)

    # Les ventes ne peuvent pas être négatives
    predictions = np.maximum(predictions, 0)

    # Calculer le score RMSLE de validation
    rmsle = np.sqrt(mean_squared_log_error(y_valid, predictions))

    print("RMSLE sur les 14 dernières dates :", round(rmsle, 5))

    # Afficher quelques comparaisons entre vrai et prédit
    comparison = pd.DataFrame(
        {
            "date": df.loc[valid_mask, "date"].values,
            "ventes_reelles": y_valid.values,
            "ventes_predit": predictions,
        }
    )

    print("\nQuelques prédictions comparées aux ventes réelles :")
    print(comparison.head(10).to_string(index=False))
    baseline_predictions = df.loc[valid_mask, "lag_14"].to_numpy()
    baseline_rmsle=np.sqrt(mean_squared_log_error(y_valid, baseline_predictions))
    print("RMSLE de la référence lag_14 :", round(baseline_rmsle, 5))
    



if __name__ == "__main__":
    main()
