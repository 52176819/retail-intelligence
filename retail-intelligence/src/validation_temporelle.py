
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


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "retail.duckdb"

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
MAX_TRAIN_ROWS = 150_000
RANDOM_STATE = 42
VALIDATION_DAYS = 14
N_PERIODS = 3


def calculate_rmsle(actual, predicted):
    """Calcule la racine de l'erreur quadratique logarithmique moyenne."""
    predicted = np.maximum(np.asarray(predicted), 0)
    actual = np.asarray(actual)

    return np.sqrt(mean_squared_log_error(actual, predicted))


def create_model():
    """Construit le pipeline de prétraitement et le modèle Random Forest."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLS),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=14,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main():
    print("Connexion à la base DuckDB...")

    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        columns = ", ".join(["date", "sales"] + FEATURE_COLS)

        query = f"""
            SELECT {columns}
            FROM features.sales_features
            ORDER BY date
        """

        df = con.execute(query).fetchdf()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    last_date = df["date"].max()

    print("Première date :", df["date"].min().date())
    print("Dernière date :", last_date.date())
    print("Nombre total de lignes :", len(df))
    print()

    results = []

    # On teste trois périodes consécutives de 14 jours.
    # Pour chaque période, l'entraînement s'arrête avant son début.
    for period_number in range(N_PERIODS, 0, -1):
        validation_start = last_date - pd.Timedelta(
            days=VALIDATION_DAYS * period_number - 1
        )
        validation_end = validation_start + pd.Timedelta(
            days=VALIDATION_DAYS - 1
        )

        train_mask = df["date"] < validation_start
        valid_mask = (
            (df["date"] >= validation_start)
            & (df["date"] <= validation_end)
        )

        train_df = df.loc[train_mask]
        valid_df = df.loc[valid_mask]

        if train_df.empty or valid_df.empty:
            print("Période ignorée : données insuffisantes.")
            continue

        X_train = train_df[FEATURE_COLS]
        y_train = train_df["sales"]

        X_valid = valid_df[FEATURE_COLS]
        y_valid = valid_df["sales"]

        # Limiter le temps d'entraînement tout en gardant
        # un échantillon reproductible.
        if len(X_train) > MAX_TRAIN_ROWS:
            X_train = X_train.sample(
                n=MAX_TRAIN_ROWS,
                random_state=RANDOM_STATE,
            )
            y_train = y_train.loc[X_train.index]

        print("=" * 60)
        print(f"PÉRIODE DE VALIDATION {period_number}")
        print("Début :", validation_start.date())
        print("Fin   :", validation_end.date())
        print("Lignes d'entraînement :", len(X_train))
        print("Lignes de validation  :", len(X_valid))
        print("Entraînement du Random Forest...")

        model = create_model()

        y_train_log = np.log1p(y_train)
        model.fit(X_train, y_train_log)

        predictions_log = model.predict(X_valid)
        predictions = np.expm1(predictions_log)
        predictions = np.maximum(predictions, 0)

        rmsle_rf = calculate_rmsle(y_valid, predictions)

        # Modèle de référence : ventes observées 14 lignes avant.
        baseline_predictions = valid_df["lag_14"].to_numpy()
        rmsle_baseline = calculate_rmsle(
            y_valid,
            baseline_predictions,
        )

        improvement = (
            (rmsle_baseline - rmsle_rf) / rmsle_baseline * 100
            if rmsle_baseline > 0
            else np.nan
        )

        print("RMSLE Random Forest :", round(rmsle_rf, 5))
        print("RMSLE lag_14         :", round(rmsle_baseline, 5))
        print("Amélioration (%)     :", round(improvement, 2))

        results.append(
            {
                "periode_debut": validation_start.date(),
                "periode_fin": validation_end.date(),
                "lignes_validation": len(X_valid),
                "RMSLE_Random_Forest": rmsle_rf,
                "RMSLE_lag_14": rmsle_baseline,
                "amelioration_pct": improvement,
            }
        )

    print("\n" + "=" * 75)
    print("RÉSUMÉ DES VALIDATIONS TEMPORELLES")

    if results:
        summary = pd.DataFrame(results)
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

        print("\nMOYENNES DES TROIS PÉRIODES")
        print(
            summary[
                [
                    "RMSLE_Random_Forest",
                    "RMSLE_lag_14",
                    "amelioration_pct",
                ]
            ].mean().to_string(float_format=lambda x: f"{x:.5f}")
        )
    else:
        print("Aucune période n'a pu être évaluée.")


if __name__ == "__main__":
    main()
