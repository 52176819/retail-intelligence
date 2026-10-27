
from pathlib import Path

import duckdb
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "retail.duckdb"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "random_forest_retail.joblib"

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
    print("Lecture des données DuckDB...")

    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        query = """
            SELECT date, sales, *
            FROM features.sales_features
            ORDER BY date
        """
        df = con.execute(
            "SELECT * FROM features.sales_features ORDER BY date"
        ).fetchdf()

    df["date"] = np.array(df["date"], dtype="datetime64[ns]")
    df = df.dropna(subset=["sales"])

    # Réserver les 14 derniers jours pour une évaluation finale.
    last_date = df["date"].max()
    validation_start = last_date - np.timedelta64(13, "D")

    train_df = df[df["date"] < validation_start].copy()

    if train_df.empty:
        raise ValueError("Pas assez de données pour entraîner le modèle.")

    # Limiter le nombre de lignes pour contrôler le temps de calcul.
    if len(train_df) > 150_000:
        train_df = train_df.sample(n=150_000, random_state=42)

    X_train = train_df[FEATURE_COLS]
    y_train = np.log1p(train_df["sales"].clip(lower=0))

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_COLS),
        ("categorical", categorical_pipeline, CATEGORICAL_COLS),
    ])

    model = Pipeline([
        ("preprocess", preprocess),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            max_depth=14,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        )),
    ])

    print("Entraînement du Random Forest...")
    model.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Modèle sauvegardé : {MODEL_PATH}")
    print(f"Lignes utilisées : {len(train_df)}")
    print(f"Dernière date des données : {last_date.date()}")
    print("Entraînement terminé.")


if __name__ == "__main__":
    main()
