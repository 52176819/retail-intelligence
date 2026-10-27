
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import joblib
from sklearn.metrics import mean_squared_log_error

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "retail.duckdb"

st.set_page_config(
    page_title="Retail Intelligence",
    page_icon="📊",
    layout="wide",
)

# Style général
st.title("📊 Retail Intelligence")
st.caption("Analyse des ventes et aide à la décision")
st.markdown(
    "Explore les performances commerciales en filtrant les dates, "
    "les magasins et les familles de produits."
)

if not DB_PATH.exists():
    st.error(f"Base de données introuvable : {DB_PATH}")
    st.stop()


@st.cache_data
def load_data():
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        return con.execute("""
            SELECT
                date,
                store_nbr,
                family,
                sales,
                onpromotion,
                store_type,
                store_cluster,
                is_holiday,
                dow,
                dom,
                month,
                lag_14,
                lag_21,
                lag_28,
                roll_mean_7,
                roll_mean_27,
                roll_std_27,
                promo_mean_7
            FROM features.sales_features
            ORDER BY date
        """).fetchdf()



try:
    df = load_data()
except Exception as exc:
    st.error(f"Erreur pendant le chargement des données : {exc}")
    st.stop()

df["date"] = pd.to_datetime(df["date"])

# -------------------------
# FILTRES
# -------------------------
st.sidebar.header("🔎 Filtres")

min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.sidebar.date_input(
    "Période",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

stores = sorted(df["store_nbr"].dropna().unique().tolist())
selected_stores = st.sidebar.multiselect(
    "Magasins",
    options=stores,
    default=stores,
)

families = sorted(df["family"].dropna().unique().tolist())
selected_families = st.sidebar.multiselect(
    "Familles de produits",
    options=families,
    default=families,
)

if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    st.info("Sélectionne une date de début et une date de fin.")
    st.stop()

if not selected_stores or not selected_families:
    st.warning("Sélectionne au moins un magasin et une famille de produits.")
    st.stop()

start_date = pd.Timestamp(date_range[0])
end_date = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)

filtered = df[
    (df["date"] >= start_date)
    & (df["date"] < end_date)
    & (df["store_nbr"].isin(selected_stores))
    & (df["family"].isin(selected_families))
].copy()

if filtered.empty:
    st.warning("Aucune donnée pour les filtres sélectionnés.")
    st.stop()

# -------------------------
# INDICATEURS CLÉS (KPI)
# -------------------------
total_sales = filtered["sales"].sum()
mean_sales = filtered["sales"].mean()
active_stores = filtered["store_nbr"].nunique()
total_promotions = filtered["onpromotion"].sum()

st.subheader("📌 Indicateurs clés")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Ventes totales", f"{total_sales:,.0f}")
c2.metric("Ventes moyennes par ligne", f"{mean_sales:,.2f}")
c3.metric("Magasins actifs", f"{active_stores:,}")
c4.metric("Somme des promotions", f"{total_promotions:,.0f}")

st.caption(
    "Les ventes sont exprimées dans l'unité de la colonne sales. "
    "La somme de onpromotion ne représente pas nécessairement "
    "un nombre de promotions distinctes."
)

st.divider()

# -------------------------
# EVOLUTION QUOTIDIENNE
# -------------------------
st.subheader("📈 Évolution des ventes")

daily_sales = (
    filtered.groupby("date", as_index=False)["sales"]
    .sum()
    .sort_values("date")
)

fig_sales = px.line(
    daily_sales,
    x="date",
    y="sales",
    title="Ventes totales par jour",
    labels={"date": "Date", "sales": "Ventes"},
    template="plotly_dark",
)

fig_sales.update_layout(
    hovermode="x unified",
    xaxis_title="Date",
    yaxis_title="Ventes",
    margin=dict(l=20, r=20, t=60, b=20),
)

st.plotly_chart(fig_sales, width="stretch")

# -------------------------
# EVOLUTION MENSUELLE
# -------------------------
st.subheader("📅 Analyse mensuelle")

monthly_data = filtered.copy()
monthly_data["mois"] = (
    monthly_data["date"].dt.to_period("M").dt.to_timestamp()
)

monthly_sales = (
    monthly_data.groupby("mois", as_index=False)["sales"]
    .sum()
    .sort_values("mois")
)

fig_monthly = px.bar(
    monthly_sales,
    x="mois",
    y="sales",
    title="Ventes totales par mois",
    labels={"mois": "Mois", "sales": "Ventes"},
    template="plotly_dark",
)

fig_monthly.update_layout(
    xaxis_title="Mois",
    yaxis_title="Ventes",
    margin=dict(l=20, r=20, t=60, b=20),
)

st.plotly_chart(fig_monthly, width="stretch")

# -------------------------
# ANALYSE PAR FAMILLE ET MAGASIN
# -------------------------
left, right = st.columns(2)

with left:
    st.subheader("🛒 Ventes par famille")

    family_sales = (
        filtered.groupby("family", as_index=False)["sales"]
        .sum()
        .sort_values("sales", ascending=False)
        .head(15)
    )

    fig_family = px.bar(
        family_sales,
        x="sales",
        y="family",
        orientation="h",
        title="Top 15 des familles",
        labels={"sales": "Ventes", "family": "Famille"},
        template="plotly_dark",
    )

    fig_family.update_layout(
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(fig_family, width="stretch")

with right:
    st.subheader("🏬 Performances des magasins")

    store_sales = (
        filtered.groupby("store_nbr", as_index=False)["sales"]
        .sum()
        .sort_values("sales", ascending=False)
    )

    fig_store = px.bar(
        store_sales,
        x="store_nbr",
        y="sales",
        title="Ventes par magasin",
        labels={"store_nbr": "Magasin", "sales": "Ventes"},
        template="plotly_dark",
    )

    fig_store.update_layout(
        xaxis_title="Numéro du magasin",
        yaxis_title="Ventes",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(fig_store, width="stretch")

# -------------------------
# CLASSEMENT DES MAGASINS
# -------------------------
st.subheader("🏆 Top 10 des magasins")

top_stores = store_sales.head(10).copy()
top_stores = top_stores.reset_index(drop=True)
top_stores.index = top_stores.index + 1
top_stores.index.name = "Rang"

st.dataframe(
    top_stores,
    width="stretch",
)

# -------------------------
# PREVISIONS RANDOM FOREST
# -------------------------
st.divider()
st.header("Prévisions des ventes — Random Forest")

MODEL_PATH = ROOT / "models" / "random_forest_retail.joblib"

if not MODEL_PATH.exists():
    st.warning(
        "Le modèle sauvegardé est introuvable. "
        "Vérifie le fichier models/random_forest_retail.joblib."
    )
else:
    @st.cache_resource
    def load_model(path):
        return joblib.load(path)

    try:
        model = load_model(str(MODEL_PATH))

        # Les 14 dernières journées constituent la période de test.
        last_date = df["date"].max()
        validation_start = last_date - pd.Timedelta(days=13)

        test_df = df[
            (df["date"] >= validation_start)
            & (df["date"] <= last_date)
        ].copy()

        # Appliquer les filtres du tableau de bord.
        test_df = test_df[
            (test_df["date"] >= start_date)
            & (test_df["date"] < end_date)
            & (test_df["store_nbr"].isin(selected_stores))
            & (test_df["family"].isin(selected_families))
        ].copy()

        if test_df.empty:
            st.info(
                "Aucune donnée de la période de test ne correspond "
                "aux filtres actuels. Choisis une période qui inclut "
                f"les derniers jours disponibles, jusqu'au {last_date.date()}."
            )
        else:
            numeric_cols = [
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

            categorical_cols = [
                "store_nbr",
                "family",
                "store_type",
                "store_cluster",
            ]

            feature_cols = numeric_cols + categorical_cols

            X_test = test_df[feature_cols]
            predictions_log = model.predict(X_test)
            test_df["ventes_predites"] = np.maximum(
                np.expm1(predictions_log), 0
            )

            # Comparaison sur les ventes réelles de la période test.
            actual = test_df["sales"].clip(lower=0).to_numpy()
            predicted = test_df["ventes_predites"].to_numpy()

            rmsle = np.sqrt(
                mean_squared_log_error(actual, predicted)
            )

            # Comparaison avec la référence lag_14
            baseline_predictions = np.maximum(
                test_df["lag_14"].to_numpy(),
                0
                )

            rmsle_baseline = np.sqrt(
                mean_squared_log_error(actual, baseline_predictions)
                )

            improvement = (
                (rmsle_baseline - rmsle) / rmsle_baseline * 100
            if rmsle_baseline > 0
            else np.nan
            )

            

            total_actual = test_df["sales"].sum()
            total_predicted = test_df["ventes_predites"].sum()

            
            k1, k2, k3, k4 = st.columns(4)

            k1.metric("Ventes réelles", f"{total_actual:,.0f}")
            k2.metric("Ventes prédites", f"{total_predicted:,.0f}")
            k3.metric("RMSLE Random Forest", f"{rmsle:.4f}")
            k4.metric("RMSLE lag_14", f"{rmsle_baseline:.4f}")

            st.metric(
                "Amélioration par rapport à lag_14",
                f"{improvement:.2f} %" if np.isfinite(improvement) else "Non calculable",
                help="Une valeur positive signifie que le Random Forest obtient un RMSLE inférieur à la référence."
                )


            st.caption(
                f"Évaluation historique du {validation_start.date()} "
                f"au {last_date.date()}. "
                "Le RMSLE mesure l'écart entre les ventes réelles "
                "et les ventes prédites ; plus il est faible, mieux c'est."
            )

            daily_comparison = (
                test_df.groupby("date", as_index=False)[
                    ["sales", "ventes_predites"]
                ]
                .sum()
                .sort_values("date")
            )

            fig_predictions = px.line(
                daily_comparison,
                x="date",
                y=["sales", "ventes_predites"],
                title="Ventes réelles vs ventes prédites",
                labels={
                    "date": "Date",
                    "value": "Ventes",
                    "variable": "Série",
                    "sales": "Ventes réelles",
                    "ventes_predites": "Ventes prédites",
                },
                template="plotly_dark",
            )

            st.plotly_chart(
                fig_predictions,
                width="stretch",
            )

            st.subheader("Détail des prédictions")

            prediction_table = test_df[
                [
                    "date",
                    "store_nbr",
                    "family",
                    "sales",
                    "ventes_predites",
                ]
            ].copy()

            prediction_table = prediction_table.rename(
                columns={
                    "sales": "ventes_reelles",
                    "store_nbr": "magasin",
                    "family": "famille",
                }
            )

            st.dataframe(
                prediction_table.head(1000),
                width="stretch",
                hide_index=True,
            )

            predictions_csv = prediction_table.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                label="Télécharger les prédictions (CSV)",
                data=predictions_csv,
                file_name="retail_intelligence_predictions.csv",
                mime="text/csv",
            )

    except Exception as exc:
        st.error(
            f"Erreur lors du calcul des prédictions : {exc}"
        )


# -------------------------
# DONNEES ET EXPORT CSV
# -------------------------
st.divider()
st.subheader("📋 Données filtrées")

st.caption(
    f"{len(filtered):,} lignes correspondent aux filtres. "
    "Le tableau affiche les 1 000 premières lignes."
)

st.dataframe(
    filtered.head(1000),
    width="stretch",
    hide_index=True,
)

csv_data = filtered.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="⬇️ Télécharger toutes les données filtrées (CSV)",
    data=csv_data,
    file_name="retail_intelligence_donnees_filtrees.csv",
    mime="text/csv",
)
