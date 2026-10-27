
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "retail.duckdb"

con = duckdb.connect(str(DB_PATH), read_only=True)

try:
    # 1. Valeurs manquantes
    print("\n=== 1. VALEURS MANQUANTES ===")

    query = """
    SELECT
        COUNT(*) AS nombre_lignes,
        COUNT(*) FILTER (WHERE date IS NULL) AS date_null,
        COUNT(*) FILTER (WHERE store_nbr IS NULL) AS magasin_null,
        COUNT(*) FILTER (WHERE family IS NULL) AS famille_null,
        COUNT(*) FILTER (WHERE sales IS NULL) AS ventes_null,
        COUNT(*) FILTER (WHERE onpromotion IS NULL) AS promotion_null,
        COUNT(*) FILTER (WHERE is_holiday IS NULL) AS jour_ferie_null,
        COUNT(*) FILTER (WHERE lag_14 IS NULL) AS lag_14_null,
        COUNT(*) FILTER (WHERE lag_21 IS NULL) AS lag_21_null,
        COUNT(*) FILTER (WHERE lag_28 IS NULL) AS lag_28_null,
        COUNT(*) FILTER (WHERE roll_mean_7 IS NULL) AS moyenne_7_null,
        COUNT(*) FILTER (WHERE roll_mean_27 IS NULL) AS moyenne_27_null,
        COUNT(*) FILTER (WHERE roll_std_27 IS NULL) AS ecart_type_27_null,
        COUNT(*) FILTER (WHERE promo_mean_7 IS NULL) AS promo_mean_7_null
    FROM features.sales_features;
    """

    print(con.execute(query).fetchdf().to_string(index=False))

    # 2. Ventes nulles et négatives
    print("\n=== 2. VENTES NULLES ET NEGATIVES ===")

    query_ventes = """
    SELECT
        COUNT(*) FILTER (WHERE sales = 0) AS ventes_nulles,
        COUNT(*) FILTER (WHERE sales < 0) AS ventes_negatives,
        COUNT(*) FILTER (WHERE sales > 0) AS ventes_positives,
        MIN(sales) AS vente_minimale,
        MAX(sales) AS vente_maximale,
        AVG(sales) AS vente_moyenne
    FROM features.sales_features;
    """

    print(con.execute(query_ventes).fetchdf().to_string(index=False))

    # 3. Doublons sur la clé date-magasin-famille
    print("\n=== 3. DOUBLONS ===")

    query_doublons = """
    SELECT
        COUNT(*) AS groupes_dupliques,
        COALESCE(SUM(nb - 1), 0) AS lignes_en_trop
    FROM (
        SELECT COUNT(*) AS nb
        FROM features.sales_features
        GROUP BY date, store_nbr, family
        HAVING COUNT(*) > 1
    ) AS d;
    """

    print(con.execute(query_doublons).fetchdf().to_string(index=False))

    # 4. Nombre de dates, magasins et familles
    print("\n=== 4. COHERENCE DES DIMENSIONS ===")

    query_dimensions = """
    SELECT
        COUNT(DISTINCT date) AS nombre_dates,
        MIN(date) AS premiere_date,
        MAX(date) AS derniere_date,
        COUNT(DISTINCT store_nbr) AS nombre_magasins,
        COUNT(DISTINCT family) AS nombre_familles,
        COUNT(DISTINCT (date, store_nbr, family)) AS cles_uniques
    FROM features.sales_features;
    """

    print(con.execute(query_dimensions).fetchdf().to_string(index=False))

    # 5. Examiner les ventes les plus élevées
    print("\n=== 5. LES 10 VENTES LES PLUS ELEVEES ===")

    query_extremes = """
    SELECT date, store_nbr, family, sales
    FROM features.sales_features
    ORDER BY sales DESC
    LIMIT 10;
    """

    print(con.execute(query_extremes).fetchdf().to_string(index=False))

finally:
    con.close()
