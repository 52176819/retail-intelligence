
from pathlib import Path

import duckdb
import pandas as pd

# 1. Localiser la base de données
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "retail.duckdb"

con = duckdb.connect(str(DB_PATH), read_only=True)

try:
    # 2. Vue générale des données
    resume = con.execute("""
        SELECT
            COUNT(*) AS nombre_lignes,
            COUNT(DISTINCT store_nbr) AS nombre_magasins,
            COUNT(DISTINCT family) AS nombre_familles,
            MIN(date) AS premiere_date,
            MAX(date) AS derniere_date,
            SUM(sales) AS ventes_totales,
            AVG(sales) AS ventes_moyennes
        FROM features.sales_features
    """).fetchdf()

    print("\n=== 1. VUE GENERALE ===")
    print(resume.to_string(index=False))

    # 3. Familles de produits avec le plus de ventes
    familles = con.execute("""
        SELECT
            family,
            ROUND(SUM(sales), 2) AS ventes_totales,
            ROUND(AVG(sales), 2) AS ventes_moyennes
        FROM features.sales_features
        GROUP BY family
        ORDER BY ventes_totales DESC
        LIMIT 10
    """).fetchdf()

    print("\n=== 2. TOP 10 DES FAMILLES ===")
    print(familles.to_string(index=False))

    # 4. Magasins avec le plus de ventes
    magasins = con.execute("""
        SELECT
            store_nbr,
            ROUND(SUM(sales), 2) AS ventes_totales,
            ROUND(AVG(sales), 2) AS ventes_moyennes
        FROM features.sales_features
        GROUP BY store_nbr
        ORDER BY ventes_totales DESC
        LIMIT 10
    """).fetchdf()

    print("\n=== 3. TOP 10 DES MAGASINS ===")
    print(magasins.to_string(index=False))

    # 5. Evolution mensuelle des ventes
    mensuel = con.execute("""
        SELECT
            DATE_TRUNC('month', date) AS mois,
            ROUND(SUM(sales), 2) AS ventes_totales
        FROM features.sales_features
        GROUP BY DATE_TRUNC('month', date)
        ORDER BY mois
    """).fetchdf()

    print("\n=== 4. DERNIERS MOIS ===")
    print(mensuel.tail(12).to_string(index=False))

    # 6. Comparaison des jours avec et sans promotions
    promotions = con.execute("""
        SELECT
            CASE
                WHEN onpromotion > 0 THEN 'Avec promotion'
                ELSE 'Sans promotion'
            END AS type_jour,
            COUNT(*) AS nombre_observations,
            ROUND(AVG(sales), 2) AS ventes_moyennes,
            ROUND(SUM(sales), 2) AS ventes_totales
        FROM features.sales_features
        GROUP BY type_jour
        ORDER BY type_jour
    """).fetchdf()

    print("\n=== 5. PROMOTIONS ===")
    print(promotions.to_string(index=False))
    
    print("\n=== 3. DOUBLONS DATE-MAGASIN-FAMILLE ===")

    query_doublons = """
    SELECT
        COUNT(*) AS nombre_groupes_dupliques,
        COALESCE(SUM(nombre_lignes - 1), 0)
            AS lignes_supplementaires,
        COALESCE(MAX(nombre_lignes), 1)
            AS repetitions_maximales
    FROM (
        SELECT
            date,
            store_nbr,
            family,
            COUNT(*) AS nombre_lignes
        FROM features.sales_features
        GROUP BY date, store_nbr, family
        HAVING COUNT(*) > 1
    ) AS doublons;
    """

    result_doublons = con.execute(query_doublons).fetchdf()
    print(result_doublons.to_string(index=False))

    
    print("\n=== 4. VALEURS EXTREMES ===")

    query_extremes = """
    SELECT
        quantile_cont(sales, 0.25) AS q1,
        quantile_cont(sales, 0.50) AS mediane,
        quantile_cont(sales, 0.75) AS q3,
        quantile_cont(sales, 0.75)
          - quantile_cont(sales, 0.25) AS ecart_interquartile,
        quantile_cont(sales, 0.75)
          + 1.5 * (
              quantile_cont(sales, 0.75)
              - quantile_cont(sales, 0.25)
          ) AS seuil_extreme
    FROM features.sales_features;
    """

    stats = con.execute(query_extremes).fetchdf()
    print(stats.to_string(index=False))

    seuil = float(stats["seuil_extreme"].iloc[0])

    query_top = """
    SELECT date, store_nbr, family, sales
    FROM features.sales_features
    WHERE sales > ?
    ORDER BY sales DESC
    LIMIT 20;
    """

    print("\nLes 20 ventes les plus élevées au-dessus du seuil :")
    top = con.execute(query_top, [seuil]).fetchdf()
    print(top.to_string(index=False))

    print("\nNombre total de ventes au-dessus du seuil :")
    nombre = con.execute(
        "SELECT COUNT(*) FROM features.sales_features WHERE sales > ?",
        [seuil]
    ).fetchone()[0]
    print(nombre)

    
    print("\n=== 5. VALEURS EXTREMES PAR FAMILLE ===")

    query_familles = """
    WITH seuils AS (
        SELECT
            family,
            quantile_cont(sales, 0.25) AS q1,
            quantile_cont(sales, 0.75) AS q3
        FROM features.sales_features
        GROUP BY family
    ),
    bornes AS (
        SELECT
            family,
            q1,
            q3,
            q3 + 1.5 * (q3 - q1) AS seuil
        FROM seuils
    )
    SELECT
        f.family,
        COUNT(*) AS nombre_observations,
        ROUND(AVG(f.sales), 2) AS ventes_moyennes,
        ROUND(MAX(f.sales), 2) AS vente_maximale,
        ROUND(b.seuil, 2) AS seuil_extreme,
        COUNT(*) FILTER (
            WHERE f.sales > b.seuil
        ) AS nombre_valeurs_extremes
    FROM features.sales_features AS f
    JOIN bornes AS b USING (family)
    GROUP BY f.family, b.seuil
    ORDER BY nombre_valeurs_extremes DESC;
    """

    result_familles = con.execute(query_familles).fetchdf()
    print(result_familles.to_string(index=False))

    
    # Etape 6 : examiner les ventes autour d'une valeur extreme
    print("\n" + "=" * 60)
    print("ETAPE 6 : EXAMEN D'UNE VENTE EXTREME")
    print("=" * 60)

    query = """
        SELECT
            date,
            store_nbr,
            family,
            sales,
            onpromotion,
            lag_14,
            lag_21,
            lag_28,
            roll_mean_7,
            roll_mean_27
        FROM features.sales_features
        WHERE store_nbr = 2
          AND family = 'GROCERY I'
          AND date BETWEEN DATE '2016-04-25'
                       AND DATE '2016-05-10'
        ORDER BY date;
    """

    df = con.execute(query).fetchdf()
    print(df.to_string(index=False))

    
    # Etape 7 : identifier les tables de la base
    print("\n" + "=" * 60)
    print("ETAPE 7 : TABLES DISPONIBLES DANS DUCKDB")
    print("=" * 60)

    tables = con.execute("""
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_type = 'BASE TABLE'
        ORDER BY table_schema, table_name;
    """).fetchdf()

    print(tables.to_string(index=False))

    
    # Etape 8 : verifier la vente dans les donnees sources
    print("\n" + "=" * 60)
    print("ETAPE 8 : VERIFICATION DANS RAW.TRAIN")
    print("=" * 60)

    query = """
        SELECT
            date,
            store_nbr,
            family,
            sales,
            onpromotion
        FROM raw.train
        WHERE store_nbr = 2
          AND family = 'GROCERY I'
          AND date BETWEEN DATE '2016-04-25'
                       AND DATE '2016-05-10'
        ORDER BY date;
    """

    df_source = con.execute(query).fetchdf()
    print(df_source.to_string(index=False))

    
    # Etape 9 : verifier les ventes de toutes les familles du magasin 2
    print("\n" + "=" * 60)
    print("ETAPE 9 : VENTES DU MAGASIN 2 AUTOUR DU 2 MAI 2016")
    print("=" * 60)

    query = """
        SELECT
            date,
            family,
            sales,
            onpromotion
        FROM raw.train
        WHERE store_nbr = 2
          AND date BETWEEN DATE '2016-04-30'
                       AND DATE '2016-05-04'
        ORDER BY date, sales DESC;
    """

    df_magasin = con.execute(query).fetchdf()
    print(df_magasin.to_string(index=False))

    
    # Etape 10 : comparer GROCERY I entre tous les magasins
    print("\n" + "=" * 60)
    print("ETAPE 10 : GROCERY I DANS TOUS LES MAGASINS")
    print("=" * 60)

    query = """
        SELECT
            date,
            store_nbr,
            family,
            sales,
            onpromotion
        FROM raw.train
        WHERE family = 'GROCERY I'
          AND date BETWEEN DATE '2016-04-30'
                       AND DATE '2016-05-04'
        ORDER BY date, sales DESC;
    """

    df_comparaison = con.execute(query).fetchdf()
    print(df_comparaison.to_string(index=False))

    
    # Etape 11 : verifier les evenements et les transactions
    print("\n" + "=" * 60)
    print("ETAPE 11A : EVENEMENTS AUTOUR DU 2 MAI 2016")
    print("=" * 60)

    query = """
        SELECT *
        FROM raw.holidays_events
        WHERE date BETWEEN DATE '2016-04-30'
                       AND DATE '2016-05-04'
        ORDER BY date;
    """

    df_evenements = con.execute(query).fetchdf()
    print(df_evenements.to_string(index=False))

    print("\n" + "=" * 60)
    print("ETAPE 11B : TRANSACTIONS DU MAGASIN 2")
    print("=" * 60)

    query = """
        SELECT
            date,
            store_nbr,
            transactions
        FROM raw.transactions
        WHERE store_nbr = 2
          AND date BETWEEN DATE '2016-04-25'
                       AND DATE '2016-05-10'
        ORDER BY date;
    """

    df_transactions = con.execute(query).fetchdf()
    print(df_transactions.to_string(index=False))

    
    # Etape 12 : verifier les ventes et promotions dans les donnees sources
    print("\n" + "=" * 60)
    print("ETAPE 12 : VENTES ET PROMOTIONS DANS RAW.TRAIN")
    print("=" * 60)

    query = """
        SELECT
            date,
            store_nbr,
            family,
            sales,
            onpromotion
        FROM raw.train
        WHERE store_nbr = 2
          AND family = 'GROCERY I'
          AND date BETWEEN DATE '2016-04-25'
                       AND DATE '2016-05-10'
        ORDER BY date;
    """

    df_source = con.execute(query).fetchdf()
    print(df_source.to_string(index=False))

    
    # Etape 13 : comparer la meme periode sur les autres annees
    print("\n" + "=" * 60)
    print("ETAPE 13 : COMPARAISON AVEC LES AUTRES ANNEES")
    print("=" * 60)

    query = """
        SELECT
            date,
            sales,
            onpromotion
        FROM raw.train
        WHERE store_nbr = 2
          AND family = 'GROCERY I'
          AND EXTRACT(MONTH FROM date) = 4
          AND EXTRACT(DAY FROM date) BETWEEN 20 AND 30

        UNION ALL

        SELECT
            date,
            sales,
            onpromotion
        FROM raw.train
        WHERE store_nbr = 2
          AND family = 'GROCERY I'
          AND EXTRACT(MONTH FROM date) = 5
          AND EXTRACT(DAY FROM date) BETWEEN 1 AND 12

        ORDER BY date;
    """

    df_comparaison = con.execute(query).fetchdf()
    print(df_comparaison.to_string(index=False))

    
    # Etape 14 : identifier les ventes les plus elevees sur toute la periode
    print("\n" + "=" * 60)
    print("ETAPE 14 : LES 15 PLUS GRANDES VENTES")
    print("=" * 60)

    query = """
        SELECT
            date,
            sales,
            onpromotion
        FROM raw.train
        WHERE store_nbr = 2
          AND family = 'GROCERY I'
        ORDER BY sales DESC
        LIMIT 15;
    """

    df_top_ventes = con.execute(query).fetchdf()
    print(df_top_ventes.to_string(index=False))

    
    # Etape 15 : comparer les pics entre les magasins
    print("\n" + "=" * 60)
    print("ETAPE 15 : COMPARAISON DES MAGASINS AUX DATES SUSPECTES")
    print("=" * 60)

    query = """
        SELECT
            date,
            store_nbr,
            sales,
            onpromotion
        FROM raw.train
        WHERE family = 'GROCERY I'
          AND date IN (
              DATE '2016-04-21',
              DATE '2016-04-26',
              DATE '2016-05-02',
              DATE '2016-05-09'
          )
        ORDER BY date, sales DESC;
    """

    df_magasins = con.execute(query).fetchdf()
    print(df_magasins.to_string(index=False))

    
    # Etape 16 : verifier les variables explicatives
    print("\n" + "=" * 60)
    print("ETAPE 16 : CONTROLE DES FEATURES")
    print("=" * 60)

    query = """
        SELECT
            COUNT(*) AS total_lignes,
            COUNT(*) FILTER (WHERE lag_14 IS NULL) AS lag_14_manquants,
            COUNT(*) FILTER (WHERE lag_21 IS NULL) AS lag_21_manquants,
            COUNT(*) FILTER (WHERE lag_28 IS NULL) AS lag_28_manquants,
            COUNT(*) FILTER (WHERE roll_mean_7 IS NULL) AS roll_mean_7_manquants,
            COUNT(*) FILTER (WHERE roll_mean_27 IS NULL) AS roll_mean_27_manquants,
            COUNT(*) FILTER (WHERE roll_std_27 IS NULL) AS roll_std_27_manquants,
            COUNT(*) FILTER (WHERE promo_mean_7 IS NULL) AS promo_mean_7_manquants,

            COUNT(*) FILTER (
                WHERE lag_14 < 0
                   OR lag_21 < 0
                   OR lag_28 < 0
                   OR roll_mean_7 < 0
                   OR roll_mean_27 < 0
                   OR roll_std_27 < 0
                   OR promo_mean_7 < 0
            ) AS lignes_avec_features_negatives

        FROM features.sales_features;
    """

    df_features = con.execute(query).fetchdf()
    print(df_features.to_string(index=False))






finally:
    con.close()
