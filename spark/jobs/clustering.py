"""
Clustering — K-Means su feature aggregate per compagnia (gen-apr 2025)

Feature (per carrier, 12 dimensioni):
  avg_dep_delay       media DEP_DELAY voli non cancellati
  avg_arr_delay       media ARR_DELAY voli non cancellati e non deviati
  cancellation_rate   % voli cancellati su totale
  avg_carrier_delay   media CARRIER_DELAY (NULL → 0, voli non cancellati)
  avg_weather_delay   media WEATHER_DELAY (NULL → 0, voli non cancellati)
  avg_nas_delay       media NAS_DELAY (NULL → 0, voli non cancellati)
  avg_security_delay  media SECURITY_DELAY (NULL → 0, voli non cancellati)
  avg_late_aircraft   media LATE_AIRCRAFT_DELAY (NULL → 0, voli non cancellati)
  std_dep_delay       deviazione standard DEP_DELAY voli non cancellati (imprevedibilità)
  on_time_rate        % voli con DEP_DELAY ≤ 0 tra i non cancellati (puntualità)
  diverted_rate       % voli deviati su totale
  avg_delay_recovery  media (DEP_DELAY − ARR_DELAY) voli completati (recupero in volo)

Pipeline:
  1. Aggregazione features per top-N carrier per numero voli
  2. Pipeline ML: VectorAssembler → StandardScaler (mean=0, std=1)
  3. Elbow + Silhouette (ClusteringEvaluator, squared euclidean) per k in [2..6]
     → k ottimale = max silhouette
  4. KMeans con k ottimale
  5. PCA 2D → scatter plot PNG (matplotlib)
  6. Export CSV locale + HDFS

Modalità:
  - Dev locale (Mac M1):  SPARK_MASTER=local[2]  (default)
  - Cluster / EC2:        SPARK_MASTER=spark://spark-master:7077
"""
import csv
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyspark.sql.functions import (
    col, avg, count, sum as spark_sum,
    when, coalesce, lit, round as spark_round, desc,
    stddev,
)
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# jobs/utils in testa a sys.path: utils.py/utils_output.py vivono lì (cartella
# senza __init__.py); senza questo `from utils import ...` non risolve.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))

from utils import build_spark_session

# ── Configurazione ────────────────────────────────────────────────────────────
SPARK_MASTER      = os.getenv("SPARK_MASTER", "spark://spark-master:7077")
HDFS_INPUT        = "hdfs://namenode:9000/sabd/processed/"
HDFS_OUTPUT       = "hdfs://namenode:9000/sabd/results/clustering/"
LOCAL_OUT_CSV     = "/opt/results/clustering.csv"
LOCAL_OUT_PNG_PCA = "/opt/results/clustering_pca.png"
LOCAL_OUT_PNG_ELB = "/opt/results/clustering_elbow.png"
TOP_N_CARRIERS    = 15
K_RANGE           = range(2, 7)

FEATURE_COLS = [
    "avg_dep_delay",
    "avg_arr_delay",
    "cancellation_rate",
    "avg_carrier_delay",
    "avg_weather_delay",
    "avg_nas_delay",
    "avg_security_delay",
    "avg_late_aircraft",
    "std_dep_delay",
    "on_time_rate",
    "diverted_rate",
    "avg_delay_recovery",
]

os.makedirs(os.path.dirname(LOCAL_OUT_CSV), exist_ok=True)


# ── 1. Caricamento e selezione carrier ───────────────────────────────────────
def load_and_select_carriers(spark):
    df_raw = spark.read.parquet(HDFS_INPUT)

    top_carriers = (
        df_raw.groupBy("OP_UNIQUE_CARRIER")
        .agg(count("*").alias("total_flights"))
        .orderBy(desc("total_flights"))
        .limit(TOP_N_CARRIERS)
        .select("OP_UNIQUE_CARRIER")
    )
    carrier_list = [r[0] for r in top_carriers.collect()]
    print(f"Carrier selezionati ({len(carrier_list)}): {carrier_list}")

    return df_raw.filter(col("OP_UNIQUE_CARRIER").isin(carrier_list))


# ── 2. Aggregazione feature ───────────────────────────────────────────────────
def aggregate_features(df):
    features_df = (
        df.groupBy("OP_UNIQUE_CARRIER")
        .agg(
            count("*").alias("total_flights"),
            spark_round(
                avg(when(col("CANCELLED") == 0, col("DEP_DELAY"))), 4
            ).alias("avg_dep_delay"),
            spark_round(
                avg(when((col("CANCELLED") == 0) & (col("DIVERTED") == 0), col("ARR_DELAY"))), 4
            ).alias("avg_arr_delay"),
            spark_round(
                spark_sum("CANCELLED") / count("*") * 100, 4
            ).alias("cancellation_rate"),
            spark_round(avg(when(col("CANCELLED") == 0, coalesce(col("CARRIER_DELAY"),       lit(0.0)))), 4).alias("avg_carrier_delay"),
            spark_round(avg(when(col("CANCELLED") == 0, coalesce(col("WEATHER_DELAY"),       lit(0.0)))), 4).alias("avg_weather_delay"),
            spark_round(avg(when(col("CANCELLED") == 0, coalesce(col("NAS_DELAY"),           lit(0.0)))), 4).alias("avg_nas_delay"),
            spark_round(avg(when(col("CANCELLED") == 0, coalesce(col("SECURITY_DELAY"),      lit(0.0)))), 4).alias("avg_security_delay"),
            spark_round(avg(when(col("CANCELLED") == 0, coalesce(col("LATE_AIRCRAFT_DELAY"), lit(0.0)))), 4).alias("avg_late_aircraft"),
            spark_round(
                stddev(when(col("CANCELLED") == 0, col("DEP_DELAY"))), 4
            ).alias("std_dep_delay"),
            spark_round(
                spark_sum(when((col("CANCELLED") == 0) & (col("DEP_DELAY") <= 0), lit(1.0)).otherwise(lit(0.0))) /
                spark_sum(when(col("CANCELLED") == 0, lit(1.0)).otherwise(lit(0.0))) * 100, 4
            ).alias("on_time_rate"),
            spark_round(
                spark_sum("DIVERTED") / count("*") * 100, 4
            ).alias("diverted_rate"),
            spark_round(
                avg(when((col("CANCELLED") == 0) & (col("DIVERTED") == 0), col("DEP_DELAY") - col("ARR_DELAY"))), 4
            ).alias("avg_delay_recovery"),
        )
    )
    features_df.cache()
    features_df.count()

    print("\nFeature aggregate per carrier:")
    features_df.orderBy(desc("total_flights")).show(truncate=False)

    return features_df


# ── 3. Pipeline ML: VectorAssembler → StandardScaler ─────────────────────────
def build_scaled_features(features_df):
    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_raw")
    scaler    = StandardScaler(inputCol="features_raw", outputCol="features",
                               withMean=True, withStd=True)

    # Stage di preprocessing incapsulati in una Pipeline (fit → PipelineModel)
    prep_pipeline = Pipeline(stages=[assembler, scaler])
    prep_model    = prep_pipeline.fit(features_df)
    return prep_model.transform(features_df)


# ── 4. Elbow + Silhouette per k in [2..6] ────────────────────────────────────
def find_best_k(df_scaled):
    # Silhouette di Spark ML (ClusteringEvaluator, squared euclidean) — come nel riferimento
    evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="prediction",
                                    metricName="silhouette",
                                    distanceMeasure="squaredEuclidean")

    wssse_list      = []
    silhouette_list = []

    print(f"\nRicerca k ottimale (range {list(K_RANGE)}):")
    for k in K_RANGE:
        km    = KMeans(featuresCol="features", k=k, seed=42, maxIter=30)
        model = km.fit(df_scaled)
        preds = model.transform(df_scaled)
        wssse = model.summary.trainingCost
        sil   = evaluator.evaluate(preds)
        wssse_list.append(wssse)
        silhouette_list.append(sil)
        print(f"  k={k}  WSSSE={wssse:.4f}  Silhouette={sil:.4f}")

    # Plot elbow + silhouette
    k_values = list(K_RANGE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(k_values, wssse_list, "bo-", markersize=7)
    ax1.set_xlabel("k")
    ax1.set_ylabel("WSSSE")
    ax1.set_title("Elbow Method — WSSSE")
    ax1.grid(True, alpha=0.3)
    for x, y in zip(k_values, wssse_list):
        ax1.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax2.plot(k_values, silhouette_list, "rs-", markersize=7)
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score (ClusteringEvaluator) per k")
    ax2.grid(True, alpha=0.3)
    for x, y in zip(k_values, silhouette_list):
        ax2.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)

    plt.tight_layout()
    plt.savefig(LOCAL_OUT_PNG_ELB, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nElbow/Silhouette plot: {LOCAL_OUT_PNG_ELB}")

    # k ottimale = max silhouette (ClusteringEvaluator)
    best_k = k_values[silhouette_list.index(max(silhouette_list))]
    print(f"k ottimale (max silhouette={max(silhouette_list):.4f}): k={best_k}")

    return best_k


# ── 5. KMeans finale con k ottimale ──────────────────────────────────────────
def run_final_kmeans(df_scaled, best_k):
    km_final  = KMeans(featuresCol="features", k=best_k, seed=42, maxIter=30)
    km_model  = km_final.fit(df_scaled)
    return km_model.transform(df_scaled)


# ── 6. PCA 2D → scatter plot ─────────────────────────────────────────────────
def pca_scatter_plot(df_scaled, df_result, best_k):
    pca       = PCA(k=2, inputCol="features", outputCol="pca_coords")
    pca_model = pca.fit(df_scaled)
    df_pca    = pca_model.transform(df_result)

    explained = pca_model.explainedVariance
    var_pc1   = float(explained[0]) * 100
    var_pc2   = float(explained[1]) * 100
    print(f"PCA varianza spiegata: PC1={var_pc1:.1f}%, PC2={var_pc2:.1f}%  "
          f"(totale: {var_pc1+var_pc2:.1f}%)")

    rows_pca = df_pca.select(
        "OP_UNIQUE_CARRIER", "pca_coords", "prediction"
    ).collect()

    carriers   = [r["OP_UNIQUE_CARRIER"] for r in rows_pca]
    coords     = np.array([[r["pca_coords"][0], r["pca_coords"][1]] for r in rows_pca])
    labels     = [r["prediction"] for r in rows_pca]

    colors = plt.cm.tab10(np.linspace(0, 0.9, best_k))
    fig, ax = plt.subplots(figsize=(10, 7))

    for cid in range(best_k):
        idx = [i for i, l in enumerate(labels) if l == cid]
        ax.scatter(
            coords[idx, 0], coords[idx, 1],
            c=[colors[cid]], s=140, label=f"Cluster {cid}",
            edgecolors="black", linewidths=0.6, zorder=3,
        )

    for i, carrier in enumerate(carriers):
        ax.annotate(
            carrier, (coords[i, 0], coords[i, 1]),
            textcoords="offset points", xytext=(6, 4), fontsize=10, fontweight="bold",
        )

    ax.set_xlabel(f"PC1 ({var_pc1:.1f}% varianza spiegata)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_pc2:.1f}% varianza spiegata)", fontsize=11)
    ax.set_title(
        f"K-Means Clustering delle compagnie aeree (k={best_k})\n"
        f"PCA 2D — varianza totale spiegata: {var_pc1+var_pc2:.1f}%",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOCAL_OUT_PNG_PCA, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PCA scatter plot: {LOCAL_OUT_PNG_PCA}")


# ── 7. Stampa risultati ───────────────────────────────────────────────────────
def print_results(df_result, best_k, elapsed):
    print(f"\n{'='*70}")
    print(f"CLUSTERING — RISULTATI  (k={best_k}, tempo: {elapsed:.2f}s)")
    print(f"{'='*70}")
    df_result.select(
        "OP_UNIQUE_CARRIER", "total_flights", *FEATURE_COLS, "prediction"
    ).orderBy("prediction", "OP_UNIQUE_CARRIER").show(20, truncate=False)


# ── 8. Export CSV locale + HDFS ──────────────────────────────────────────────
def export_results(df_result):
    out_df = df_result.select(
        "OP_UNIQUE_CARRIER", "total_flights", *FEATURE_COLS, "prediction"
    ).orderBy("prediction", "OP_UNIQUE_CARRIER")
    out_rows = out_df.collect()

    with open(LOCAL_OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(out_df.columns)
        for row in out_rows:
            writer.writerow([row[c] for c in out_df.columns])
    print(f"CSV locale: {LOCAL_OUT_CSV}")

    out_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(HDFS_OUTPUT)
    print(f"CSV HDFS:   {HDFS_OUTPUT}")


# ── Orchestrazione ────────────────────────────────────────────────────────────
def main():
    spark = build_spark_session(
        app_name="Clustering_KMeans",
        master=SPARK_MASTER,
    )

    # 1. Caricamento e selezione carrier
    df = load_and_select_carriers(spark)

    # 2-5. Aggregazione → scaling → ricerca k → KMeans finale (fasi cronometrate)
    t0          = time.time()
    features_df = aggregate_features(df)
    df_scaled   = build_scaled_features(features_df)
    best_k      = find_best_k(df_scaled)
    df_result   = run_final_kmeans(df_scaled, best_k)
    elapsed     = time.time() - t0

    # 6. PCA 2D → scatter plot
    pca_scatter_plot(df_scaled, df_result, best_k)

    # 7. Stampa risultati
    print_results(df_result, best_k, elapsed)

    # 8. Export CSV locale + HDFS
    export_results(df_result)

    print(f"\nTempo totale clustering: {elapsed:.2f}s")
    print(f"{'='*70}\n")

    spark._sc._jvm.System.exit(0)


if __name__ == "__main__":
    main()
