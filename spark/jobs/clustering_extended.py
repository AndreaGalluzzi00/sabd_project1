"""
Clustering EXTENDED — K-Means sulle 8 feature di traccia + 4 feature aggiunte

Feature (12, per carrier):
  [8 base, come clustering_base.py]
  + std_dep_delay       deviazione standard DEP_DELAY (imprevedibilità)
  + on_time_rate        % voli con DEP_DELAY ≤ 0 tra i non cancellati (puntualità)
  + diverted_rate       % voli deviati su totale
  + avg_delay_recovery  media (DEP_DELAY − ARR_DELAY) voli completati (recupero in volo)

Le 4 feature aggiunte sono legittime ("...un insieme di feature... tra cui..."),
ma vanno giustificate: la correlation clustermap verifica se portano informazione
indipendente rispetto alle 8 base o se sono ridondanti (verdetto KEEP/REDUNDANT).

Pipeline: aggregazione → StandardScaler → elbow+silhouette (k∈[2..6]) → KMeans →
          PCA 2D scatter → profiling cluster (heatmap z-score + tabella delta)
          → correlation clustermap (giustificazione feature aggiunte).

Modalità:
  - Dev locale (Mac M1):  SPARK_MASTER=local[2]
  - Cluster / EC2:        SPARK_MASTER=spark://spark-master:7077
"""
import os
import sys
import time

# jobs/utils PRIMA di jobs/ in sys.path: così `utils`/`utils_clustering` risolvono
# ai file in jobs/utils/ e non al package-namespace jobs/utils/ (import altrimenti rotto).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))

from utils import build_spark_session
import utils_clustering as uc

# ── Configurazione ────────────────────────────────────────────────────────────
SPARK_MASTER  = os.getenv("SPARK_MASTER", "spark://spark-master:7077")
HDFS_INPUT    = "hdfs://namenode:9000/sabd/processed/"
HDFS_OUTPUT   = "hdfs://namenode:9000/sabd/results/clustering_extended/"
RESULTS_DIR   = "/opt/spark/jobs/results"
TOP_N_CARRIERS = 15
K_RANGE        = range(2, 7)

BASE_COLS = [
    "avg_dep_delay",
    "avg_arr_delay",
    "cancellation_rate",
    "avg_carrier_delay",
    "avg_weather_delay",
    "avg_nas_delay",
    "avg_security_delay",
    "avg_late_aircraft",
]
EXTRA_COLS = [
    "std_dep_delay",
    "on_time_rate",
    "diverted_rate",
    "avg_delay_recovery",
]
FEATURE_COLS = BASE_COLS + EXTRA_COLS

OUT_CSV         = f"{RESULTS_DIR}/clustering_extended.csv"
OUT_METRICS_CSV = f"{RESULTS_DIR}/clustering_extended_kmetrics.csv"
OUT_PROFILE_CSV = f"{RESULTS_DIR}/clustering_extended_profile.csv"
OUT_CORR_CSV    = f"{RESULTS_DIR}/clustering_extended_corr.csv"
OUT_PNG_ELBOW   = f"{RESULTS_DIR}/clustering_extended_elbow.png"
OUT_PNG_PCA     = f"{RESULTS_DIR}/clustering_extended_pca.png"
OUT_PNG_PROFILE = f"{RESULTS_DIR}/clustering_extended_profile.png"
OUT_PNG_CORR    = f"{RESULTS_DIR}/clustering_extended_corr.png"

os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    spark = build_spark_session(app_name="Clustering_Extended", master=SPARK_MASTER)

    df = uc.load_and_select_carriers(spark, HDFS_INPUT, TOP_N_CARRIERS)

    t0          = time.time()
    features_df = uc.aggregate_features(df, FEATURE_COLS)
    df_scaled   = uc.build_scaled_features(features_df, FEATURE_COLS)
    best_k, k_values, wssse_list, sil_list = uc.find_best_k(df_scaled, K_RANGE, OUT_PNG_ELBOW)
    df_result   = uc.run_final_kmeans(df_scaled, best_k)
    elapsed     = time.time() - t0

    uc.pca_scatter_plot(df_scaled, df_result, best_k, OUT_PNG_PCA, label="extended (12 feature)")
    uc.profile_clusters(features_df, df_result, FEATURE_COLS, best_k, OUT_PNG_PROFILE, OUT_PROFILE_CSV)

    # Analisi di collinearità: giustifica/contesta le 4 feature aggiunte
    uc.correlation_clustermap(features_df, FEATURE_COLS, BASE_COLS, EXTRA_COLS,
                              OUT_PNG_CORR, OUT_CORR_CSV)

    uc.print_results(df_result, FEATURE_COLS, best_k, elapsed, label="EXTENDED")
    uc.export_k_metrics(k_values, wssse_list, sil_list, OUT_METRICS_CSV)
    uc.export_results(df_result, FEATURE_COLS, OUT_CSV, HDFS_OUTPUT)

    print(f"\nTempo totale clustering (extended): {elapsed:.2f}s")
    print(f"{'='*70}\n")

    spark._sc._jvm.System.exit(0)


if __name__ == "__main__":
    main()
