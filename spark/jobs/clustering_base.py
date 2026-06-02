"""
Clustering BASE — K-Means sulle sole feature richieste dalla traccia (gen-apr 2025)

Feature (8, per carrier) — esattamente quelle elencate nella traccia:
  avg_dep_delay       media DEP_DELAY voli non cancellati
  avg_arr_delay       media ARR_DELAY voli non cancellati e non deviati
  cancellation_rate   % voli cancellati su totale
  avg_carrier_delay   media CARRIER_DELAY  (NULL → 0, voli non cancellati)
  avg_weather_delay   media WEATHER_DELAY  (NULL → 0, voli non cancellati)
  avg_nas_delay       media NAS_DELAY      (NULL → 0, voli non cancellati)
  avg_security_delay  media SECURITY_DELAY (NULL → 0, voli non cancellati)
  avg_late_aircraft   media LATE_AIRCRAFT_DELAY (NULL → 0, voli non cancellati)

Pipeline: aggregazione → StandardScaler → elbow+silhouette (k∈[2..6]) → KMeans →
          PCA 2D scatter → profiling cluster (heatmap z-score + tabella delta).

È la controparte "minimale" di clustering_extended.py: stessa pipeline, senza le
4 feature aggiunte. Serve per confrontare best_k / silhouette / WSSSE /
interpretabilità con la versione estesa.

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
HDFS_OUTPUT   = "hdfs://namenode:9000/sabd/results/clustering_base/"
RESULTS_DIR   = "/opt/spark/jobs/results"
TOP_N_CARRIERS = 15
K_RANGE        = range(2, 7)

FEATURE_COLS = [
    "avg_dep_delay",
    "avg_arr_delay",
    "cancellation_rate",
    "avg_carrier_delay",
    "avg_weather_delay",
    "avg_nas_delay",
    "avg_security_delay",
    "avg_late_aircraft",
]

OUT_CSV         = f"{RESULTS_DIR}/clustering_base.csv"
OUT_METRICS_CSV = f"{RESULTS_DIR}/clustering_base_kmetrics.csv"
OUT_PROFILE_CSV = f"{RESULTS_DIR}/clustering_base_profile.csv"
OUT_PNG_ELBOW   = f"{RESULTS_DIR}/clustering_base_elbow.png"
OUT_PNG_PCA     = f"{RESULTS_DIR}/clustering_base_pca.png"
OUT_PNG_PROFILE = f"{RESULTS_DIR}/clustering_base_profile.png"

os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    spark = build_spark_session(app_name="Clustering_Base", master=SPARK_MASTER)

    df = uc.load_and_select_carriers(spark, HDFS_INPUT, TOP_N_CARRIERS)

    t0          = time.time()
    features_df = uc.aggregate_features(df, FEATURE_COLS)
    df_scaled   = uc.build_scaled_features(features_df, FEATURE_COLS)
    best_k, k_values, wssse_list, sil_list = uc.find_best_k(df_scaled, K_RANGE, OUT_PNG_ELBOW)
    df_result   = uc.run_final_kmeans(df_scaled, best_k)
    elapsed     = time.time() - t0

    uc.pca_scatter_plot(df_scaled, df_result, best_k, OUT_PNG_PCA, label="base (8 feature)")
    uc.profile_clusters(features_df, df_result, FEATURE_COLS, best_k, OUT_PNG_PROFILE, OUT_PROFILE_CSV)

    uc.print_results(df_result, FEATURE_COLS, best_k, elapsed, label="BASE")
    uc.export_k_metrics(k_values, wssse_list, sil_list, OUT_METRICS_CSV)
    uc.export_results(df_result, FEATURE_COLS, OUT_CSV, HDFS_OUTPUT)

    print(f"\nTempo totale clustering (base): {elapsed:.2f}s")
    print(f"{'='*70}\n")

    spark._sc._jvm.System.exit(0)


if __name__ == "__main__":
    main()
