"""
Clustering Study — BASE vs EXTENDED

Esegue in un unico script i due studi di clustering:

1. BASE:
   usa solo le 8 feature richieste dalla traccia.

2. EXTENDED:
   usa le 8 feature richieste dalla traccia
   + 4 feature aggiunte (poi sottoposte a feature selection per correlazione).

L'obiettivo è confrontare i due studi mantenendo identici:
- dataset
- top-N carrier
- pipeline
- scaling
- range di k
- algoritmo KMeans

L'unica cosa che cambia è il set di feature.
"""

import csv
import os
import sys
import time

# jobs/utils prima di jobs/ in sys.path
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils")
)

from utils import build_spark_session
import utils_clustering as uc


# ── Configurazione generale ──────────────────────────────────────────────────

SPARK_MASTER = os.getenv("SPARK_MASTER", "spark://spark-master:7077")

HDFS_INPUT = "hdfs://namenode:9000/sabd/processed/"
HDFS_OUTPUT_ROOT = "hdfs://namenode:9000/sabd/results"

RESULTS_DIR = "/opt/results"
TOP_N_CARRIERS = 15
K_RANGE = range(2, 7)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Feature ──────────────────────────────────────────────────────────────────

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
]


# ── Studi da eseguire ─────────────────────────────────────────────────────────

STUDIES = [
    {
        "name": "base",
        "label": "BASE",
        "feature_cols": BASE_COLS,
        "hdfs_output": f"{HDFS_OUTPUT_ROOT}/clustering_base/",
        "out_csv": f"{RESULTS_DIR}/clustering_base.csv",
        "out_metrics_csv": f"{RESULTS_DIR}/clustering_base_kmetrics.csv",
        "out_profile_csv": f"{RESULTS_DIR}/clustering_base_profile.csv",
        "out_png_elbow": f"{RESULTS_DIR}/clustering_base_elbow.png",
        "out_png_pca": f"{RESULTS_DIR}/clustering_base_pca.png",
        "out_png_profile": f"{RESULTS_DIR}/clustering_base_profile.png",
        "run_correlation": False,
        "out_corr_csv": None,
        "out_png_corr": None,
    },
    {
        "name": "extended",
        "label": "EXTENDED",
        "feature_cols": BASE_COLS + EXTRA_COLS,
        "hdfs_output": f"{HDFS_OUTPUT_ROOT}/clustering_extended/",
        "out_csv": f"{RESULTS_DIR}/clustering_extended.csv",
        "out_metrics_csv": f"{RESULTS_DIR}/clustering_extended_kmetrics.csv",
        "out_profile_csv": f"{RESULTS_DIR}/clustering_extended_profile.csv",
        "out_png_elbow": f"{RESULTS_DIR}/clustering_extended_elbow.png",
        "out_png_pca": f"{RESULTS_DIR}/clustering_extended_pca.png",
        "out_png_profile": f"{RESULTS_DIR}/clustering_extended_profile.png",
        "run_correlation": True,
        "out_corr_csv": f"{RESULTS_DIR}/clustering_extended_corr.csv",
        "out_png_corr": f"{RESULTS_DIR}/clustering_extended_corr.png",
    },
]


def run_study(df, study):

    name = study["name"]
    label = study["label"]
    feature_cols = study["feature_cols"]

    print("\n" + "=" * 90)
    print(f"CLUSTERING {label}")
    print("=" * 90)
    print(f"Numero feature: {len(feature_cols)}")
    print("Feature usate:")
    for c in feature_cols:
        print(f"  - {c}")

    t0 = time.time()

    # 1. Aggregazione per carrier
    features_df = uc.aggregate_features(df, feature_cols)

    # 1b. Feature selection tramite correlazione
    if study["run_correlation"]:
        features_to_remove = uc.correlation_clustermap(
            features_df,
            feature_cols,
            BASE_COLS,
            EXTRA_COLS,
            study["out_png_corr"],
            study["out_corr_csv"],
        )

        # 1c. Rimozione effettiva delle feature ridondanti
        feature_cols = uc.remove_features(
            feature_cols,
            features_to_remove,
        )

    # 2. Scaling
    df_scaled = uc.build_scaled_features(features_df, feature_cols)

    # 3. Ricerca miglior k
    best_k, k_values, wssse_list, sil_list = uc.find_best_k(
        df_scaled,
        K_RANGE,
        study["out_png_elbow"],
    )

    # 4. KMeans finale
    df_result = uc.run_final_kmeans(
        df_scaled,
        best_k,
    )

    elapsed = time.time() - t0

    # 5. PCA
    uc.pca_scatter_plot(
        df_scaled,
        df_result,
        best_k,
        study["out_png_pca"],
        label=f"{name} ({len(feature_cols)} feature)",
    )

    # 6. Profiling cluster
    uc.profile_clusters(
        features_df,
        df_result,
        feature_cols,
        best_k,
        study["out_png_profile"],
        study["out_profile_csv"],
    )

    # 8. Stampa risultati
    uc.print_results(
        df_result,
        feature_cols,
        best_k,
        elapsed,
        label=label,
    )

    # 9. Export metriche k
    uc.export_k_metrics(
        k_values,
        wssse_list,
        sil_list,
        study["out_metrics_csv"],
    )

    # 10. Export risultati finali
    uc.export_results(
        df_result,
        feature_cols,
        study["out_csv"],
        study["hdfs_output"],
    )

    best_index = k_values.index(best_k)

    summary = {
        "study": name,
        "label": label,
        "n_features": len(feature_cols),
        "features": ",".join(feature_cols),
        "best_k": best_k,
        "best_silhouette": round(float(sil_list[best_index]), 4),
        "best_wssse": round(float(wssse_list[best_index]), 4),
        "elapsed_seconds": round(float(elapsed), 4),
        "local_csv": study["out_csv"],
        "metrics_csv": study["out_metrics_csv"],
        "hdfs_output": study["hdfs_output"],
    }

    print("\n" + "-" * 90)
    print(f"Fine studio {label}")
    print(f"Tempo: {elapsed:.2f}s")
    print(f"Best k: {best_k}")
    print(f"Best silhouette: {summary['best_silhouette']}")
    print(f"Best WSSSE: {summary['best_wssse']}")
    print("-" * 90)

    return summary


def export_comparison_summary(rows, out_csv):
    """
    Esporta il confronto finale BASE vs EXTENDED.
    """

    fieldnames = [
        "study",
        "label",
        "n_features",
        "features",
        "best_k",
        "best_silhouette",
        "best_wssse",
        "elapsed_seconds",
        "local_csv",
        "metrics_csv",
        "hdfs_output",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nComparison summary: {out_csv}")


def print_comparison_summary(rows):
    """
    Stampa confronto finale tra BASE ed EXTENDED.
    """

    print("\n" + "=" * 90)
    print("CONFRONTO FINALE — BASE vs EXTENDED")
    print("=" * 90)

    for r in rows:
        print(f"\nStudio: {r['label']}")
        print(f"N feature: {r['n_features']}")
        print(f"Best k: {r['best_k']}")
        print(f"Best silhouette: {r['best_silhouette']}")
        print(f"Best WSSSE: {r['best_wssse']}")
        print(f"Tempo: {r['elapsed_seconds']}s")
        print(f"CSV locale: {r['local_csv']}")
        print(f"HDFS output: {r['hdfs_output']}")

    if len(rows) == 2:
        base = rows[0]
        extended = rows[1]

        delta_sil = extended["best_silhouette"] - base["best_silhouette"]
        delta_wssse = extended["best_wssse"] - base["best_wssse"]

        print("\n" + "-" * 90)
        print("DELTA EXTENDED - BASE")
        print("-" * 90)
        print(f"Delta silhouette: {delta_sil:+.4f}")
        print(f"Delta WSSSE: {delta_wssse:+.4f}")

        print(
            "\nNota: il WSSSE non è direttamente confrontabile in modo forte "
            "tra BASE ed EXTENDED, perché cambia lo spazio delle feature. "
            "Per il confronto guarda soprattutto silhouette, PCA e profiling."
        )

    print("=" * 90 + "\n")


def main():
    spark = build_spark_session(app_name="Clustering_Study_Base_vs_Extended",master=SPARK_MASTER)
    print("CLUSTERING STUDY — BASE vs EXTENDED")
    print(f"Input HDFS: {HDFS_INPUT}")
    print(f"Top N carriers: {TOP_N_CARRIERS}")
    print(f"K range: {list(K_RANGE)}")
    print(f"Results dir: {RESULTS_DIR}")
    print("=" * 90)

    # Carico e filtro una sola volta.
    # Così BASE ed EXTENDED lavorano sugli stessi identici carrier.
    df_base = spark.read.parquet(HDFS_INPUT)
    df = uc.load_and_select_carriers(df_base, TOP_N_CARRIERS)

    df.cache()
    df.count()

    comparison_rows = []

    for study in STUDIES:
        row = run_study(df,study)
        comparison_rows.append(row)

    comparison_csv = f"{RESULTS_DIR}/clustering_comparison_summary.csv"

    export_comparison_summary(
        comparison_rows,
        comparison_csv,
    )

    print_comparison_summary(
        comparison_rows,
    )

    df.unpersist()

    print("\nFine clustering study.")
    print("=" * 90)

    spark.stop()


if __name__ == "__main__":
    main()