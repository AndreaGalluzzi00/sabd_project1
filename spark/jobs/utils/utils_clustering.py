import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyspark.sql.functions import (
    col, avg, count, sum as spark_sum,
    when, coalesce, lit, round as spark_round, desc,
    stddev, mean as spark_mean,
)
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


# Registry delle feature
def feature_expressions():
    # cause di ritardo (NULL -> 0) sui soli voli non cancellati, riusate per la
    # composizione "controllable" (carrier + late aircraft) vs totale delle 5 cause.

    return {
        # feature richieste dalla traccia
        "avg_dep_delay": spark_round(
            avg(when(col("CANCELLED") == 0, col("DEP_DELAY"))), 4),
        "avg_arr_delay": spark_round(
            avg(when((col("CANCELLED") == 0) & (col("DIVERTED") == 0), col("ARR_DELAY"))), 4),
        "cancellation_rate": spark_round(
            spark_sum("CANCELLED") / count("*") * 100, 4),
        "avg_carrier_delay": spark_round(
            avg(when(col("CANCELLED") == 0, coalesce(col("CARRIER_DELAY"), lit(0.0)))), 4),
        "avg_weather_delay": spark_round(
            avg(when(col("CANCELLED") == 0, coalesce(col("WEATHER_DELAY"), lit(0.0)))), 4),
        "avg_nas_delay": spark_round(
            avg(when(col("CANCELLED") == 0, coalesce(col("NAS_DELAY"), lit(0.0)))), 4),
        "avg_security_delay": spark_round(
            avg(when(col("CANCELLED") == 0, coalesce(col("SECURITY_DELAY"), lit(0.0)))), 4),
        "avg_late_aircraft": spark_round(
            avg(when(col("CANCELLED") == 0, coalesce(col("LATE_AIRCRAFT_DELAY"), lit(0.0)))), 4),
        #3 feature aggiunte (solo versione extended)
        "std_dep_delay": spark_round(
            stddev(when(col("CANCELLED") == 0, col("DEP_DELAY"))), 4),
        "on_time_rate": spark_round(
            spark_sum(when((col("CANCELLED") == 0) & (col("DEP_DELAY") <= 0), lit(1.0)).otherwise(lit(0.0))) /
            spark_sum(when(col("CANCELLED") == 0, lit(1.0)).otherwise(lit(0.0))) * 100, 4),
        "diverted_rate": spark_round(
            spark_sum("DIVERTED") / count("*") * 100, 4),

    }


#  Caricamento e selezione top carrier
def load_and_select_carriers(df_raw, top_n):

    top_carriers = (
        df_raw.groupBy("OP_UNIQUE_CARRIER")
        .agg(count("*").alias("total_flights"))
        .orderBy(desc("total_flights"))
        .limit(top_n)
        .select("OP_UNIQUE_CARRIER")
    )
    carrier_list = [r[0] for r in top_carriers.collect()]
    print(f"Carrier selezionati ({len(carrier_list)}): {carrier_list}")

    return df_raw.filter(col("OP_UNIQUE_CARRIER").isin(carrier_list))


#  Aggregazione delle sole feature richieste
def aggregate_features(df, feature_cols):
    exprs = feature_expressions()
    agg_exprs = [count("*").alias("total_flights")] + [exprs[c].alias(c) for c in feature_cols]

    features_df = df.groupBy("OP_UNIQUE_CARRIER").agg(*agg_exprs)
    features_df.cache()
    features_df.count()

    print("\nFeature aggregate per carrier:")
    features_df.orderBy(desc("total_flights")).show(truncate=False)

    return features_df


# Pipeline ML: VectorAssembler → StandardScaler
def build_scaled_features(features_df, feature_cols):
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scaler    = StandardScaler(inputCol="features_raw", outputCol="features",
                               withMean=True, withStd=True)
    prep_model = Pipeline(stages=[assembler, scaler]).fit(features_df)
    return prep_model.transform(features_df)


def evaluate_k_values(df_scaled, k_range):
    """
    Esegue K-Means per ciascun valore di k e calcola WSSSE e silhouette.
    """
    evaluator = ClusteringEvaluator(
        featuresCol="features",
        predictionCol="prediction",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )

    k_values = list(k_range)
    wssse_list = []
    silhouette_list = []

    print(f"\nRicerca k ottimale (range {k_values}):")

    for k in k_values:
        km = KMeans(
            featuresCol="features",
            k=k,
            seed=42,
            maxIter=100,
            initSteps=10,
        )

        model = km.fit(df_scaled)
        preds = model.transform(df_scaled)

        wssse = model.summary.trainingCost
        silhouette = evaluator.evaluate(preds)

        wssse_list.append(wssse)
        silhouette_list.append(silhouette)

        print(f"  k={k}  WSSSE={wssse:.4f}  Silhouette={silhouette:.4f}")

    return k_values, wssse_list, silhouette_list



def select_best_k(k_values, silhouette_list):
    """
    Seleziona il valore di k che massimizza il silhouette score.
    """
    best_index = silhouette_list.index(max(silhouette_list))
    best_k = k_values[best_index]

    print(f"k ottimale (max silhouette={silhouette_list[best_index]:.4f}): "
          f"k={best_k}")

    return best_k


def plot_k_metrics(k_values, wssse_list, silhouette_list, out_png):
    """
    Genera il grafico con elbow method e silhouette score.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Elbow method - WSSSE
    ax1.plot(k_values, wssse_list, "bo-", markersize=7)
    ax1.set_xlabel("k")
    ax1.set_ylabel("WSSSE")
    ax1.set_title("Elbow Method — WSSSE")
    ax1.grid(True, alpha=0.3)

    for x, y in zip(k_values, wssse_list):
        ax1.annotate(
            f"{y:.1f}",
            (x, y),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    # Silhouette score
    ax2.plot(k_values, silhouette_list, "rs-", markersize=7)
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score per k")
    ax2.grid(True, alpha=0.3)

    for x, y in zip(k_values, silhouette_list):
        ax2.annotate(
            f"{y:.3f}",
            (x, y),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nElbow/Silhouette plot: {out_png}")

# Elbow + Silhouette per k nel range
def find_best_k(df_scaled, k_range, out_png_elbow):

    k_values, wssse_list, silhouette_list = evaluate_k_values(df_scaled,k_range)

    plot_k_metrics(k_values,wssse_list,silhouette_list,out_png_elbow)

    best_k = select_best_k(k_values,silhouette_list)

    return best_k, k_values, wssse_list, silhouette_list


# KMeans finale con k ottimale
def run_final_kmeans(df_scaled, best_k):
    km_model = KMeans(featuresCol="features", k=best_k, seed=42, maxIter=100, initSteps=10).fit(df_scaled)
    return km_model.transform(df_scaled)


# PCA 2D scatter plot
def pca_scatter_plot(df_scaled, df_result, best_k, out_png_pca, label=""):
    pca_model = PCA(k=2, inputCol="features", outputCol="pca_coords").fit(df_scaled)
    df_pca    = pca_model.transform(df_result)

    explained = pca_model.explainedVariance
    var_pc1   = float(explained[0]) * 100
    var_pc2   = float(explained[1]) * 100
    print(f"PCA varianza spiegata: PC1={var_pc1:.1f}%, PC2={var_pc2:.1f}%  "
          f"(totale: {var_pc1+var_pc2:.1f}%)")

    rows_pca = df_pca.select("OP_UNIQUE_CARRIER", "pca_coords", "prediction").collect()
    carriers = [r["OP_UNIQUE_CARRIER"] for r in rows_pca]
    coords   = np.array([[r["pca_coords"][0], r["pca_coords"][1]] for r in rows_pca])
    labels   = [r["prediction"] for r in rows_pca]

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
    title = f"K-Means delle compagnie aeree (k={best_k})"
    if label:
        title += f" — {label}"
    ax.set_title(f"{title}\nPCA 2D — varianza totale spiegata: {var_pc1+var_pc2:.1f}%", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png_pca, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PCA scatter plot: {out_png_pca}")


# Profiling dei cluster: heatmap z-score + tabella delta
def profile_clusters(features_df, df_result, feature_cols, best_k, out_png, out_csv):
    g = features_df.select(
        *[spark_mean(c).alias(f"{c}__mean") for c in feature_cols],
        *[stddev(c).alias(f"{c}__std") for c in feature_cols],
    ).collect()[0]
    global_mean = np.array([g[f"{c}__mean"] for c in feature_cols], dtype=float)
    global_std  = np.array([g[f"{c}__std"]  for c in feature_cols], dtype=float)
    global_std  = np.where((global_std == 0) | np.isnan(global_std), 1.0, global_std)

    rows = (
        df_result.groupBy("prediction")
        .agg(count("*").alias("n_carriers"), *[spark_mean(c).alias(c) for c in feature_cols])
        .orderBy("prediction")
        .collect()
    )
    cluster_ids = [r["prediction"] for r in rows]
    sizes       = [r["n_carriers"] for r in rows]
    means       = np.array([[r[c] for c in feature_cols] for r in rows], dtype=float)  # (k, n_feat)

    delta  = means - global_mean
    zscore = delta / global_std

    # --- heatmap z-score ---
    fig, ax = plt.subplots(figsize=(max(8, len(feature_cols) * 0.95),
                                    max(3.5, len(cluster_ids) * 0.8) + 1.5))
    vmax = float(np.nanmax(np.abs(zscore))) or 1.0
    im = ax.imshow(zscore, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cluster_ids)))
    ax.set_yticklabels([f"Cluster {c} (n={n})" for c, n in zip(cluster_ids, sizes)], fontsize=9)
    for i in range(len(cluster_ids)):
        for j in range(len(feature_cols)):
            ax.text(j, i, f"{zscore[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="z-score (scarto dalla media globale, in σ)")
    ax.set_title("Profiling cluster — centroidi in z-score per feature")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Profiling heatmap: {out_png}")

    # --- tabella delta (CSV) ---
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["prediction", "n_carriers"]
        for c in feature_cols:
            header += [f"{c}_mean", f"{c}_delta", f"{c}_zscore"]
        w.writerow(header)
        for i, cid in enumerate(cluster_ids):
            row = [cid, sizes[i]]
            for j, c in enumerate(feature_cols):
                row += [round(float(means[i, j]), 4), round(float(delta[i, j]), 4), round(float(zscore[i, j]), 4)]
            w.writerow(row)
    print(f"Profiling tabella delta: {out_csv}")

    print("\nDelta dei centroidi rispetto alla media globale (unità originali):")
    print("  cluster | n  | " + " | ".join(f"{c[:10]:>10s}" for c in feature_cols))
    for i, cid in enumerate(cluster_ids):
        print(f"  {cid:7d} | {sizes[i]:2d} | "
              + " | ".join(f"{delta[i, j]:>+10.2f}" for j in range(len(feature_cols))))




def build_feature_matrix(features_df, feature_cols):

    rows = features_df.select(*feature_cols).collect()

    matrix = np.array(
        [[r[c] for c in feature_cols] for r in rows],
        dtype=float
    )

    return np.nan_to_num(matrix, nan=0.0)
def compute_correlation_matrix(feature_matrix):

    corr = np.corrcoef(feature_matrix, rowvar=False)
    return np.nan_to_num(corr, nan=0.0)


def find_redundant_extra_features(corr, feature_cols, extra_cols, redundancy_threshold):

    idx = {c: i for i, c in enumerate(feature_cols)}
    features_to_remove = []
    report_rows = []

    print(f"\nGiustificazione feature aggiunte (soglia ridondanza |r| >= {redundancy_threshold}):")

    for e in extra_cols:
        corrs = {
            c: float(corr[idx[e], idx[c]])
            for c in feature_cols
            if c != e
        }

        best_c = max(corrs, key=lambda c: abs(corrs[c]))
        max_abs = abs(corrs[best_c])

        verdict = "REDUNDANT" if max_abs >= redundancy_threshold else "KEEP"

        if verdict == "REDUNDANT":
            features_to_remove.append(e)

        print(f"  {e:18s} -> {best_c:18s} |r|={max_abs:.2f}  => {verdict}")

        report_rows.append({
            "extra_feature": e,
            "best_correlate": best_c,
            "max_abs_corr": round(max_abs, 4),
            "verdict": verdict,
            "corr_values": {
                c: round(float(corr[idx[e], idx[c]]), 4)
                for c in feature_cols
            }
        })

    return features_to_remove, report_rows

def export_correlation_report(report_rows, feature_cols, out_csv):

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)

        w.writerow(
            ["extra_feature", "best_correlate", "max_abs_corr", "verdict"]
            + [f"corr_{c}" for c in feature_cols]
        )

        for row in report_rows:
            w.writerow(
                [
                    row["extra_feature"],
                    row["best_correlate"],
                    row["max_abs_corr"],
                    row["verdict"],
                ]
                + [row["corr_values"][c] for c in feature_cols]
            )

    print(f"Correlation summary CSV: {out_csv}")


def correlation_feature_selection(features_df,feature_cols,extra_cols,out_csv):

    rows = features_df.select(*feature_cols).collect()

    feature_matrix = np.array(
        [[r[c] for c in feature_cols] for r in rows],
        dtype=float
    )

    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)

    corr = np.nan_to_num(
        np.corrcoef(feature_matrix, rowvar=False),
        nan=0.0
    )

    features_to_remove, report_rows = find_redundant_extra_features(
        corr,
        feature_cols,
        extra_cols,
        redundancy_threshold=0.8
    )

    export_correlation_report(
        report_rows,
        feature_cols,
        out_csv
    )

    return features_to_remove




def remove_features(feature_cols, features_to_remove):

    if not features_to_remove:
        print("\nNessuna feature ridondante da rimuovere.")
        return feature_cols

    filtered_cols = [c for c in feature_cols if c not in features_to_remove]

    print("\nFeature rimosse per alta correlazione:")
    for c in features_to_remove:
        print(f"  - {c}")

    print("\nFeature finali usate da KMeans:")
    for c in filtered_cols:
        print(f"  - {c}")

    return filtered_cols


# Metriche per il confronto base vs extended
def export_k_metrics(k_values, wssse_list, silhouette_list, local_csv):
    with open(local_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "wssse", "silhouette"])
        for k, wss, sil in zip(k_values, wssse_list, silhouette_list):
            w.writerow([k, round(float(wss), 4), round(float(sil), 4)])
    print(f"Metriche k (WSSSE/Silhouette): {local_csv}")


# Stampa + export risultati
def print_results(df_result, feature_cols, best_k, elapsed, label):
    print(f"\n{'='*70}")
    print(f"CLUSTERING [{label}] — RISULTATI  (k={best_k}, tempo: {elapsed:.2f}s)")
    print(f"{'='*70}")
    df_result.select(
        "OP_UNIQUE_CARRIER", "total_flights", *feature_cols, "prediction"
    ).orderBy("prediction", "OP_UNIQUE_CARRIER").show(20, truncate=False)


def export_results(df_result, feature_cols, local_csv, hdfs_output):
    out_df = df_result.select(
        "OP_UNIQUE_CARRIER", "total_flights", *feature_cols, "prediction"
    ).orderBy("prediction", "OP_UNIQUE_CARRIER")
    out_rows = out_df.collect()

    with open(local_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(out_df.columns)
        for row in out_rows:
            writer.writerow([row[c] for c in out_df.columns])
    print(f"CSV locale: {local_csv}")

    out_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(hdfs_output)
    print(f"CSV HDFS:   {hdfs_output}")
