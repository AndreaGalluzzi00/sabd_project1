import os
import time
import csv
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, floor, expr, min as spark_min, max as spark_max

SPARK_MASTER            = os.getenv("SPARK_MASTER", "local[2]")
HDFS_INPUT              = "hdfs://namenode:9000/sabd/processed/"
HDFS_OUTPUT_PERCENTILES = "hdfs://namenode:9000/sabd/results/q3/percentiles/"
HDFS_OUTPUT_RANGE       = "hdfs://namenode:9000/sabd/results/q3/delay_range/"
LOCAL_OUT_PERCENTILES   = "/opt/spark/jobs/results/q3_percentiles.csv"
LOCAL_OUT_RANGE         = "/opt/spark/jobs/results/q3_delay_range.csv"
LOCAL_OUT               = "/opt/spark/jobs/results"
LOCAL_OUT_PNG_PERCENTILES = "/opt/spark/jobs/results/q3_percentiles.png"

os.makedirs(LOCAL_OUT, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("Q3_Percentiles_by_Hour")
    .master(SPARK_MASTER)
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")
print(f"Master: {SPARK_MASTER}")

df = (
    spark.read.parquet(HDFS_INPUT)
    .filter(
        (col("OP_UNIQUE_CARRIER").isin("AA", "DL", "UA", "WN")) &
        (col("CANCELLED") == 0)
    )
    .withColumn("hour", floor(col("CRS_DEP_TIME") / 100))
)
# ── Calcolo Percentili ───────────────────────────────────────────────────────────────

t0 = time.time()

percentiles = (
    df.groupBy("OP_UNIQUE_CARRIER", "hour")
    .agg(
        expr("percentile_approx(DEP_DELAY, 0.25)").alias("p25"),
        expr("percentile_approx(DEP_DELAY, 0.50)").alias("p50"),
        expr("percentile_approx(DEP_DELAY, 0.75)").alias("p75"),
        expr("percentile_approx(DEP_DELAY, 0.90)").alias("p90"),
    )
    .orderBy("OP_UNIQUE_CARRIER", "hour")
)

delay_range = (
    df.groupBy("OP_UNIQUE_CARRIER")
    .agg(
        spark_min("DEP_DELAY").alias("min_delay"),
        spark_max("DEP_DELAY").alias("max_delay"),
    )
    .orderBy("OP_UNIQUE_CARRIER")
)

percentiles.cache()
delay_range.cache()

percentile_rows = percentiles.collect()
range_rows = delay_range.collect()
elapsed = time.time() - t0

# ── Output a schermo ─────────────────────────────────────────────────────────

print("=== Q3 Percentili per compagnia e fascia oraria ===")
percentiles.show(100, truncate=False)

print("=== Min/Max DEP_DELAY per compagnia ===")
delay_range.show(truncate=False)

# ── Salvataggio CSV locale (directory montata → visibile sull'host) ──────────

with open(LOCAL_OUT_PERCENTILES, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(percentiles.columns)
    for row in percentile_rows:
        writer.writerow([row[c] for c in percentiles.columns])

with open(LOCAL_OUT_RANGE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(delay_range.columns)
    for row in range_rows:
        writer.writerow([row[c] for c in delay_range.columns])

percentiles.coalesce(1).write.mode("overwrite").option("header", True).csv(HDFS_OUTPUT_PERCENTILES)
delay_range.coalesce(1).write.mode("overwrite").option("header", True).csv(HDFS_OUTPUT_RANGE)

print(f"CSV locale percentili: {LOCAL_OUT_PERCENTILES}")
print(f"CSV locale min/max: {LOCAL_OUT_RANGE}")
print(f"CSV HDFS percentili: {HDFS_OUTPUT_PERCENTILES}")
print(f"CSV HDFS min/max: {HDFS_OUTPUT_RANGE}")
print(f"Tempo Q3: {elapsed:.2f}s")

# ── Grafici Q3 ────────────────────────────────────────────────────────────────
# Organizza i dati per carrier
data_by_carrier = defaultdict(lambda: {"hour": [], "p25": [], "p50": [], "p75": [], "p90": []})
for row in percentile_rows:
    c = row["OP_UNIQUE_CARRIER"]
    data_by_carrier[c]["hour"].append(int(row["hour"]))
    for pct in ["p25", "p50", "p75", "p90"]:
        data_by_carrier[c][pct].append(float(row[pct]))

carriers_q3 = sorted(data_by_carrier.keys())
pct_styles = {
    "p25": ("#2196F3", "-"),
    "p50": ("#4CAF50", "-"),
    "p75": ("#FF9800", "-"),
    "p90": ("#F44336", "--"),
}
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
axes = axes.flatten()
for ax, carrier in zip(axes, carriers_q3):
    d = data_by_carrier[carrier]
    for pct, (color, ls) in pct_styles.items():
        ax.plot(d["hour"], d[pct], marker="o", markersize=3,
                label=pct.upper(), color=color, linestyle=ls)
    ax.set_title(carrier, fontsize=13, fontweight="bold")
    ax.set_xlabel("Ora del giorno", fontsize=10)
    ax.set_ylabel("DEP_DELAY (min)", fontsize=10)
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
fig.suptitle("Q3 — Distribuzione ritardi in partenza per fascia oraria (AA, DL, UA, WN)",
             fontsize=13)
plt.tight_layout()
plt.savefig(LOCAL_OUT_PNG_PERCENTILES, dpi=150, bbox_inches="tight")
plt.close()
print(f"Grafico percentili:   {LOCAL_OUT_PNG_PERCENTILES}")

spark.stop()