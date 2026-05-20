"""
Q1 — AA e DL: statistiche mensili DEP_DELAY e cancellation rate (gen-apr 2025)

Metriche:
  - cancellation_rate: su TUTTI i voli del mese
  - avg/min/max DEP_DELAY: solo su voli NON cancellati (CANCELLED = 0)

Modalità:
  - Dev locale (Mac M1):  SPARK_MASTER=local[2]  (default)
  - Cluster / EC2:        SPARK_MASTER=spark://spark-master:7077
"""
import csv
import os
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, min as spark_min, max as spark_max,
    count, sum as spark_sum, when, round as spark_round,
)

SPARK_MASTER = os.getenv("SPARK_MASTER", "local[2]")
HDFS_INPUT   = "hdfs://namenode:9000/sabd/processed/"
HDFS_OUTPUT  = "hdfs://namenode:9000/sabd/results/q1/"
LOCAL_OUT    = "/opt/spark/jobs/results/q1.csv"

os.makedirs(os.path.dirname(LOCAL_OUT), exist_ok=True)

spark = (
    SparkSession.builder
    .appName("Q1_AA_DL_monthly_stats")
    .master(SPARK_MASTER)
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print(f"Master: {SPARK_MASTER}")

# ── Lettura ──────────────────────────────────────────────────────────────────
df = (
    spark.read.parquet(HDFS_INPUT)
    .filter(col("OP_UNIQUE_CARRIER").isin("AA", "DL"))
)

# ── Calcolo Q1 ───────────────────────────────────────────────────────────────
t0 = time.time()

result = (
    df.groupBy("OP_UNIQUE_CARRIER", "YEAR", "MONTH")
    .agg(
        count("*").alias("total_flights"),
        spark_sum(col("CANCELLED").cast("long")).alias("cancelled_flights"),
        spark_round(
            spark_sum("CANCELLED") / count("*") * 100, 4
        ).alias("cancellation_rate_pct"),
        spark_round(
            avg(when(col("CANCELLED") == 0, col("DEP_DELAY"))), 4
        ).alias("avg_dep_delay"),
        spark_round(
            spark_min(when(col("CANCELLED") == 0, col("DEP_DELAY"))), 4
        ).alias("min_dep_delay"),
        spark_round(
            spark_max(when(col("CANCELLED") == 0, col("DEP_DELAY"))), 4
        ).alias("max_dep_delay"),
    )
    .orderBy("OP_UNIQUE_CARRIER", "YEAR", "MONTH")
)

result.cache()
rows = result.collect()
elapsed = time.time() - t0

# ── Output a schermo ─────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Q1 — RISULTATI  (tempo esecuzione: {elapsed:.2f}s)")
print(f"{'='*70}")
result.show(20, truncate=False)

# ── Salvataggio CSV locale (directory montata → visibile sull'host) ──────────
cols = result.columns
with open(LOCAL_OUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(cols)
    for row in rows:
        writer.writerow([row[c] for c in cols])
print(f"CSV locale:  {LOCAL_OUT}")

# ── Salvataggio su HDFS ───────────────────────────────────────────────────────
result.coalesce(1).write.mode("overwrite").option("header", "true").csv(HDFS_OUTPUT)
print(f"CSV su HDFS: {HDFS_OUTPUT}")

print(f"\nTempo Q1 (DataFrame API): {elapsed:.2f}s")
print(f"{'='*70}\n")

spark.stop()