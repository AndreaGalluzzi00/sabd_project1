import os
import time
import csv

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, floor, expr, min as spark_min, max as spark_max

SPARK_MASTER = os.getenv("SPARK_MASTER", "local[2]")
HDFS_INPUT = "hdfs://namenode:9000/sabd/processed/"
HDFS_OUTPUT_PERCENTILES = "hdfs://namenode:9000/sabd/results/q3/percentiles/"
HDFS_OUTPUT_RANGE = "hdfs://namenode:9000/sabd/results/q3/delay_range/"
LOCAL_OUT_PERCENTILES = "/opt/spark/jobs/results/q3_percentiles.csv"
LOCAL_OUT_RANGE = "/opt/spark/jobs/results/q3_delay_range.csv"
LOCAL_OUT = "/opt/spark/jobs/results"

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

spark.stop()