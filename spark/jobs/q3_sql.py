import csv
import os
import time

from pyspark.sql import SparkSession

SPARK_MASTER = os.getenv("SPARK_MASTER", "local[2]")
HDFS_INPUT = "hdfs://namenode:9000/sabd/processed/"
HDFS_OUTPUT_PERCENTILES = "hdfs://namenode:9000/sabd/results/q3_sql/percentiles/"
HDFS_OUTPUT_RANGE = "hdfs://namenode:9000/sabd/results/q3_sql/delay_range/"
LOCAL_OUT_PERCENTILES = "/opt/spark/jobs/results/q3_sql_percentiles.csv"
LOCAL_OUT_RANGE = "/opt/spark/jobs/results/q3_sql_delay_range.csv"
LOCAL_OUT = "/opt/spark/jobs/results"

os.makedirs(LOCAL_OUT, exist_ok=True)
os.makedirs(os.path.dirname(LOCAL_OUT_PERCENTILES), exist_ok=True)

spark = (
    SparkSession.builder
    .appName("Q3_SQL_Percentiles")
    .master(SPARK_MASTER)
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")
print(f"Master: {SPARK_MASTER}")

df = spark.read.parquet(HDFS_INPUT)
# registro questo DataFrame come tabella SQL temporanea chiamata flights
df.createOrReplaceTempView("flights")

# ── Calcolo Q3 SQL ───────────────────────────────────────────────────────────

t0 = time.time()

percentiles = spark.sql("""
    SELECT
        OP_UNIQUE_CARRIER,
        FLOOR(CRS_DEP_TIME / 100) AS hour,

        percentile_approx(DEP_DELAY, 0.25) AS p25,
        percentile_approx(DEP_DELAY, 0.50) AS p50,
        percentile_approx(DEP_DELAY, 0.75) AS p75,
        percentile_approx(DEP_DELAY, 0.90) AS p90

    FROM flights

    WHERE OP_UNIQUE_CARRIER IN ('AA', 'DL', 'UA', 'WN')
      AND CANCELLED = 0

    GROUP BY
        OP_UNIQUE_CARRIER,
        FLOOR(CRS_DEP_TIME / 100)

    ORDER BY
        OP_UNIQUE_CARRIER,
        hour
""")

delay_range = spark.sql("""
    SELECT
        OP_UNIQUE_CARRIER,

        MIN(DEP_DELAY) AS min_delay,
        MAX(DEP_DELAY) AS max_delay

    FROM flights

    WHERE OP_UNIQUE_CARRIER IN ('AA', 'DL', 'UA', 'WN')
      AND CANCELLED = 0

    GROUP BY OP_UNIQUE_CARRIER

    ORDER BY OP_UNIQUE_CARRIER
""")

percentiles.cache()
delay_range.cache()

percentile_rows = percentiles.collect()
range_rows = delay_range.collect()

elapsed = time.time() - t0

# ── Output a schermo ─────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"Q3 SQL — RISULTATI  (tempo esecuzione: {elapsed:.2f}s)")
print(f"{'='*70}")

print("\n=== Percentili per compagnia e fascia oraria ===")
percentiles.show(100, truncate=False)

print("\n=== Min/Max DEP_DELAY per compagnia ===")
delay_range.show(truncate=False)

# ── Salvataggio CSV locale ───────────────────────────────────────────────────

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

print(f"\nCSV locale percentili: {LOCAL_OUT_PERCENTILES}")
print(f"CSV locale min/max: {LOCAL_OUT_RANGE}")

# ── Salvataggio su HDFS ──────────────────────────────────────────────────────

percentiles.coalesce(1) \
    .write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(HDFS_OUTPUT_PERCENTILES)

delay_range.coalesce(1) \
    .write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(HDFS_OUTPUT_RANGE)

print(f"\nCSV HDFS percentili: {HDFS_OUTPUT_PERCENTILES}")
print(f"CSV HDFS min/max: {HDFS_OUTPUT_RANGE}")

print(f"\nTempo Q3 SQL: {elapsed:.2f}s")
print(f"{'='*70}\n")

spark.stop()