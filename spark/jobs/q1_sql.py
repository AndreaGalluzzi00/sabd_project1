"""
Q1 SQL — AA e DL: statistiche mensili DEP_DELAY e cancellation rate (gen-apr 2025)

Metriche:
  - cancellation_rate: su TUTTI i voli del mese
  - avg/min/max DEP_DELAY: solo su voli NON cancellati (CANCELLED = 0)

Modalità:
  - Dev locale:  SPARK_MASTER=local[2]  (default)
  - Cluster:     SPARK_MASTER=spark://spark-master:7077
"""
import csv
import os
import time
from pyspark.sql import SparkSession

SPARK_MASTER = os.getenv("SPARK_MASTER", "local[2]")
HDFS_INPUT   = "hdfs://namenode:9000/sabd/processed/"
HDFS_OUTPUT  = "hdfs://namenode:9000/sabd/results/q1_sql/"
LOCAL_OUT    = "/opt/spark/jobs/results/q1_sql.csv"

os.makedirs(os.path.dirname(LOCAL_OUT), exist_ok=True)

spark = (
    SparkSession.builder
    .appName("Q1_SQL_AA_DL_monthly_stats")
    .master(SPARK_MASTER)
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print(f"Master: {SPARK_MASTER}")

# ── Lettura e registrazione come vista temporanea ─────────────────────────────
df = spark.read.parquet(HDFS_INPUT)
df.createOrReplaceTempView("flights")

# ── Calcolo Q1 con Spark SQL ──────────────────────────────────────────────────
t0 = time.time()

result = spark.sql("""
    SELECT
        OP_UNIQUE_CARRIER,
        YEAR,
        MONTH,
        COUNT(*)                                                        AS total_flights,
        SUM(CANCELLED)                                                  AS cancelled_flights,
        ROUND(SUM(CANCELLED) / COUNT(*) * 100, 4)                      AS cancellation_rate_pct,
        ROUND(AVG(CASE WHEN CANCELLED = 0 THEN DEP_DELAY END), 4)      AS avg_dep_delay,
        ROUND(MIN(CASE WHEN CANCELLED = 0 THEN DEP_DELAY END), 4)      AS min_dep_delay,
        ROUND(MAX(CASE WHEN CANCELLED = 0 THEN DEP_DELAY END), 4)      AS max_dep_delay
    FROM flights
    WHERE OP_UNIQUE_CARRIER IN ('AA', 'DL')
    GROUP BY OP_UNIQUE_CARRIER, YEAR, MONTH
    ORDER BY OP_UNIQUE_CARRIER, YEAR, MONTH
""")

result.cache()
rows = result.collect()
elapsed = time.time() - t0

# ── Output a schermo ─────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Q1 SQL — RISULTATI  (tempo esecuzione: {elapsed:.2f}s)")
print(f"{'='*70}")
result.show(20, truncate=False)

# ── Salvataggio CSV locale ────────────────────────────────────────────────────
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

print(f"\nTempo Q1 (Spark SQL): {elapsed:.2f}s")
print(f"{'='*70}\n")

spark.stop()