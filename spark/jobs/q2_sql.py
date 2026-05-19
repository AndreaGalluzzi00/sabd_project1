"""
Q2 SQL — Top-10 compagnie per ARR_DELAY medio (gen-apr 2025)

Filtro base: voli non cancellati e non deviati (CANCELLED=0, DIVERTED=0)
Soglia:      solo compagnie con >= 500 voli nel filtro base
Metriche:    num_flights, avg_arr_delay, media delle 5 cause di ritardo
NULL cause:  trattati come 0 (BTS li omette quando delay totale < 15min)
Ordine:      top-10 per avg_arr_delay decrescente

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
HDFS_OUTPUT  = "hdfs://namenode:9000/sabd/results/q2_sql/"
LOCAL_OUT    = "/opt/spark/jobs/results/q2_sql.csv"

os.makedirs(os.path.dirname(LOCAL_OUT), exist_ok=True)

spark = (
    SparkSession.builder
    .appName("Q2_SQL_top10_carriers_arr_delay")
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

# ── Calcolo Q2 con Spark SQL ──────────────────────────────────────────────────
t0 = time.time()

result = spark.sql("""
    SELECT
        OP_UNIQUE_CARRIER,
        COUNT(*)                                            AS num_flights,
        ROUND(AVG(ARR_DELAY), 4)                           AS avg_arr_delay,
        ROUND(AVG(COALESCE(CARRIER_DELAY,      0.0)), 4)   AS avg_carrier_delay,
        ROUND(AVG(COALESCE(WEATHER_DELAY,      0.0)), 4)   AS avg_weather_delay,
        ROUND(AVG(COALESCE(NAS_DELAY,          0.0)), 4)   AS avg_nas_delay,
        ROUND(AVG(COALESCE(SECURITY_DELAY,     0.0)), 4)   AS avg_security_delay,
        ROUND(AVG(COALESCE(LATE_AIRCRAFT_DELAY,0.0)), 4)   AS avg_late_aircraft_delay
    FROM flights
    WHERE CANCELLED = 0
      AND DIVERTED  = 0
    GROUP BY OP_UNIQUE_CARRIER
    HAVING COUNT(*) >= 500
    ORDER BY avg_arr_delay DESC
    LIMIT 10
""")

result.cache()
rows = result.collect()
elapsed = time.time() - t0

# ── Output a schermo ─────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Q2 SQL — RISULTATI  (tempo esecuzione: {elapsed:.2f}s)")
print(f"{'='*70}")
result.show(10, truncate=False)

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

print(f"\nTempo Q2 (Spark SQL): {elapsed:.2f}s")
print(f"{'='*70}\n")

spark._sc._jvm.System.exit(0)
