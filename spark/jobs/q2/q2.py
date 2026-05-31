"""
Q2 — Top-10 compagnie per ARR_DELAY medio (gen-apr 2025)

Filtro base: voli non cancellati e non deviati (CANCELLED=0, DIVERTED=0)
Soglia:      solo compagnie con ≥500 voli nel filtro base
Metriche:    num_flights, avg_arr_delay, media delle 5 cause di ritardo
NULL cause:  trattati come 0 (BTS li omette quando delay totale < 15min)
Ordine:      top-10 per avg_arr_delay decrescente

Modalità:
  - Dev locale (Mac M1):  SPARK_MASTER=local[2]  (default)
  - Cluster / EC2:        SPARK_MASTER=spark://spark-master:7077
"""
import csv
import os
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, count, desc,
    coalesce, lit,
    round as spark_round,
)

SPARK_MASTER = os.getenv("SPARK_MASTER","spark://spark-master:7077")
HDFS_INPUT = "hdfs://namenode:9000/sabd/processed/"
HDFS_OUTPUT = "hdfs://namenode:9000/sabd/results/q2/"
LOCAL_OUT = "/opt/spark/jobs/results/q2.csv"
LOCAL_OUT_PNG_RANKING = "/opt/spark/jobs/results/q2_ranking.png"
LOCAL_OUT_PNG_COMPONENTS = "/opt/spark/jobs/results/q2_delay_components.png"

os.makedirs(os.path.dirname(LOCAL_OUT), exist_ok=True)

spark = (
    SparkSession.builder
    .appName("Q2_top10_carriers_arr_delay")
    .master(SPARK_MASTER)
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print(f"Master: {SPARK_MASTER}")

# Filtro base: voli completati (non cancellati, non deviati)
df = (
    spark.read.parquet(HDFS_INPUT)
    .filter((col("CANCELLED") == 0) & (col("DIVERTED") == 0))
)

# Calcolo Q2
t0 = time.time()

result = (
    df.groupBy("OP_UNIQUE_CARRIER")
    .agg(
        count("*").alias("num_flights"),
        spark_round(avg("ARR_DELAY"), 4).alias("avg_arr_delay"),
        spark_round(
            avg(coalesce(col("CARRIER_DELAY"),      lit(0.0))), 4
        ).alias("avg_carrier_delay"),
        spark_round(
            avg(coalesce(col("WEATHER_DELAY"),      lit(0.0))), 4
        ).alias("avg_weather_delay"),
        spark_round(
            avg(coalesce(col("NAS_DELAY"),          lit(0.0))), 4
        ).alias("avg_nas_delay"),
        spark_round(
            avg(coalesce(col("SECURITY_DELAY"),     lit(0.0))), 4
        ).alias("avg_security_delay"),
        spark_round(
            avg(coalesce(col("LATE_AIRCRAFT_DELAY"),lit(0.0))), 4
        ).alias("avg_late_aircraft_delay"),
    )
    .filter(col("num_flights") >= 500)
    .orderBy(desc("avg_arr_delay"))
    .limit(10)
)

result.cache()
rows = result.collect()
elapsed = time.time() - t0

# Output a schermo
print(f"\n{'='*70}")
print(f"Q2 — RISULTATI  (tempo esecuzione: {elapsed:.2f}s)")
print(f"{'='*70}")
result.show(10, truncate=False)

# Salvataggio CSV locale
cols = result.columns
with open(LOCAL_OUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(cols)
    for row in rows:
        writer.writerow([row[c] for c in cols])
print(f"CSV locale:  {LOCAL_OUT}")

# Salvataggio su HDFS
result.coalesce(1).write.mode("overwrite").option("header", "true").csv(HDFS_OUTPUT)
print(f"CSV su HDFS: {HDFS_OUTPUT}")

print(f"\nTempo Q2 (DataFrame API): {elapsed:.2f}s")
print(f"{'='*70}\n")


spark.stop()