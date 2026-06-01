"""
Q3 — Percentili DEP_DELAY per compagnia e fascia oraria
Implementazione con t-digest via DataFrame API.

A differenza della versione RDD (q3_tdigest.py), che fonde i digest map-side con
combineByKey, qui restiamo nell'API DataFrame: i valori di ogni gruppo vengono
raccolti con collect_list e il t-digest viene costruito dentro una UDF che
restituisce i 4 percentili (p25/p50/p75/p90) come array<double>.

Trade-off: collect_list materializza i valori grezzi per gruppo (niente merge
map-side dei digest). È il prezzo per usare t-digest — libreria Python esterna —
con l'API dichiarativa di Spark, che non ha un aggregato t-digest nativo.

t-digest (Dunning, 2019): errore relativo adattivo, minore sulle code
(p90/p95/p99) rispetto al centro (p50) — utile per i ritardi estremi.

Parametro delta (compressione): default della libreria (0.01); valori più bassi
→ più centroidi → più precisione → più memoria.

Modalità:
  - Dev locale (Mac M1):  SPARK_MASTER=local[2]  (default)
  - Cluster / EC2:        SPARK_MASTER=spark://spark-master:7077
"""
import csv
import os
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, floor, collect_list, udf,
    min as spark_min, max as spark_max,
)
from pyspark.sql.types import ArrayType, DoubleType
from tdigest import TDigest

SPARK_MASTER            = os.getenv("SPARK_MASTER", "spark://spark-master:7077")
HDFS_INPUT              = "hdfs://namenode:9000/sabd/processed/"
HDFS_OUTPUT_PERCENTILES = "hdfs://namenode:9000/sabd/results/q3_tdigest_df/percentiles/"
HDFS_OUTPUT_RANGE       = "hdfs://namenode:9000/sabd/results/q3_tdigest_df/delay_range/"
LOCAL_OUT_PERCENTILES   = "/opt/spark/jobs/results/q3_percentiles_tdigest_df.csv"
LOCAL_OUT_RANGE         = "/opt/spark/jobs/results/q3_delay_range_tdigest_df.csv"
LOCAL_OUT               = "/opt/spark/jobs/results"

os.makedirs(LOCAL_OUT, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("Q3_TDigest_DataFrame_Percentiles")
    .master(SPARK_MASTER)
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print(f"Master: {SPARK_MASTER}")


@udf(returnType=ArrayType(DoubleType()))
def tdigest_percentiles(values):
    """Costruisce un t-digest sui valori del gruppo → [p25, p50, p75, p90]."""
    if not values:
        return None
    digest = TDigest()
    for v in values:
        digest.update(float(v))
    return [
        round(float(digest.percentile(25)), 2),
        round(float(digest.percentile(50)), 2),
        round(float(digest.percentile(75)), 2),
        round(float(digest.percentile(90)), 2),
    ]


df = (
    spark.read.parquet(HDFS_INPUT)
    .filter(
        (col("OP_UNIQUE_CARRIER").isin("AA", "DL", "UA", "WN")) &
        (col("CANCELLED") == 0)
    )
    .withColumn("hour", floor(col("CRS_DEP_TIME") / 100))
)

t0 = time.time()

# collect_list ignora i NULL → niente dropna esplicito necessario.
percentiles = (
    df.groupBy("OP_UNIQUE_CARRIER", "hour")
    .agg(collect_list("DEP_DELAY").alias("vals"))
    .withColumn("p", tdigest_percentiles(col("vals")))
    .select(
        "OP_UNIQUE_CARRIER", "hour",
        col("p")[0].alias("p25"),
        col("p")[1].alias("p50"),
        col("p")[2].alias("p75"),
        col("p")[3].alias("p90"),
    )
    .orderBy("OP_UNIQUE_CARRIER", "hour")
)

# Min/Max: esatti, aggregazione nativa DataFrame (no approssimazione).
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

# ── Output ────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Q3 t-digest (DataFrame) — RISULTATI  (tempo esecuzione: {elapsed:.2f}s)")
print(f"{'='*70}")
print("\n=== Percentili per compagnia e fascia oraria ===")
percentiles.show(100, truncate=False)
print("\n=== Min/Max DEP_DELAY per compagnia ===")
delay_range.show(truncate=False)

# ── CSV locale ────────────────────────────────────────────────────────────────
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
print(f"CSV locale min/max:    {LOCAL_OUT_RANGE}")

# ── HDFS ──────────────────────────────────────────────────────────────────────
percentiles.coalesce(1).write.mode("overwrite").option("header", True).csv(HDFS_OUTPUT_PERCENTILES)
delay_range.coalesce(1).write.mode("overwrite").option("header", True).csv(HDFS_OUTPUT_RANGE)

print(f"CSV HDFS percentili:   {HDFS_OUTPUT_PERCENTILES}")
print(f"CSV HDFS min/max:      {HDFS_OUTPUT_RANGE}")
print(f"\nTempo Q3 t-digest DataFrame: {elapsed:.2f}s")
print(f"{'='*70}\n")

spark.stop()
