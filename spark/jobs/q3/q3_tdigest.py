"""
Q3 — Percentili DEP_DELAY per compagnia e fascia oraria
Implementazione con t-digest via RDD API

t-digest (Dunning, 2019): struttura a centroidi pesati con errore adattivo.
A differenza di percentile_approx (Greenwald-Khanna, errore uniforme),
t-digest garantisce errore relativo minore sulle code (p90, p95, p99)
rispetto al centro (p50) — proprietà utile per analisi di ritardi estremi.

La mergeability nativa dei digest permette una pipeline distribuita corretta:
  - ogni nodo accumula un digest locale (combineByKey, fase map-side)
  - i digest locali vengono mergiati senza riaggregare i dati originali
  - nessuno shuffle dei dati grezzi, solo trasferimento di strutture compatte

Parametro delta (compressione): default 0.01
  - valori più bassi → più centroidi → maggiore precisione → più memoria
"""
import csv
import os
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, floor, min as spark_min, max as spark_max
from tdigest import TDigest

SPARK_MASTER            = os.getenv("SPARK_MASTER", "spark://spark-master:7077")
HDFS_INPUT              = "hdfs://namenode:9000/sabd/processed/"
HDFS_OUTPUT_PERCENTILES = "hdfs://namenode:9000/sabd/results/q3_tdigest/percentiles/"
HDFS_OUTPUT_RANGE       = "hdfs://namenode:9000/sabd/results/q3_tdigest/delay_range/"
LOCAL_OUT_PERCENTILES   = "/opt/spark/jobs/results/q3_percentiles_tdigest.csv"
LOCAL_OUT_RANGE         = "/opt/spark/jobs/results/q3_delay_range_tdigest.csv"
LOCAL_OUT               = "/opt/spark/jobs/results"

os.makedirs(LOCAL_OUT, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("Q3_TDigest_Percentiles")
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
    .select("OP_UNIQUE_CARRIER", "hour", "DEP_DELAY")
    .dropna(subset=["DEP_DELAY"])
)

t0 = time.time()

# ── Percentili via RDD + t-digest ─────────────────────────────────────────────
#
# combineByKey è preferibile ad aggregateByKey perché usa una factory
# (create_combiner) invece di un valore zero condiviso — evita il problema
# di mutare lo stesso oggetto TDigest su chiavi diverse nella stessa partizione.
#
# Pipeline:
#   create_combiner : prima occorrenza di una chiave → crea un nuovo TDigest
#   merge_value     : occorrenze successive nella stessa partizione → update
#   merge_combiners : riduzione tra partizioni → merge dei centroidi

def create_combiner(value: float) -> TDigest:
    d = TDigest()
    d.update(value)
    return d

def merge_value(digest: TDigest, value: float) -> TDigest:
    digest.update(value)
    return digest

def merge_combiners(d1: TDigest, d2: TDigest) -> TDigest:
    # Itera i centroidi di d2 e li inserisce in d1 con il loro peso.
    # Questo è il merge distribuito nativo di t-digest: trasferisce solo
    # le strutture compatte (centroidi), non i dati grezzi.
    for centroid in d2.C.values():
        d1.update(centroid.mean, centroid.count)
    return d1

rdd = df.rdd.map(lambda row: (
    (row.OP_UNIQUE_CARRIER, int(row.hour)),
    float(row.DEP_DELAY)
))

percentile_rdd = (
    rdd
    .combineByKey(create_combiner, merge_value, merge_combiners)
    .map(lambda kv: (
        kv[0][0],                            # carrier
        kv[0][1],                            # hour
        round(kv[1].percentile(25), 2),      # p25
        round(kv[1].percentile(50), 2),      # p50
        round(kv[1].percentile(75), 2),      # p75
        round(kv[1].percentile(90), 2),      # p90
    ))
    .sortBy(lambda x: (x[0], x[1]))
)

# Min/Max: operazioni esatte, calcolate su DataFrame (no approssimazione)
delay_range = (
    df.groupBy("OP_UNIQUE_CARRIER")
    .agg(
        spark_min("DEP_DELAY").alias("min_delay"),
        spark_max("DEP_DELAY").alias("max_delay"),
    )
    .orderBy("OP_UNIQUE_CARRIER")
)

percentile_rows = percentile_rdd.collect()
range_rows = delay_range.collect()
elapsed = time.time() - t0

# ── Output ────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Q3 t-digest — RISULTATI  (tempo esecuzione: {elapsed:.2f}s)")
print(f"{'='*70}")
print(f"{'carrier':<10} {'hour':>4} {'p25':>6} {'p50':>6} {'p75':>6} {'p90':>6}")
for row in sorted(percentile_rows):
    print(f"{row[0]:<10} {row[1]:>4} {row[2]:>6} {row[3]:>6} {row[4]:>6} {row[5]:>6}")

print("\nMin/Max DEP_DELAY per compagnia:")
for row in range_rows:
    print(f"  {row.OP_UNIQUE_CARRIER}: min={row.min_delay}, max={row.max_delay}")

# ── CSV locale ────────────────────────────────────────────────────────────────
COLS_PERC = ["OP_UNIQUE_CARRIER", "hour", "p25", "p50", "p75", "p90"]

with open(LOCAL_OUT_PERCENTILES, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(COLS_PERC)
    writer.writerows(percentile_rows)

with open(LOCAL_OUT_RANGE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(delay_range.columns)
    for row in range_rows:
        writer.writerow([row[c] for c in delay_range.columns])

print(f"\nCSV locale percentili: {LOCAL_OUT_PERCENTILES}")
print(f"CSV locale min/max:    {LOCAL_OUT_RANGE}")

# ── HDFS ──────────────────────────────────────────────────────────────────────
perc_df = spark.createDataFrame(percentile_rows, COLS_PERC)
perc_df.coalesce(1).write.mode("overwrite").option("header", True).csv(HDFS_OUTPUT_PERCENTILES)
delay_range.coalesce(1).write.mode("overwrite").option("header", True).csv(HDFS_OUTPUT_RANGE)

print(f"CSV HDFS percentili:   {HDFS_OUTPUT_PERCENTILES}")
print(f"CSV HDFS min/max:      {HDFS_OUTPUT_RANGE}")
print(f"\nTempo Q3 t-digest: {elapsed:.2f}s")
print(f"{'='*70}\n")

spark.stop()
