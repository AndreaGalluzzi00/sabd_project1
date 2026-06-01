"""
Q3 (RDD) — Percentili DEP_DELAY per compagnia e fascia oraria
Versione della famiglia "approximate di Spark" con le API RDD.

NOTA IMPORTANTE — le API RDD non espongono percentile_approx / approxQuantile:
l'algoritmo (Greenwald-Khanna) vive nel motore SQL/Catalyst ed è disponibile
solo su DataFrame e Spark SQL (vedi q3.py e q3_sql.py). Questa versione RDD
raggruppa i valori per (compagnia, fascia oraria) con groupByKey e calcola i
percentili in modo ESATTO via ordinamento + interpolazione lineare.

È quindi il RIFERIMENTO ESATTO contro cui confrontare le stime approssimate:
  - percentile_approx (q3.py / q3_sql.py)
  - KLL sketch        (q3_kll*.py)
  - t-digest          (q3_tdigest*.py)

Scalabilità: groupByKey materializza tutti i valori di un gruppo in memoria.
Accettabile su questo dataset (4 compagnie × 24 fasce). Gli sketch sono
precisamente l'alternativa che evita questa materializzazione.

Modalità:
  - Dev locale (Mac M1):  SPARK_MASTER=local[2]  (default)
  - Cluster / EC2:        SPARK_MASTER=spark://spark-master:7077
"""
import csv
import math
import os
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, floor

SPARK_MASTER            = os.getenv("SPARK_MASTER", "spark://spark-master:7077")
HDFS_INPUT              = "hdfs://namenode:9000/sabd/processed/"
HDFS_OUTPUT_PERCENTILES = "hdfs://namenode:9000/sabd/results/q3_rdd/percentiles/"
HDFS_OUTPUT_RANGE       = "hdfs://namenode:9000/sabd/results/q3_rdd/delay_range/"
LOCAL_OUT_PERCENTILES   = "/opt/spark/jobs/results/q3_percentiles_rdd.csv"
LOCAL_OUT_RANGE         = "/opt/spark/jobs/results/q3_delay_range_rdd.csv"
LOCAL_OUT               = "/opt/spark/jobs/results"

COLS_PERC  = ["OP_UNIQUE_CARRIER", "hour", "p25", "p50", "p75", "p90"]
COLS_RANGE = ["OP_UNIQUE_CARRIER", "min_delay", "max_delay"]

os.makedirs(LOCAL_OUT, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("Q3_RDD_Percentiles")
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


# ─────────────────────────────────────────────────────────────────────────────
# Percentile esatto su lista ordinata (interpolazione lineare, metodo 'linear'
# / type-7 di NumPy). Eseguito sugli executor dentro mapValues.
# ─────────────────────────────────────────────────────────────────────────────
def percentile(sorted_vals, q):
    n = len(sorted_vals)
    if n == 0:
        return None
    if n == 1:
        return float(sorted_vals[0])
    pos = q * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_vals[int(pos)])
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def quantiles(values_iter):
    vals = sorted(values_iter)
    return (
        round(percentile(vals, 0.25), 2),
        round(percentile(vals, 0.50), 2),
        round(percentile(vals, 0.75), 2),
        round(percentile(vals, 0.90), 2),
    )


t0 = time.time()

# Base RDD (carrier, hour, delay) riusato per percentili e min/max.
base = (
    df.rdd
    .map(lambda r: (r["OP_UNIQUE_CARRIER"], int(r["hour"]), float(r["DEP_DELAY"])))
    .cache()
)

# Percentili: groupByKey per (carrier, hour) → calcolo esatto sul gruppo ordinato.
percentile_rdd = (
    base
    .map(lambda t: ((t[0], t[1]), t[2]))
    .groupByKey()
    .mapValues(quantiles)
    .map(lambda kv: (kv[0][0], kv[0][1], *kv[1]))
    .sortBy(lambda x: (x[0], x[1]))
)

# Min/Max DEP_DELAY per compagnia (esatto, reduceByKey).
range_rdd = (
    base
    .map(lambda t: (t[0], (t[2], t[2])))
    .reduceByKey(lambda a, b: (min(a[0], b[0]), max(a[1], b[1])))
    .sortByKey()
)

percentile_rows = percentile_rdd.collect()
range_rows = [(c, mn, mx) for c, (mn, mx) in range_rdd.collect()]
elapsed = time.time() - t0

# ── Output ────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Q3 (RDD, percentili esatti) — RISULTATI  (tempo esecuzione: {elapsed:.2f}s)")
print(f"{'='*70}")
print(f"{'carrier':<10} {'hour':>4} {'p25':>8} {'p50':>8} {'p75':>8} {'p90':>8}")
for row in percentile_rows:
    print(f"{row[0]:<10} {row[1]:>4} {row[2]:>8} {row[3]:>8} {row[4]:>8} {row[5]:>8}")

print("\nMin/Max DEP_DELAY per compagnia:")
for c, mn, mx in range_rows:
    print(f"  {c}: min={mn}, max={mx}")

# ── CSV locale ────────────────────────────────────────────────────────────────
with open(LOCAL_OUT_PERCENTILES, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(COLS_PERC)
    writer.writerows(percentile_rows)

with open(LOCAL_OUT_RANGE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(COLS_RANGE)
    writer.writerows(range_rows)

print(f"\nCSV locale percentili: {LOCAL_OUT_PERCENTILES}")
print(f"CSV locale min/max:    {LOCAL_OUT_RANGE}")

# ── HDFS ──────────────────────────────────────────────────────────────────────
spark.createDataFrame(percentile_rows, COLS_PERC) \
    .coalesce(1).write.mode("overwrite").option("header", True).csv(HDFS_OUTPUT_PERCENTILES)
spark.createDataFrame(range_rows, COLS_RANGE) \
    .coalesce(1).write.mode("overwrite").option("header", True).csv(HDFS_OUTPUT_RANGE)

print(f"CSV HDFS percentili:   {HDFS_OUTPUT_PERCENTILES}")
print(f"CSV HDFS min/max:      {HDFS_OUTPUT_RANGE}")
print(f"\nTempo Q3 (RDD, esatto): {elapsed:.2f}s")
print(f"{'='*70}\n")

spark.stop()
