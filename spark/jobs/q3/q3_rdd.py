"""
Q3 (RDD) — Percentili DEP_DELAY per compagnia e fascia oraria
Versione della famiglia "approximate di Spark" con le API RDD.

NOTA IMPORTANTE — le API RDD non espongono percentile_approx / approxQuantile:
l'algoritmo (Greenwald-Khanna) vive nel motore SQL/Catalyst ed è disponibile
solo su DataFrame e Spark SQL (vedi q3_df.py e q3_sql.py). Questa versione RDD
raggruppa i valori per (compagnia, fascia oraria) con groupByKey e calcola i
percentili in modo ESATTO via ordinamento + interpolazione lineare.

È quindi il RIFERIMENTO ESATTO contro cui confrontare le stime approssimate:
  - percentile_approx (q3_df.py / q3_sql.py)
  - KLL sketch        (q3_kll*.py)
  - t-digest          (q3_tdigest*.py)

Scalabilità: groupByKey materializza tutti i valori di un gruppo in memoria.
Accettabile su questo dataset (4 compagnie × 24 fasce). Gli sketch sono
precisamente l'alternativa che evita questa materializzazione.

Modalità:
  - Dev locale (Mac M1):  SPARK_MASTER=local[2]  (default)
  - Cluster / EC2:        SPARK_MASTER=spark://spark-master:7077
"""
import math
import os
import sys
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, floor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from utils_output import (
    show_rdd_result,
    save_rdd_csv_local
)

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
    .config("spark.ui.enabled", "true")
    .config("spark.ui.port", "4040")
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
show_rdd_result(
    rows=percentile_rows,
    header=COLS_PERC,
    query_name="Q3 RDD — Percentili DEP_DELAY per compagnia e fascia oraria",
    elapsed=elapsed,
)

show_rdd_result(
    rows=range_rows,
    header=COLS_RANGE,
    query_name="Q3 RDD — Min/Max DEP_DELAY per compagnia",
    elapsed=elapsed,
)

# ── CSV locale ────────────────────────────────────────────────────────────────
save_rdd_csv_local(
    path=LOCAL_OUT_PERCENTILES,
    header=COLS_PERC,
    rows=percentile_rows,
)

save_rdd_csv_local(
    path=LOCAL_OUT_RANGE,
    header=COLS_RANGE,
    rows=range_rows,
)


# ── HDFS ──────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# HDFS: salvataggio RDD puro con saveAsTextFile
# ─────────────────────────────────────────────────────────────────────────────
sc = spark.sparkContext

# saveAsTextFile NON sovrascrive: cancelliamo prima eventuali output esistenti
# usando le API Hadoop FileSystem, come già fatto nelle altre versioni RDD.
hadoop_conf = sc._jsc.hadoopConfiguration()
fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)

out_percentiles = sc._jvm.org.apache.hadoop.fs.Path(HDFS_OUTPUT_PERCENTILES)
out_range = sc._jvm.org.apache.hadoop.fs.Path(HDFS_OUTPUT_RANGE)

if fs.exists(out_percentiles):
    fs.delete(out_percentiles, True)

if fs.exists(out_range):
    fs.delete(out_range, True)

# Costruzione delle righe CSV.
percentile_lines = (
    [",".join(COLS_PERC)] +
    [",".join(str(x) for x in row) for row in percentile_rows]
)

range_lines = (
    [",".join(COLS_RANGE)] +
    [",".join(str(x) for x in row) for row in range_rows]
)

# Un solo part-file per ciascun output.
sc.parallelize(percentile_lines, numSlices=1).saveAsTextFile(HDFS_OUTPUT_PERCENTILES)
sc.parallelize(range_lines, numSlices=1).saveAsTextFile(HDFS_OUTPUT_RANGE)

print(f"CSV su HDFS: {HDFS_OUTPUT_PERCENTILES}")
print(f"CSV su HDFS: {HDFS_OUTPUT_RANGE}")
print(f"\nTempo Q3 (RDD, esatto): {elapsed:.2f}s")
print(f"{'='*70}\n")

if os.getenv("SPARK_DEBUG_UI", "0") == "1":
    print("\nSpark UI attiva. Apri http://localhost:4040 per vedere il DAG.")
    input("Premi INVIO per terminare l'applicazione...")

spark.stop()
