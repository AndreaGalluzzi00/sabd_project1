"""
Q3 — Percentili DEP_DELAY per compagnia e fascia oraria
Implementazione con KLL sketch via RDD API

KLL sketch (Karnin, Lang, Liberty — FOCS 2016): struttura a livelli a cascata.
È il primo algoritmo dimostrato ottimale per il problema dei quantili in streaming:
raggiunge il lower bound teorico sulla memoria O(1/ε · log(1/δ)) che nessun
algoritmo precedente (incluso Greenwald-Khanna) aveva provato di raggiungere.

A differenza di t-digest (errore adattivo sulle code), KLL garantisce errore
uniforme ε su tutti i percentili con probabilità 1-δ — garanzia probabilistica
dimostrabilmente ottimale.

La mergeability nativa di KLL permette la stessa pipeline distribuita di t-digest:
i singoli sketch vengono mergiati cross-partizione tramite serializzazione a byte,
senza mai aggregare i dati grezzi.

Parametro k (accuratezza): default 200
  - garanzia teorica: errore relativo sul rango ≈ 1.5 / k
  - k=200 → errore ≈ 0.75% — sufficiente per analisi operative
  - valori più alti → maggiore precisione → più memoria
"""
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import datasketches
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, floor

from utils_output import (
    show_rdd_result,
    save_rdd_csv_local,
)


SPARK_MASTER            = os.getenv("SPARK_MASTER", "spark://spark-master:7077")
HDFS_INPUT              = "hdfs://namenode:9000/sabd/processed/"
HDFS_OUTPUT_PERCENTILES = "hdfs://namenode:9000/sabd/results/q3_kll/percentiles/"
HDFS_OUTPUT_RANGE       = "hdfs://namenode:9000/sabd/results/q3_kll/delay_range/"
LOCAL_OUT_PERCENTILES   = "/opt/spark/jobs/results/q3_percentiles_kll.csv"
LOCAL_OUT_RANGE         = "/opt/spark/jobs/results/q3_delay_range_kll.csv"
LOCAL_OUT               = "/opt/spark/jobs/results"

K = 200  # parametro di accuratezza KLL

COLS_PERC  = ["OP_UNIQUE_CARRIER", "hour", "p25", "p50", "p75", "p90"]
COLS_RANGE = ["OP_UNIQUE_CARRIER", "min_delay", "max_delay"]

os.makedirs(LOCAL_OUT, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("Q3_KLL_Percentiles")
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

t0 = time.time()

# ── Percentili via RDD + KLL sketch ───────────────────────────────────────────
#
# Lo sketch KLL non è direttamente serializzabile da pickle (usato da Spark
# per i task Python), quindi lo serializziamo esplicitamente in bytes prima
# di ogni operazione cross-partizione e lo deserializziamo all'arrivo.
#
# combineByKey pipeline:
#   create_combiner : primo valore per una chiave → crea sketch, inserisce, serializza
#   merge_value     : valori successivi stessa partizione → deserializza, update, serializza
#   merge_combiners : riduzione tra partizioni → deserializza entrambi, merge, serializza
#
# Il merge KLL è il cuore dell'algoritmo: combina due sketch mantenendo la
# garanzia di errore senza accedere ai dati originali.

def create_combiner(value: float) -> bytes:
    sketch = datasketches.kll_floats_sketch(K)
    sketch.update(float(value))
    return sketch.serialize()

def merge_value(sketch_bytes: bytes, value: float) -> bytes:
    sketch = datasketches.kll_floats_sketch.deserialize(sketch_bytes)
    sketch.update(float(value))
    return sketch.serialize()

def merge_combiners(b1: bytes, b2: bytes) -> bytes:
    s1 = datasketches.kll_floats_sketch.deserialize(b1)
    s2 = datasketches.kll_floats_sketch.deserialize(b2)
    s1.merge(s2)
    return s1.serialize()

rdd = df.rdd.map(lambda row: (
    (row.OP_UNIQUE_CARRIER, int(row.hour)),
    float(row.DEP_DELAY)
))

percentile_rdd = (
    rdd
    .combineByKey(create_combiner, merge_value, merge_combiners)
    .map(lambda kv: (
        kv[0][0],                                                          # carrier
        kv[0][1],                                                          # hour
        round(datasketches.kll_floats_sketch.deserialize(kv[1]).get_quantile(0.25), 2),  # p25
        round(datasketches.kll_floats_sketch.deserialize(kv[1]).get_quantile(0.50), 2),  # p50
        round(datasketches.kll_floats_sketch.deserialize(kv[1]).get_quantile(0.75), 2),  # p75
        round(datasketches.kll_floats_sketch.deserialize(kv[1]).get_quantile(0.90), 2),  # p90
    ))
    .sortBy(lambda x: (x[0], x[1]))
)

# Min/Max: operazioni esatte, calcolate su DataFrame (no approssimazione)
# Min/Max DEP_DELAY per compagnia, calcolato con RDD.
range_rdd = (
    rdd
    .map(lambda kv: (kv[0][0], (kv[1], kv[1])))
    .reduceByKey(lambda a, b: (min(a[0], b[0]), max(a[1], b[1])))
    .map(lambda kv: (kv[0], kv[1][0], kv[1][1]))
    .sortBy(lambda x: x[0])
)

percentile_rows = percentile_rdd.collect()
range_rows = range_rdd.collect()
elapsed = time.time() - t0

show_rdd_result(
    rows=percentile_rows,
    header=COLS_PERC,
    query_name=f"Q3 KLL RDD — Percentili DEP_DELAY per compagnia e fascia oraria (k={K})",
    elapsed=elapsed,
)

show_rdd_result(
    rows=range_rows,
    header=COLS_RANGE,
    query_name=f"Q3 KLL RDD — Min/Max DEP_DELAY per compagnia (k={K})",
    elapsed=elapsed,
)

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
sc = spark.sparkContext

hadoop_conf = sc._jsc.hadoopConfiguration()
fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)

out_percentiles = sc._jvm.org.apache.hadoop.fs.Path(HDFS_OUTPUT_PERCENTILES)
out_range = sc._jvm.org.apache.hadoop.fs.Path(HDFS_OUTPUT_RANGE)

if fs.exists(out_percentiles):
    fs.delete(out_percentiles, True)

if fs.exists(out_range):
    fs.delete(out_range, True)

percentile_lines = (
    [",".join(COLS_PERC)] +
    [",".join(str(x) for x in row) for row in percentile_rows]
)

range_lines = (
    [",".join(COLS_RANGE)] +
    [",".join(str(x) for x in row) for row in range_rows]
)

sc.parallelize(percentile_lines, numSlices=1).saveAsTextFile(HDFS_OUTPUT_PERCENTILES)
sc.parallelize(range_lines, numSlices=1).saveAsTextFile(HDFS_OUTPUT_RANGE)

print(f"CSV su HDFS: {HDFS_OUTPUT_PERCENTILES}")
print(f"CSV su HDFS: {HDFS_OUTPUT_RANGE}")
print(f"\nTempo Q3 KLL (k={K}): {elapsed:.2f}s")
print(f"{'='*70}\n")
if os.getenv("SPARK_DEBUG_UI", "0") == "1":
    print("\nSpark UI attiva. Apri http://localhost:4040 per vedere il DAG.")
    input("Premi INVIO per terminare l'applicazione...")

spark.stop()
