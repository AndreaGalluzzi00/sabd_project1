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
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "utils")))

from pyspark.sql.functions import col, floor, min as spark_min, max as spark_max
from tdigest import TDigest

from utils_output import (
    show_rdd_result,
    save_rdd_csv_local,
    save_rdd_csv_hdfs
)
from utils import build_spark_session

SPARK_MASTER            = os.getenv("SPARK_MASTER", "spark://spark-master:7077")
HDFS_INPUT              = "hdfs://namenode:9000/sabd/processed/"
HDFS_OUTPUT_PERCENTILES = "hdfs://namenode:9000/sabd/results/q3_tdigest/percentiles/"
HDFS_OUTPUT_RANGE       = "hdfs://namenode:9000/sabd/results/q3_tdigest/delay_range/"
LOCAL_OUT_PERCENTILES   = "/opt/spark/jobs/results/q3_percentiles_tdigest.csv"
LOCAL_OUT_RANGE         = "/opt/spark/jobs/results/q3_delay_range_tdigest.csv"
LOCAL_OUT               = "/opt/spark/jobs/results"

os.makedirs(LOCAL_OUT, exist_ok=True)
COLS_PERC  = ["OP_UNIQUE_CARRIER", "hour", "p25", "p50", "p75", "p90"]
COLS_RANGE = ["OP_UNIQUE_CARRIER", "min_delay", "max_delay"]

spark = build_spark_session(
    app_name="Q3_TDigest_Percentiles",
    master=SPARK_MASTER,
    ui_enabled=True,
    ui_port="4040",
)
sc = spark.sparkContext

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
#ATTENZIONE QUA MODIFICATO RISPETTO A IERI
range_rdd = (
    rdd
    .map(lambda kv: (kv[0][0], (kv[1], kv[1])))
    .reduceByKey(lambda a, b: (min(a[0], b[0]), max(a[1], b[1])))
    .map(lambda kv: (kv[0], kv[1][0], kv[1][1]))
    .sortBy(lambda x: x[0])
)
percentile_rdd.cache()
range_rdd.cache()

percentile_rows = percentile_rdd.collect()
range_rows = range_rdd.collect()
elapsed = time.time() - t0

# ─────────────────────────────────────────────────────────────────────────────
# Output a schermo + CSV locale
# ─────────────────────────────────────────────────────────────────────────────
show_rdd_result(
    rows=percentile_rows,
    header=COLS_PERC,
    query_name="Q3 t-digest RDD — Percentili DEP_DELAY per compagnia e fascia oraria",
    elapsed=elapsed,
)



show_rdd_result(
    rows=range_rows,
    header=COLS_RANGE,
    query_name="Q3 t-digest RDD — Min/Max DEP_DELAY per compagnia",
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

# ─────────────────────────────────────────────────────────────────────────────
# HDFS: salvataggio RDD puro con saveAsTextFile
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# HDFS: salvataggio RDD con saveAsTextFile tramite utility comune
# ─────────────────────────────────────────────────────────────────────────────
save_rdd_csv_hdfs(
    sc=sc,
    path=HDFS_OUTPUT_PERCENTILES,
    header=COLS_PERC,
    data_rdd=percentile_rdd,
    row_mapper=lambda row: row,
)

save_rdd_csv_hdfs(
    sc=sc,
    path=HDFS_OUTPUT_RANGE,
    header=COLS_RANGE,
    data_rdd=range_rdd,
    row_mapper=lambda row: row,
)

print(f"\nTempo Q3 t-digest RDD: {elapsed:.2f}s")
print(f"{'=' * 70}\n")
if os.getenv("SPARK_DEBUG_UI", "0") == "1":
    print("\nSpark UI attiva. Apri http://localhost:4040 per vedere il DAG.")
    input("Premi INVIO per terminare l'applicazione...")

spark.stop()