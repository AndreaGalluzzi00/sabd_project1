"""
Q2 (RDD) — Top-10 compagnie per ARR_DELAY medio (gen-apr 2025)

Versione implementata con le API RDD (low-level) di Spark, da confrontare con
le versioni DataFrame (q2.py) e Spark SQL (q2_sql.py).

Filtro base: voli non cancellati e non deviati (CANCELLED=0, DIVERTED=0)
Soglia:      solo compagnie con >= 500 voli nel filtro base
Metriche:    num_flights, avg_arr_delay, media delle 5 cause di ritardo
NULL cause:  trattati come 0 (BTS li omette quando delay totale < 15min)
Ordine:      top-10 per avg_arr_delay decrescente

Pipeline RDD (tutta lazy, una sola action pesante):
  read parquet → filter(base) → .rdd → map(to_pair) → reduceByKey(combine)
  → mapValues(finalize) → filter(>=500) → collect()  ← action
La top-10 (ordinamento + limit) è fatta in Python sul driver: poche compagnie.

Modalità:
  - Dev locale (Mac M1):  SPARK_MASTER=local[2]  (default)
  - Cluster / EC2:        SPARK_MASTER=spark://spark-master:7077
"""
import csv
import os
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

SPARK_MASTER = os.getenv("SPARK_MASTER", "spark://spark-master:7077")
HDFS_INPUT   = "hdfs://namenode:9000/sabd/processed/"
HDFS_OUTPUT  = "hdfs://namenode:9000/sabd/results/q2_rdd/"
LOCAL_OUT    = "/opt/spark/jobs/results/q2_rdd.csv"

# Stesso ordine di colonne delle versioni DataFrame/SQL → direttamente confrontabile.
HEADER = [
    "OP_UNIQUE_CARRIER", "num_flights", "avg_arr_delay",
    "avg_carrier_delay", "avg_weather_delay", "avg_nas_delay",
    "avg_security_delay", "avg_late_aircraft_delay",
]

MIN_FLIGHTS = 500
TOP_N = 10

os.makedirs(os.path.dirname(LOCAL_OUT), exist_ok=True)

spark = (
    SparkSession.builder
    .appName("Q2_RDD_top10_carriers_arr_delay")
    .master(SPARK_MASTER)
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print(f"Master: {SPARK_MASTER}")


# ─────────────────────────────────────────────────────────────────────────────
# Funzioni della pipeline RDD
# ─────────────────────────────────────────────────────────────────────────────

def _zero(x):
    """NULL → 0.0 (BTS lascia NULL le cause quando il ritardo totale < 15min)."""
    return float(x) if x is not None else 0.0


def to_pair(r):
    """Row → (carrier, aggregato_parziale di UNA riga).

    CHIAVE = OP_UNIQUE_CARRIER   ← il GROUP BY di Q2
    VALORE = (total, arr_sum, arr_n, carrier, weather, nas, security, late)
      - total          : conta TUTTI i voli del filtro base (denominatore cause)
      - arr_sum/arr_n  : solo ARR_DELAY non-null → avg_arr_delay ignora i null,
                         come AVG() in DataFrame/SQL
      - le 5 cause     : NULL→0, sommate sul totale dei voli (coalesce)

    Il valore ha già la forma dell'aggregato finale (combiner pattern), così
    reduceByKey può combinare due valori con la stessa funzione.
    """
    arr = r["ARR_DELAY"]
    if arr is not None:
        arr_sum, arr_n = float(arr), 1
    else:
        arr_sum, arr_n = 0.0, 0

    return r["OP_UNIQUE_CARRIER"], (
        1, arr_sum, arr_n,
        _zero(r["CARRIER_DELAY"]),
        _zero(r["WEATHER_DELAY"]),
        _zero(r["NAS_DELAY"]),
        _zero(r["SECURITY_DELAY"]),
        _zero(r["LATE_AIRCRAFT_DELAY"]),
    )


def combine(a, b):
    """Fonde due aggregati parziali della stessa compagnia (associativo + commutativo)."""
    return (
        a[0] + b[0],   # total
        a[1] + b[1],   # arr_sum
        a[2] + b[2],   # arr_n
        a[3] + b[3],   # carrier_delay sum
        a[4] + b[4],   # weather_delay sum
        a[5] + b[5],   # nas_delay sum
        a[6] + b[6],   # security_delay sum
        a[7] + b[7],   # late_aircraft_delay sum
    )


def finalize(v):
    """Aggregato grezzo → metriche finali (medie: operazioni non associative)."""
    total, arr_sum, arr_n, carrier, weather, nas, security, late = v
    avg_arr = round(arr_sum / arr_n, 4) if arr_n > 0 else None
    return (
        total,
        avg_arr,
        round(carrier / total, 4),
        round(weather / total, 4),
        round(nas / total, 4),
        round(security / total, 4),
        round(late / total, 4),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline lazy — nessun calcolo finché non scatta l'action finale
# ─────────────────────────────────────────────────────────────────────────────
# Filtro base sul DataFrame (predicate pushdown) + column pruning del Parquet,
# poi passaggio all'RDD: map/reduceByKey/mapValues/filter sono tutte lazy.
df = (
    spark.read.parquet(HDFS_INPUT)
    .filter((col("CANCELLED") == 0) & (col("DIVERTED") == 0))
    .select(
        "OP_UNIQUE_CARRIER", "ARR_DELAY",
        "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
        "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
    )
)

aggregated = (
    df.rdd
    .map(to_pair)                                    # → (carrier, parziale)
    .reduceByKey(combine)                            # SHUFFLE + map-side combine
    .mapValues(finalize)                             # medie finali (no shuffle)
    .filter(lambda kv: kv[1][0] >= MIN_FLIGHTS)      # soglia >= 500 voli
)

# ─────────────────────────────────────────────────────────────────────────────
# Action: collect() → risultato minuscolo (≈14 compagnie) + misura del tempo
# ─────────────────────────────────────────────────────────────────────────────
t0 = time.time()
collected = aggregated.collect()
elapsed = time.time() - t0

# Top-10 per avg_arr_delay decrescente (sul driver: pochi elementi → niente
# sortBy distribuito). avg_arr None (mai per i top) finisce in fondo.
top = sorted(
    collected,
    key=lambda kv: kv[1][1] if kv[1][1] is not None else float("-inf"),
    reverse=True,
)[:TOP_N]

rows = [[carrier, *metrics] for carrier, metrics in top]

# ─────────────────────────────────────────────────────────────────────────────
# Output a schermo + CSV locale (dir montata → visibile sull'host)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Q2 (RDD) — RISULTATI  (tempo esecuzione: {elapsed:.2f}s)")
print(f"{'='*70}")
print("  ".join(HEADER))
for row in rows:
    print("  ".join(str(x) for x in row))

with open(LOCAL_OUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(HEADER)
    writer.writerows(rows)
print(f"\nCSV locale:  {LOCAL_OUT}")

# ─────────────────────────────────────────────────────────────────────────────
# HDFS: saveAsTextFile (coerente con q1_rdd.py — niente DataFrame writer)
# ─────────────────────────────────────────────────────────────────────────────
sc = spark.sparkContext

# saveAsTextFile NON sovrascrive: cancelliamo prima l'eventuale output esistente
# con le API Hadoop FileSystem (equivalente di mode("overwrite") del writer DF).
hadoop_conf = sc._jsc.hadoopConfiguration()
fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
out_path = sc._jvm.org.apache.hadoop.fs.Path(HDFS_OUTPUT)
if fs.exists(out_path):
    fs.delete(out_path, True)  # True = ricorsivo (è una directory)

# Header come prima riga di un RDD a singola partizione → un solo file part-00000
# con le righe nell'ordine della top-10.
lines = [",".join(HEADER)] + [",".join(str(x) for x in row) for row in rows]
sc.parallelize(lines, numSlices=1).saveAsTextFile(HDFS_OUTPUT)
print(f"CSV su HDFS: {HDFS_OUTPUT}")

print(f"\nTempo Q2 (RDD API): {elapsed:.2f}s")
print(f"{'='*70}\n")

spark.stop()
