"""
Q2 (RDD) — Top-10 compagnie per ARR_DELAY medio (gen-apr 2025)

Versione implementata con le API RDD (low-level) di Spark, da confrontare con
le versioni DataFrame (q2_df.py) e Spark SQL (q2_sql.py).

Filtro base: voli non cancellati e non deviati (CANCELLED=0, DIVERTED=0)
Soglia:      solo compagnie con >= 500 voli nel filtro base
Metriche:    num_flights, avg_arr_delay, media delle 5 cause di ritardo
NULL cause:  trattati come 0 (BTS li omette quando delay totale < 15min)
Ordine:      top-10 per avg_arr_delay decrescente

Pipeline RDD:
  read parquet → filter(base) → .rdd → map(to_pair) → reduceByKey(combine)
  → mapValues(finalize) → filter(>=500) → takeOrdered(TOP_N)  ← action

Il costo dominante è a monte (scan Parquet + shuffle di reduceByKey); takeOrdered
in sé è leggero. Lo usiamo come analogo RDD di ORDER BY ... LIMIT delle versioni
DataFrame/SQL: è il pattern corretto e scalabile (top-N distribuito, niente
collect dell'intero risultato sul driver). In Q2 il vantaggio è marginale, perché
la cardinalità di OP_UNIQUE_CARRIER è ~15-20 compagnie, ma mantiene la coerenza
tra le tre implementazioni.

Per il salvataggio HDFS, la top-10 viene trasformata in un piccolo RDD e salvata
con saveAsTextFile tramite utility comune, mantenendo lo stile RDD di q1_rdd.py.

Modalità:
  - Dev locale (Mac M1):  SPARK_MASTER=local[2]  (default)
  - Cluster / EC2:        SPARK_MASTER=spark://spark-master:7077
"""

import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))

from pyspark.sql.functions import col

from utils_output import (
    show_rdd_result,
    save_rdd_csv_local,
    save_rdd_csv_hdfs,
)
from utils import build_spark_session


SPARK_MASTER = os.getenv("SPARK_MASTER", "spark://spark-master:7077")
HDFS_INPUT   = "hdfs://namenode:9000/sabd/processed/"
HDFS_OUTPUT  = "hdfs://namenode:9000/sabd/results/q2_rdd/"
LOCAL_OUT    = "/opt/results/q2_rdd.csv"

# Stesso ordine di colonne delle versioni DataFrame/SQL → direttamente confrontabile.
HEADER = [
    "OP_UNIQUE_CARRIER", "num_flights", "avg_arr_delay",
    "avg_carrier_delay", "avg_weather_delay", "avg_nas_delay",
    "avg_security_delay", "avg_late_aircraft_delay",
]

MIN_FLIGHTS = 500
TOP_N = 10

os.makedirs(os.path.dirname(LOCAL_OUT), exist_ok=True)

spark = build_spark_session(
    app_name="Q2_RDD_top10_carriers_arr_delay",
    master=SPARK_MASTER,
    ui_enabled=True,
    ui_port="4040",
)

sc = spark.sparkContext


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

      - total          : conta TUTTI i voli del filtro base
      - arr_sum/arr_n  : solo ARR_DELAY non-null → avg_arr_delay ignora i null,
                         come AVG() in DataFrame/SQL
      - le 5 cause     : NULL→0, sommate sul totale dei voli

    Il valore ha già la forma dell'aggregato finale, così reduceByKey può
    combinare due valori con la stessa funzione.
    """
    arr = r["ARR_DELAY"]

    if arr is not None:
        arr_sum, arr_n = float(arr), 1
    else:
        arr_sum, arr_n = 0.0, 0

    return r["OP_UNIQUE_CARRIER"], (
        1,
        arr_sum,
        arr_n,
        _zero(r["CARRIER_DELAY"]),
        _zero(r["WEATHER_DELAY"]),
        _zero(r["NAS_DELAY"]),
        _zero(r["SECURITY_DELAY"]),
        _zero(r["LATE_AIRCRAFT_DELAY"]),
    )


def combine(a, b):
    """Fonde due aggregati parziali della stessa compagnia."""
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
    """Aggregato grezzo → metriche finali."""
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


def top_key(kv):
    """Chiave di ordinamento per takeOrdered.

    takeOrdered ordina in modo crescente.
    Per ottenere avg_arr_delay decrescente usiamo il valore negativo.

    Esempio:
      avg_arr_delay = 20 → key = -20
      avg_arr_delay = 10 → key = -10

    Ordinando crescente, -20 viene prima di -10, quindi il delay maggiore
    finisce in cima.

    Se avg_arr_delay è None, assegniamo +inf così finisce in fondo.
    """
    avg_arr_delay = kv[1][1]
    return -avg_arr_delay if avg_arr_delay is not None else float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline lazy — nessun calcolo finché non scatta l'action
# ─────────────────────────────────────────────────────────────────────────────
# Filtro base sul DataFrame per sfruttare predicate pushdown e column pruning
# del Parquet; poi passaggio all'RDD.
base_df = (
    spark.read.parquet(HDFS_INPUT)
    .filter((col("CANCELLED") == 0) & (col("DIVERTED") == 0))
    .select(
        "OP_UNIQUE_CARRIER",
        "ARR_DELAY",
        "CARRIER_DELAY",
        "WEATHER_DELAY",
        "NAS_DELAY",
        "SECURITY_DELAY",
        "LATE_AIRCRAFT_DELAY",
    )
)

rdd = (
    base_df.rdd
    .map(to_pair)
    .reduceByKey(combine)
    .mapValues(finalize)
    .filter(lambda kv: kv[1][0] >= MIN_FLIGHTS)
)


# ─────────────────────────────────────────────────────────────────────────────
# Action: takeOrdered() → calcola direttamente la top-10
# ─────────────────────────────────────────────────────────────────────────────
# Questa action fa scattare l'intera pipeline (il peso è lo scan Parquet + lo
# shuffle di reduceByKey, non il takeOrdered). Rispetto a collect() + sorted(),
# takeOrdered tiene su ogni partizione solo i TOP_N migliori e li fonde sul
# driver: in Q2 il risparmio è trascurabile (l'RDD aggregato ha ~15-20 righe),
# ma è l'analogo RDD di ORDER BY ... LIMIT e resta coerente con DataFrame/SQL.
t0 = time.time()

top = rdd.takeOrdered(
    TOP_N,
    key=top_key,
)

elapsed = time.time() - t0

# Versione piatta per stampa e CSV locale.
rows = [
    [carrier, *metrics]
    for carrier, metrics in top
]

# RDD finale piccolo, usato solo per il salvataggio HDFS in stile RDD.
top_rdd = sc.parallelize(top, numSlices=1)


# ─────────────────────────────────────────────────────────────────────────────
# Output a schermo + CSV locale
# ─────────────────────────────────────────────────────────────────────────────
show_rdd_result(
    rows=rows,
    header=HEADER,
    query_name="Q2 RDD",
    elapsed=elapsed,
)

save_rdd_csv_local(
    path=LOCAL_OUT,
    header=HEADER,
    rows=rows,
)


# ─────────────────────────────────────────────────────────────────────────────
# HDFS: salvataggio RDD con saveAsTextFile tramite utility comune
# ─────────────────────────────────────────────────────────────────────────────
save_rdd_csv_hdfs(
    sc=sc,
    path=HDFS_OUTPUT,
    header=HEADER,
    data_rdd=top_rdd,
    row_mapper=lambda kv: [
        kv[0],
        *kv[1],
    ],
)

print(f"{'='*70}\n")

if os.getenv("SPARK_DEBUG_UI", "0") == "1":
    print("\nSpark UI attiva. Apri http://localhost:4040 per vedere il DAG.")
    input("Premi INVIO per terminare l'applicazione...")

spark.stop()