import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))

from utils_output import (
    show_rdd_result,
    save_rdd_csv_local,
    save_rdd_csv_hdfs
)
from utils import build_spark_session

SPARK_MASTER = os.getenv("SPARK_MASTER", "spark://spark-master:7077")
DEFAULT_HDFS_INPUT = "hdfs://namenode:9000/sabd/processed_single/"
HDFS_OUTPUT  = "hdfs://namenode:9000/sabd/results/q1_rdd/"
LOCAL_OUT    = "/opt/results/q1_rdd.csv"

# HEADER corrisponde all'intestazione del CSV.
# Stesso ordine delle versioni DataFrame/SQL (per confronto).
HEADER = [
    "OP_UNIQUE_CARRIER",
    "YEAR",
    "MONTH",
    "total_flights",
    "cancelled_flights",
    "cancellation_rate_pct",
    "avg_dep_delay",
    "min_dep_delay",
    "max_dep_delay",
]

def to_key_value(r):

    # Chiave di aggregazione, in previsione di reduceByKey
    key = (r["OP_UNIQUE_CARRIER"], r["YEAR"], r["MONTH"])

    # Se il volo è cancellazione, allora conta su TUTTI i voli.
    # CANCELLED è double (0.0/1.0), ma può essere anche null (None).
    cancelled = 1 if (r["CANCELLED"] is not None and r["CANCELLED"] == 1.0) else 0

    dep = r["DEP_DELAY"]
    if cancelled == 0 and dep is not None:
        # Volo non cancellato, per cui contribuisce con il suo ritardo e vale come una singola osservazione.

        dep = float(dep)
        delay_sum, delay_n, delay_min, delay_max = dep, 1, dep, dep
    else:
        # Volo cancellato o senza ritardo, non contribuisce alla somma né al conteggio dei ritardi.
        # Si tratta comunque di un volo valido per il totale e le cancellazioni.

        delay_sum, delay_n, delay_min, delay_max = 0.0, 0, float("inf"), float("-inf") # +inf/-inf come elementi neutri di min/max

    return key, (1, cancelled, delay_sum, delay_n, delay_min, delay_max)


def combine(a, b):

    return (
        a[0] + b[0],        # total_flights
        a[1] + b[1],        # cancelled_flights
        a[2] + b[2],        # delay_sum
        a[3] + b[3],        # delay_n
        min(a[4], b[4]),    # delay_min
        max(a[5], b[5]),    # delay_max
    )


def finalize(v):

    total, cancelled, delay_sum, delay_n, delay_min, delay_max = v

    cancellation_rate = round(cancelled / total * 100, 4)

    if delay_n > 0:
        avg_delay = round(delay_sum / delay_n, 4)
        min_delay, max_delay = delay_min, delay_max
    else:
        # Nessun volo valido: evita /0 e gli +inf/-inf neutri.
        avg_delay = min_delay = max_delay = None

    return total, cancelled, cancellation_rate, avg_delay, min_delay, max_delay

def run(spark, benchmark=False, hdfs_input=None):
    input_path = hdfs_input or os.getenv("HDFS_INPUT", DEFAULT_HDFS_INPUT)
    sc = spark.sparkContext

    ### Calcolo Q1 con RDD ###

    # Spark costruisce solo il piano di esecuzione (lineage), trattandosi di Trasformazioni (lazy).
    result_rdd = (
        spark.read.parquet(input_path)
        .select("OP_UNIQUE_CARRIER", "YEAR", "MONTH", "CANCELLED", "DEP_DELAY") # Leggiamo solo le 5 colonne utili (column pruning del Parquet)
        .rdd
        .filter(lambda r: r["OP_UNIQUE_CARRIER"] in ("AA", "DL"))  # solo AA/DL
        .map(to_key_value) # passaggio a struttura (chiave, valore parziale)
        .reduceByKey(combine) # combinazione basata su aggregazione
        .mapValues(finalize) # applicazione funzione di finalizzazione senza modificare le chiavi
    )

    if not benchmark:
        # cache(): chiediamo a Spark di TENERE IN MEMORIA il risultato di questa RDD,
        # perché lo riutilizzeremo per due azioni consecutive (collect e salvataggio su HDFS).
        result_rdd.cache()

    # Calcolo tempo iniziale
    t0 = time.time()

    # Esecuzione
    collected = result_rdd.collect()

    # Calcolo tempo trascorso
    elapsed = time.time() - t0

    if benchmark:
        return elapsed

    # (chiave, valore) -> riga piatta, ordinata per (carrier, year, month).
    rows = [
        [carrier, year, month, *metrics]
        for (carrier, year, month), metrics in sorted(collected)
    ]

    # Output a schermo
    show_rdd_result(
        rows=rows,
        header=HEADER,
        query_name="Q1 RDD",
        elapsed=elapsed,
    )

    # Salvataggio CSV locale
    save_rdd_csv_local(
        path=LOCAL_OUT,
        header=HEADER,
        rows=rows,
    )

    # Salvataggio su HDFS (riusa il risultato in cache, per cui niente ricalcolo)
    save_rdd_csv_hdfs(
        sc=sc,
        path=HDFS_OUTPUT,
        header=HEADER,
        data_rdd=result_rdd.sortByKey(),
        row_mapper=lambda kv: [
            kv[0][0],
            kv[0][1],
            kv[0][2],
            *kv[1],
        ],
    )
    print(f"{'='*70}\n")

    return elapsed


def main():

    spark = build_spark_session(
        app_name="Q1_RDD_AA_DL_monthly_stats",
        master=SPARK_MASTER,
    )

    run(spark, benchmark=False)

    if os.getenv("SPARK_DEBUG_UI", "0") == "1":
        print("\nSpark UI attiva. Apri http://localhost:4040 per vedere il DAG.")
        input("Premi INVIO per terminare l'applicazione...")

    spark.stop()


if __name__ == "__main__":
    main()