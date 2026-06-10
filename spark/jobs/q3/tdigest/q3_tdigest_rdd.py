import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "utils")))

from pyspark.sql.functions import col, floor
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
LOCAL_OUT_PERCENTILES   = "/opt/results/q3_percentiles_tdigest.csv"
LOCAL_OUT_RANGE         = "/opt/results/q3_delay_range_tdigest.csv"
LOCAL_OUT               = "/opt/results"

os.makedirs(LOCAL_OUT, exist_ok=True)
COLS_PERC  = ["OP_UNIQUE_CARRIER", "hour", "p25", "p50", "p75", "p90"]
COLS_RANGE = ["OP_UNIQUE_CARRIER", "min_delay", "max_delay"]


# Percentili via RDD + t-digest
# combineByKey preferibile ad aggregateByKey perché usa una factory
# (create_combiner) invece di un valore zero condiviso — evita il problema
# di mutare lo stesso oggetto TDigest su chiavi diverse nella stessa partizione.

# Pipeline di combining
#   create_combiner : prima occorrenza di una chiave crea un nuovo TDigest
#   merge_value     : occorrenze successive nella stessa partizione update
#   merge_combiners : riduzione tra partizioni merge dei centroidi

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


def run(spark, benchmark=False):
    sc = spark.sparkContext

    # Timer attorno a costruzione del piano + azione: sortBy/sortByKey NON sono lazy,
    # lanciano un job di campionamento già in fase di definizione. Va quindi cronometrata
    # anche la costruzione della pipeline, altrimenti la misura escluderebbe il lavoro vero.


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

    rdd = df.rdd.map(lambda r: (
        r["OP_UNIQUE_CARRIER"],
        int(r["hour"]),
        float(r["DEP_DELAY"])
    ))

    rdd = rdd.cache()
    t0 = time.time()

    # cache della sorgente: riusata da percentili e min/max (coerente con q3_rdd);
    # attiva anche in benchmark per misurare lo stesso path della produzione


    #perchè combine by key e non aggregate by key?
    #aggregate crea lo zero value una volta per partizione ma lo riusa per tutti i calcoli della stessa andando quindi a sprcare il risultato
    # la struttura qui è ((carrier, hour ), tdigest_finale)
    percentile_rdd = (
        rdd
        .map(lambda t: ((t[0], t[1]), t[2]))
        .combineByKey(create_combiner, merge_value, merge_combiners)
        .sortByKey()
        .map(lambda kv: (
            kv[0][0],
            kv[0][1],
            round(kv[1].percentile(25), 2),
            round(kv[1].percentile(50), 2),
            round(kv[1].percentile(75), 2),
            round(kv[1].percentile(90), 2),
        ))
    )

    # Min/Max: operazioni esatte, calcolate su DataFrame (no approssimazione)
    range_rdd = (
        rdd
        .map(lambda t: (t[0], (t[2], t[2])))
        .reduceByKey(lambda a, b: (min(a[0], b[0]), max(a[1], b[1])))
        .sortByKey()
        .map(lambda kv: (kv[0], kv[1][0], kv[1][1]))
    )

    # Caching dei risultati per evitare ricalcolo durante il salvataggio HDFS
    percentile_rdd.cache()
    range_rdd.cache()

    percentile_rows = percentile_rdd.collect()
    range_rows = range_rdd.collect()
    elapsed = time.time() - t0

    if benchmark:
        rdd.unpersist()
        percentile_rdd.unpersist()
        range_rdd.unpersist()
        return elapsed

    # Output a schermo + CSV locale
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

    # HDFS: salvataggio RDD con saveAsTextFile tramite utility comune
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

    return elapsed


def main():
    spark = build_spark_session(
        app_name="Q3_TDigest_Percentiles",
        master=SPARK_MASTER,
    )

    run(spark, benchmark=False)

    if os.getenv("SPARK_DEBUG_UI", "0") == "1":
        print("\nSpark UI attiva. Apri http://localhost:4040 per vedere il DAG.")
        input("Premi INVIO per terminare l'applicazione...")

    spark.stop()


if __name__ == "__main__":
    main()
