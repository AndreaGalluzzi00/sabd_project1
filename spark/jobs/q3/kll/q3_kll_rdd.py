import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "utils")))

import datasketches
from pyspark.sql.functions import col, floor

from utils_output import (
    show_rdd_result,
    save_rdd_csv_local,
    save_rdd_csv_hdfs
)
from utils import build_spark_session


SPARK_MASTER            = os.getenv("SPARK_MASTER", "spark://spark-master:7077")
HDFS_INPUT              = "hdfs://namenode:9000/sabd/processed/"
HDFS_OUTPUT_PERCENTILES = "hdfs://namenode:9000/sabd/results/q3_kll/percentiles/"
HDFS_OUTPUT_RANGE       = "hdfs://namenode:9000/sabd/results/q3_kll/delay_range/"
LOCAL_OUT_PERCENTILES   = "/opt/results/q3_percentiles_kll.csv"
LOCAL_OUT_RANGE         = "/opt/results/q3_delay_range_kll.csv"
LOCAL_OUT               = "/opt/results"

K = 200  # parametro di accuratezza KLL

COLS_PERC  = ["OP_UNIQUE_CARRIER", "hour", "p25", "p50", "p75", "p90"]
COLS_RANGE = ["OP_UNIQUE_CARRIER", "min_delay", "max_delay"]

os.makedirs(LOCAL_OUT, exist_ok=True)


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

    rdd = df.rdd.map(lambda row: (
        (row.OP_UNIQUE_CARRIER, int(row.hour)),
        float(row.DEP_DELAY)
    ))

    # cache della sorgente: riusata da percentili e min/max (coerente con q3_rdd);
    # attiva anche in benchmark per misurare lo stesso path della produzione
    rdd = rdd.cache()
    t0 = time.time()
    # Percentili via RDD + KLL sketch
    percentile_rdd = (
        rdd
        .combineByKey(create_combiner, merge_value, merge_combiners)
        .map(lambda kv: (
            kv[0][0],                                                          # carrier
            kv[0][1],                                                          # hour
            # analogo a tdigest con l'aggiunta della deserialize
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

    # HDFS: salvataggio RDD con saveAsTextFile tramite utility comune.
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

    print(f"\nTempo Q3 KLL (k={K}): {elapsed:.2f}s")
    print(f"{'='*70}\n")

    return elapsed


def main():
    spark = build_spark_session(
        app_name="Q3_KLL_Percentiles",
        master=SPARK_MASTER,
    )

    run(spark, benchmark=False)

    if os.getenv("SPARK_DEBUG_UI", "0") == "1":
        print("\nSpark UI attiva. Apri http://localhost:4040 per vedere il DAG.")
        input("Premi INVIO per terminare l'applicazione...")

    spark.stop()


if __name__ == "__main__":
    main()
