import math
import os
import sys
import time

from pyspark.sql.functions import col, floor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))


from utils_output import (
    show_rdd_result,
    save_rdd_csv_local,
    save_rdd_csv_hdfs,
)
from utils import build_spark_session

SPARK_MASTER            = os.getenv("SPARK_MASTER", "spark://spark-master:7077")
HDFS_INPUT              = "hdfs://namenode:9000/sabd/processed/"
HDFS_OUTPUT_PERCENTILES = "hdfs://namenode:9000/sabd/results/q3_rdd/percentiles/"
HDFS_OUTPUT_RANGE       = "hdfs://namenode:9000/sabd/results/q3_rdd/delay_range/"
LOCAL_OUT_PERCENTILES   = "/opt/results/q3_percentiles_rdd.csv"
LOCAL_OUT_RANGE         = "/opt/results/q3_delay_range_rdd.csv"
LOCAL_OUT               = "/opt/results"

COLS_PERC  = ["OP_UNIQUE_CARRIER", "hour", "p25", "p50", "p75", "p90"]
COLS_RANGE = ["OP_UNIQUE_CARRIER", "min_delay", "max_delay"]

os.makedirs(LOCAL_OUT, exist_ok=True)


# Percentile esatto su lista ordinata (interpolazione lineare). Eseguito sugli executor dentro mapValues.
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
    #attenzione perchè il values_iter è un iteratore non una lista, che prendiamo da groupbykey
    #in sorted avviene la vera materializzazione dell'iteratore in una listya
    vals = sorted(values_iter)
    return (
        round(percentile(vals, 0.25), 2),
        round(percentile(vals, 0.50), 2),
        round(percentile(vals, 0.75), 2),
        round(percentile(vals, 0.90), 2),
    )


def run(spark, benchmark=False):
    sc = spark.sparkContext

    # Timer attorno a costruzione del piano + azione: sortBy/sortByKey NON sono lazy,
    # lanciano un job di campionamento già in fase di definizione. Va quindi cronometrata
    # anche la costruzione della pipeline, altrimenti la misura escluderebbe il lavoro vero.
    t0 = time.time()

    #qui esplicitiamo il drop null su dep_delay nelle altre versioni no ma lo fa di default il percentile_approx
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

    # Base RDD (carrier, hour, delay) riusato per percentili e min/max.
    base = df.rdd.map(lambda r: (r["OP_UNIQUE_CARRIER"], int(r["hour"]), float(r["DEP_DELAY"])))

    # cache() per evitare di rileggere i dati due volte (percentili + range);
    # attivo anche in benchmark per misurare lo stesso path della produzione
    base = base.cache()

    # Percentili: groupByKey per (carrier, hour) calcolo esatto sul gruppo ordinato.
    percentile_rdd = (
        base
        .map(lambda t: ((t[0], t[1]), t[2]))               # chiave = (carrier, hour) valore = delay
        .groupByKey()                                       # raggruppa per (carrier, hour)
        .mapValues(quantiles)                               # calcola percentili sul gruppo solo sui valori, chiavi intatte
        .map(lambda kv: (kv[0][0], kv[0][1], *kv[1]))      # appiattisce il risultato
        .sortBy(lambda x: (x[0], x[1]))
    )

    # Min/Max DEP_DELAY per compagnia.
    range_rdd = (
        base
        .map(lambda t: (t[0], (t[2], t[2])))  # chiave solo carrier value = (ritardo, ritardo)
        .reduceByKey(lambda a, b: (min(a[0], b[0]), max(a[1], b[1]))) # calcolo effettivo del minimo e massimo sulle coppie
        .sortByKey() # qui ottieni quindi (carrier, (min,max))
    )

    # Caching dei risultati per evitare ricalcolo durante il salvataggio HDFS
    percentile_rdd.cache()
    range_rdd.cache()

    percentile_rows = percentile_rdd.collect()
    range_rows = [(c, mn, mx) for c, (mn, mx) in range_rdd.collect()]
    elapsed = time.time() - t0

    if benchmark:
        return elapsed

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

    save_rdd_csv_hdfs(
        sc=sc,
        path=HDFS_OUTPUT_PERCENTILES,
        header=COLS_PERC,
        data_rdd=percentile_rdd,
        row_mapper=lambda row: row, #appiattisce come fatto prima ma per rdd sul cluster
    )

    save_rdd_csv_hdfs(
        sc=sc,
        path=HDFS_OUTPUT_RANGE,
        header=COLS_RANGE,
        data_rdd=range_rdd,
        row_mapper=lambda kv: [
            kv[0],
            kv[1][0],
            kv[1][1],
        ],
    )

    print(f"\nTempo Q3 query/calcolo (RDD, esatto): {elapsed:.2f}s")
    print(f"{'='*70}\n")

    return elapsed


def main():
    spark = build_spark_session(
        app_name="Q3_RDD_Percentiles",
        master=SPARK_MASTER,
    )

    run(spark, benchmark=False)

    if os.getenv("SPARK_DEBUG_UI", "0") == "1":
        print("\nSpark UI attiva. Apri http://localhost:4040 per vedere il DAG.")
        input("Premi INVIO per terminare l'applicazione...")

    spark.stop()


if __name__ == "__main__":
    main()
