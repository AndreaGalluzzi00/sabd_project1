import sys, os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))


from pyspark.sql.functions import col, floor, expr, min as spark_min, max as spark_max
from utils_output import (
    show_dataframe_result,
    save_csv_local,
    save_dataframe_hdfs,
)
from utils import build_spark_session

SPARK_MASTER = os.getenv("SPARK_MASTER","spark://spark-master:7077")
HDFS_INPUT = "hdfs://namenode:9000/sabd/processed/"
HDFS_OUTPUT_PERCENTILES = "hdfs://namenode:9000/sabd/results/q3_df/percentiles/"
HDFS_OUTPUT_RANGE = "hdfs://namenode:9000/sabd/results/q3_df/delay_range/"
LOCAL_OUT_PERCENTILES   = "/opt/results/q3_df_percentiles.csv"
LOCAL_OUT_RANGE = "/opt/results/q3_df_delay_range.csv"
LOCAL_OUT = "/opt/results"

os.makedirs(LOCAL_OUT, exist_ok=True)


def run(spark, benchmark=False):
    df = (
        spark.read.parquet(HDFS_INPUT)
        .filter(
            (col("OP_UNIQUE_CARRIER").isin("AA", "DL", "UA", "WN")) &
            (col("CANCELLED") == 0)
        )
        .withColumn("hour", floor(col("CRS_DEP_TIME") / 100))
        #crea una nuova colonna con alias hour
    )

    #calcolo percentili sempre tramite percentile_approx, efficiente
    percentiles = (
        df.groupBy("OP_UNIQUE_CARRIER", "hour")
        .agg(
            expr("percentile_approx(DEP_DELAY, 0.25)").alias("p25"),
            expr("percentile_approx(DEP_DELAY, 0.50)").alias("p50"),
            expr("percentile_approx(DEP_DELAY, 0.75)").alias("p75"),
            expr("percentile_approx(DEP_DELAY, 0.90)").alias("p90"),
        )
        .orderBy("OP_UNIQUE_CARRIER", "hour")
    )

    #Query separata per il calcolo di minimo e massimo come negli altri casi, abbiamo un livello di granularità diverso sul group by ed inoltre una cardinalità differente nei risultati
    delay_range = (
        df.groupBy("OP_UNIQUE_CARRIER")
        .agg(
            spark_min("DEP_DELAY").alias("min_delay"),
            spark_max("DEP_DELAY").alias("max_delay"),
        )
        .orderBy("OP_UNIQUE_CARRIER")
    )

    #caching per evitarne il ricalcolo (attivo anche in benchmark)
    percentiles.cache()
    delay_range.cache()

    t0 = time.time()
    percentile_rows = percentiles.collect()
    range_rows = delay_range.collect()
    elapsed = time.time() - t0

    if benchmark:
        return elapsed

    # Output a schermo
    show_dataframe_result(
        result=percentiles,
        query_name="Q3 — Percentili DEP_DELAY per compagnia e fascia oraria",
        elapsed=elapsed,
        n=100,
    )

    show_dataframe_result(
        result=delay_range,
        query_name="Q3 — Min/Max DEP_DELAY per compagnia",
        elapsed=elapsed,
        n=20,
    )

    # Salvataggio CSV
    save_csv_local(
        path=LOCAL_OUT_PERCENTILES,
        result=percentiles,
        rows=percentile_rows,
    )

    save_csv_local(
        path=LOCAL_OUT_RANGE,
        result=delay_range,
        rows=range_rows,
    )

    #Salvataggio anche in remoto su hdfs
    save_dataframe_hdfs(percentiles, HDFS_OUTPUT_PERCENTILES)
    save_dataframe_hdfs(delay_range, HDFS_OUTPUT_RANGE)

    print(f"\nTempo Q3: {elapsed:.2f}s")
    print(f"{'=' * 70}\n")

    return elapsed


def main():
    spark = build_spark_session(
        app_name="Q3_Percentiles_by_Hour",
        master=SPARK_MASTER,
    )

    run(spark, benchmark=False)

    spark.stop()


if __name__ == "__main__":
    main()
