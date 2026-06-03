import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))

from pyspark.sql.functions import (
    col, avg, min as spark_min, max as spark_max,
    count, sum as spark_sum, when, round as spark_round,
)

from utils_output import (
    show_dataframe_result,
    save_csv_local,
    save_dataframe_hdfs,
)
from utils import build_spark_session

SPARK_MASTER = os.getenv("SPARK_MASTER","spark://spark-master:7077")
HDFS_INPUT   = "hdfs://namenode:9000/sabd/processed/"
HDFS_OUTPUT  = "hdfs://namenode:9000/sabd/results/q1_df/"
LOCAL_OUT    = "/opt/results/q1_df.csv"

os.makedirs(os.path.dirname(LOCAL_OUT), exist_ok=True)


def main():

    spark = build_spark_session(
        app_name="Q1_AA_DL_monthly_stats",
        master=SPARK_MASTER,
    )

    # Lettura
    df = (
        spark.read.parquet(HDFS_INPUT)
        .filter(col("OP_UNIQUE_CARRIER").isin("AA", "DL"))
    )

    ### Calcolo Q1

    # Calcolo tempo iniziale
    t0 = time.time()

    result = (
        df.groupBy("OP_UNIQUE_CARRIER", "YEAR", "MONTH")
        .agg(
            count("*").alias("total_flights"),
            spark_sum(col("CANCELLED").cast("long")).alias("cancelled_flights"),
            spark_round(
                spark_sum("CANCELLED") / count("*") * 100, 4
            ).alias("cancellation_rate_pct"),
            spark_round(
                avg(when(col("CANCELLED") == 0, col("DEP_DELAY"))), 4
            ).alias("avg_dep_delay"),
            spark_round(
                spark_min(when(col("CANCELLED") == 0, col("DEP_DELAY"))), 4
            ).alias("min_dep_delay"),
            spark_round(
                spark_max(when(col("CANCELLED") == 0, col("DEP_DELAY"))), 4
            ).alias("max_dep_delay"),
        )
        .orderBy("OP_UNIQUE_CARRIER", "YEAR", "MONTH")
    )

    # Caching
    result.cache()

    # Esecuzione
    rows = result.collect()

    # Calcolo tempo trascorso
    elapsed = time.time() - t0

    # Output a schermo
    show_dataframe_result(result, "Q1", elapsed, 20)

    # Salvataggio CSV locale (directory montata → visibile sull'host)
    save_csv_local(LOCAL_OUT, result, rows)

    # Salvataggio su HDFS
    save_dataframe_hdfs(result, HDFS_OUTPUT)

    print(f"\nTempo Q1 (DataFrame API): {elapsed:.2f}s")
    print(f"{'='*70}\n")

    spark.stop()


if __name__ == "__main__":
    main()
