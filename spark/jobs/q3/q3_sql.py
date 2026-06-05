import sys, os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))

from utils_output import (
    show_dataframe_result,
    save_csv_local,
    save_dataframe_hdfs,
)
from utils import build_spark_session


SPARK_MASTER = os.getenv("SPARK_MASTER","spark://spark-master:7077")
HDFS_INPUT = "hdfs://namenode:9000/sabd/processed/"
HDFS_OUTPUT_PERCENTILES = "hdfs://namenode:9000/sabd/results/q3_sql/percentiles/"
HDFS_OUTPUT_RANGE = "hdfs://namenode:9000/sabd/results/q3_sql/delay_range/"
LOCAL_OUT_PERCENTILES = "/opt/results/q3_percentiles_sql.csv"
LOCAL_OUT_RANGE = "/opt/results/q3_delay_range_sql.csv"
LOCAL_OUT = "/opt/results"

os.makedirs(LOCAL_OUT, exist_ok=True)
os.makedirs(os.path.dirname(LOCAL_OUT_PERCENTILES), exist_ok=True)


def run(spark, benchmark=False):
    df = spark.read.parquet(HDFS_INPUT)
    # registro questo DataFrame come tabella SQL temporanea chiamata flights
    df.createOrReplaceTempView("flights")

    #NB i NULL in dep_delay vengono ignorati per natura di spark e la divisione per 100 serve a rimappare sulle fasce orarie
    percentiles = spark.sql("""
        SELECT
            OP_UNIQUE_CARRIER,
            FLOOR(CRS_DEP_TIME / 100) AS hour,

            percentile_approx(DEP_DELAY, 0.25) AS p25,
            percentile_approx(DEP_DELAY, 0.50) AS p50,
            percentile_approx(DEP_DELAY, 0.75) AS p75,
            percentile_approx(DEP_DELAY, 0.90) AS p90

        FROM flights

        WHERE OP_UNIQUE_CARRIER IN ('AA', 'DL', 'UA', 'WN')
          AND CANCELLED = 0

        GROUP BY
            OP_UNIQUE_CARRIER,
            FLOOR(CRS_DEP_TIME / 100)

        ORDER BY
            OP_UNIQUE_CARRIER,
            FLOOR(CRS_DEP_TIME / 100)
    """)

    delay_range = spark.sql("""
        SELECT
            OP_UNIQUE_CARRIER,

            MIN(DEP_DELAY) AS min_delay,
            MAX(DEP_DELAY) AS max_delay

        FROM flights

        WHERE OP_UNIQUE_CARRIER IN ('AA', 'DL', 'UA', 'WN')
          AND CANCELLED = 0

        GROUP BY OP_UNIQUE_CARRIER

        ORDER BY OP_UNIQUE_CARRIER
    """)

    #caching per evitarne il ricalcolo (attivo anche in benchmark)
    percentiles.cache()
    delay_range.cache()

    t0 = time.time()
    percentile_rows = percentiles.collect()
    range_rows = delay_range.collect()
    elapsed = time.time() - t0

    if benchmark:
        return elapsed

    show_dataframe_result(
        result=percentiles,
        query_name="Q3 SQL — Percentili DEP_DELAY per compagnia e fascia oraria",
        elapsed=elapsed,
        n=100,
    )

    show_dataframe_result(
        result=delay_range,
        query_name="Q3 SQL — Min/Max DEP_DELAY per compagnia",
        elapsed=elapsed,
        n=20,
    )

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

    #Ricordiamo che le query sono separate per più motivi, hanno granularità diverse a livello di order by ed inotre la query dei percentili restituisce 4 (carrier) x 24 (ore del giorno) righe
    #Mentre per il calcolo dei min e max abbiamo bisogno solo di 1 riga per cisascun carrier ovvero 4 righe
    save_dataframe_hdfs(percentiles, HDFS_OUTPUT_PERCENTILES)
    save_dataframe_hdfs(delay_range, HDFS_OUTPUT_RANGE)

    print(f"\nTempo Q3 SQL: {elapsed:.2f}s")
    print(f"{'=' * 70}\n")

    return elapsed


def main():
    spark = build_spark_session(
        app_name="Q3_SQL_Percentiles",
        master=SPARK_MASTER,
    )

    run(spark, benchmark=False)

    spark.stop()


if __name__ == "__main__":
    main()
