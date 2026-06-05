import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from utils_output import (
    show_dataframe_result,
    save_csv_local,
    save_dataframe_hdfs,
)
from utils import build_spark_session
SPARK_MASTER = os.getenv("SPARK_MASTER","spark://spark-master:7077")
DEFAULT_HDFS_INPUT = "hdfs://namenode:9000/sabd/processed_single/"
HDFS_OUTPUT  = "hdfs://namenode:9000/sabd/results/q1_sql/"
LOCAL_OUT    = "/opt/results/q1_sql.csv"

os.makedirs(os.path.dirname(LOCAL_OUT), exist_ok=True)


def run(spark, benchmark=False):
    # Lettura e registrazione come vista temporanea
    df = spark.read.parquet(input_path)

    # La vista temporanea "flights" permette di eseguire query SQL su questo DataFrame.
    # Infatti, il motore SQL ragiona in termini di tabelle e viste, non di oggetti Python - come per esempio DataFrame
    # La TempView è il ponte che collega i due mondi: prende il tuo oggetto DataFrame df (che vive nella memoria del programma Python)
    # e lo espone al parser SQL con un nome interrogabile.
    df.createOrReplaceTempView("flights")

    ### Calcolo Q1 con Spark SQL

    # Calcolo tempo iniziale


    result = spark.sql("""
        SELECT
            OP_UNIQUE_CARRIER,
            YEAR,
            MONTH,
            COUNT(*)                                                        AS total_flights,
            SUM(CAST(CANCELLED AS LONG))                                    AS cancelled_flights,
            ROUND(SUM(CANCELLED) / COUNT(*) * 100, 4)                      AS cancellation_rate_pct,
            ROUND(AVG(CASE WHEN CANCELLED = 0 THEN DEP_DELAY END), 4)      AS avg_dep_delay,
            ROUND(MIN(CASE WHEN CANCELLED = 0 THEN DEP_DELAY END), 4)      AS min_dep_delay,
            ROUND(MAX(CASE WHEN CANCELLED = 0 THEN DEP_DELAY END), 4)      AS max_dep_delay
        FROM flights
        WHERE OP_UNIQUE_CARRIER IN ('AA', 'DL')
        GROUP BY OP_UNIQUE_CARRIER, YEAR, MONTH
        ORDER BY OP_UNIQUE_CARRIER, YEAR, MONTH
    """)

    if not benchmark:
        result.cache()
    t0 = time.time()
    # Esecuzione
    rows = result.collect()

    # Calcolo tempo trascorso
    elapsed = time.time() - t0

    if benchmark:
        return elapsed

    # Output a schermo
    show_dataframe_result(result, "Q1_SQL", elapsed, 20)

    # Salvataggio CSV locale
    save_csv_local(LOCAL_OUT, result, rows)

    # Salvataggio su HDFS
    save_dataframe_hdfs(result, HDFS_OUTPUT)

    print(f"\nTempo Q1 (Spark SQL): {elapsed:.2f}s")
    print(f"{'='*70}\n")

    return elapsed


def main():

    spark = build_spark_session(
        app_name="Q1_SQL_AA_DL_monthly_stats",
        master=SPARK_MASTER,
    )

    run(spark, benchmark=False)

    spark.stop()


if __name__ == "__main__":
    main()
