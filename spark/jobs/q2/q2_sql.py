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
HDFS_OUTPUT  = "hdfs://namenode:9000/sabd/results/q2_sql/"
LOCAL_OUT    = "/opt/results/q2_sql.csv"

os.makedirs(os.path.dirname(LOCAL_OUT), exist_ok=True)


def run(spark, benchmark=False, hdfs_input=None):
    input_path = hdfs_input or os.getenv("HDFS_INPUT", DEFAULT_HDFS_INPUT)
    # Lettura e registrazione come vista temporanea
    df = spark.read.parquet(input_path)
    df.createOrReplaceTempView("flights")

    # Calcolo Q2 con Spark SQL


    # per arr_delay possiamo ignorare il valore null poichè presente solo se sono diverted o cancelled
    # per le componenti di ritardo conteggiamo nella media anche i null (settati a 0)

    # arr_delay non potrà mai essere null a causa del where su cancelled e diverted = 0
    result = spark.sql("""
        SELECT
            OP_UNIQUE_CARRIER,
            COUNT(*)                                            AS num_flights,
            ROUND(AVG(ARR_DELAY), 4)                           AS avg_arr_delay,
            ROUND(AVG(COALESCE(CARRIER_DELAY,      0.0)), 4)   AS avg_carrier_delay,
            ROUND(AVG(COALESCE(WEATHER_DELAY,      0.0)), 4)   AS avg_weather_delay,
            ROUND(AVG(COALESCE(NAS_DELAY,          0.0)), 4)   AS avg_nas_delay,
            ROUND(AVG(COALESCE(SECURITY_DELAY,     0.0)), 4)   AS avg_security_delay,
            ROUND(AVG(COALESCE(LATE_AIRCRAFT_DELAY,0.0)), 4)   AS avg_late_aircraft_delay
        FROM flights
        WHERE CANCELLED = 0
          AND DIVERTED  = 0
        GROUP BY OP_UNIQUE_CARRIER
        HAVING COUNT(*) >= 500
        ORDER BY avg_arr_delay DESC
        LIMIT 10
    """)

    if not benchmark:
        result.cache()
    t0 = time.time()
    rows = result.collect()
    elapsed = time.time() - t0

    if benchmark:
        return elapsed

    # Output a schermo
    show_dataframe_result(result, "Q2_SQL", elapsed, 20)

    # Salvataggio CSV locale
    save_csv_local(LOCAL_OUT, result, rows)

    # Salvataggio su HDFS
    save_dataframe_hdfs(result, HDFS_OUTPUT)

    print(f"\nTempo Q2 (Spark SQL): {elapsed:.2f}s")
    print(f"{'='*70}\n")

    return elapsed


def main():
    spark = build_spark_session(
        app_name="Q2_SQL_top10_carriers_arr_delay",
        master=SPARK_MASTER,
    )

    run(spark, benchmark=False)

    spark.stop()


if __name__ == "__main__":
    main()
