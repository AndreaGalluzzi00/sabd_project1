import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))

from pyspark.sql.functions import (
    col, avg, count, desc,
    coalesce, lit,
    round as spark_round,
)

from utils_output import (
    show_dataframe_result,
    save_csv_local,
    save_dataframe_hdfs,
)
from utils import build_spark_session


SPARK_MASTER = os.getenv("SPARK_MASTER","spark://spark-master:7077")
HDFS_INPUT = "hdfs://namenode:9000/sabd/processed/"
HDFS_OUTPUT = "hdfs://namenode:9000/sabd/results/q2_df/"
LOCAL_OUT = "/opt/spark/jobs/results/q2_df.csv"


os.makedirs(os.path.dirname(LOCAL_OUT), exist_ok=True)

spark = build_spark_session(
    app_name="Q2_top10_carriers_arr_delay",
    master=SPARK_MASTER,
)

# Filtro base: voli completati (non cancellati, non deviati)
df = (
    spark.read.parquet(HDFS_INPUT)
    .filter((col("CANCELLED") == 0) & (col("DIVERTED") == 0))
)

# Calcolo Q2
t0 = time.time()

#  Calcoliamo il numero di voli, il ritardo medio all'arrivo e i ritardi medi per ciascuna categoria di ritardo per ogni compagnia aerea
#  Coalesce converte in valore oppure null in caso null viene sostituito con literal 0.0, in modo da non influenzare la media con valori nulli
result = (
    df.groupBy("OP_UNIQUE_CARRIER")
    .agg(
        count("*").alias("num_flights"),
        spark_round(avg("ARR_DELAY"), 4).alias("avg_arr_delay"),
        spark_round(
            avg(coalesce(col("CARRIER_DELAY"),      lit(0.0))), 4
        ).alias("avg_carrier_delay"),
        spark_round(
            avg(coalesce(col("WEATHER_DELAY"),      lit(0.0))), 4
        ).alias("avg_weather_delay"),
        spark_round(
            avg(coalesce(col("NAS_DELAY"),          lit(0.0))), 4
        ).alias("avg_nas_delay"),
        spark_round(
            avg(coalesce(col("SECURITY_DELAY"),     lit(0.0))), 4
        ).alias("avg_security_delay"),
        spark_round(
            avg(coalesce(col("LATE_AIRCRAFT_DELAY"),lit(0.0))), 4
        ).alias("avg_late_aircraft_delay"),
    )
    .filter(col("num_flights") >= 500)
    .orderBy(desc("avg_arr_delay"))
    .limit(10)
)

#cachato per evitare di ripetere il calcolo più volte durante la fase di output e salvataggio
result.cache()
rows = result.collect()
elapsed = time.time() - t0

# Output a schermo
show_dataframe_result(result, "Q2", elapsed, 20)

# Salvataggio CSV locale
save_csv_local(LOCAL_OUT, result, rows)

# Salvataggio su HDFS
save_dataframe_hdfs(result, HDFS_OUTPUT)

print(f"\nTempo Q2 (DataFrame API): {elapsed:.2f}s")
print(f"{'='*70}\n")


spark.stop()