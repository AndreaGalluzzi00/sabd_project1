"""
Q3 — Percentili DEP_DELAY per compagnia e fascia oraria
Implementazione con t-digest via Spark SQL.

Spark SQL non ha un aggregato t-digest nativo: registriamo una UDF Python
(tdigest_percentiles) che riceve l'array dei valori di un gruppo — prodotto
dall'aggregato builtin collect_list — e restituisce [p25, p50, p75, p90].
La query SQL raggruppa per (compagnia, fascia oraria), costruisce il digest
nella UDF ed estrae i 4 percentili indicizzando l'array risultante (p[0]..p[3]).

Stesso trade-off della versione DataFrame (q3_tdigest_df.py): collect_list
materializza i valori grezzi per gruppo (niente merge map-side dei digest).

t-digest (Dunning, 2019): errore relativo adattivo, minore sulle code
(p90/p95/p99) rispetto al centro (p50) — utile per i ritardi estremi.

Modalità:
  - Dev locale:  SPARK_MASTER=local[2]  (default)
  - Cluster:     SPARK_MASTER=spark://spark-master:7077
"""
import sys, os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils_output import (
    show_dataframe_result,
    save_csv_local,
    save_dataframe_hdfs,
)

from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, DoubleType
from tdigest import TDigest


SPARK_MASTER            = os.getenv("SPARK_MASTER", "spark://spark-master:7077")
HDFS_INPUT              = "hdfs://namenode:9000/sabd/processed/"
HDFS_OUTPUT_PERCENTILES = "hdfs://namenode:9000/sabd/results/q3_tdigest_sql/percentiles/"
HDFS_OUTPUT_RANGE       = "hdfs://namenode:9000/sabd/results/q3_tdigest_sql/delay_range/"
LOCAL_OUT_PERCENTILES   = "/opt/spark/jobs/results/q3_percentiles_tdigest_sql.csv"
LOCAL_OUT_RANGE         = "/opt/spark/jobs/results/q3_delay_range_tdigest_sql.csv"
LOCAL_OUT               = "/opt/spark/jobs/results"

os.makedirs(LOCAL_OUT, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("Q3_TDigest_SQL_Percentiles")
    .master(SPARK_MASTER)
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print(f"Master: {SPARK_MASTER}")


def tdigest_percentiles(values):
    """Costruisce un t-digest sui valori del gruppo → [p25, p50, p75, p90]."""
    if not values:
        return None
    digest = TDigest()
    for v in values:
        digest.update(float(v))
    return [
        round(float(digest.percentile(25)), 2),
        round(float(digest.percentile(50)), 2),
        round(float(digest.percentile(75)), 2),
        round(float(digest.percentile(90)), 2),
    ]


# Registrazione della UDF → utilizzabile direttamente nelle query SQL.
spark.udf.register("tdigest_percentiles", tdigest_percentiles, ArrayType(DoubleType()))

df = spark.read.parquet(HDFS_INPUT)
df.createOrReplaceTempView("flights")

t0 = time.time()

# Sottoquery: collect_list per gruppo → t-digest → array di percentili.
# Query esterna: estrae i 4 percentili indicizzando l'array (p[i] è 0-based).
percentiles = spark.sql("""
    SELECT
        OP_UNIQUE_CARRIER,
        hour,
        p[0] AS p25,
        p[1] AS p50,
        p[2] AS p75,
        p[3] AS p90
    FROM (
        SELECT
            OP_UNIQUE_CARRIER,
            FLOOR(CRS_DEP_TIME / 100) AS hour,
            tdigest_percentiles(collect_list(DEP_DELAY)) AS p
        FROM flights
        WHERE OP_UNIQUE_CARRIER IN ('AA', 'DL', 'UA', 'WN')
          AND CANCELLED = 0
        GROUP BY OP_UNIQUE_CARRIER, FLOOR(CRS_DEP_TIME / 100)
    )
    ORDER BY OP_UNIQUE_CARRIER, hour
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

percentiles.cache()
delay_range.cache()

percentile_rows = percentiles.collect()
range_rows = delay_range.collect()
elapsed = time.time() - t0


show_dataframe_result(
    result=percentiles,
    query_name="Q3 t-digest Spark SQL — Percentili DEP_DELAY per compagnia e fascia oraria",
    elapsed=elapsed,
    n=100,
)

show_dataframe_result(
    result=delay_range,
    query_name="Q3 t-digest Spark SQL — Min/Max DEP_DELAY per compagnia",
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

save_dataframe_hdfs(percentiles, HDFS_OUTPUT_PERCENTILES)
save_dataframe_hdfs(delay_range, HDFS_OUTPUT_RANGE)

print(f"\nTempo Q3 t-digest Spark SQL: {elapsed:.2f}s")
print(f"{'=' * 70}\n")

spark.stop()