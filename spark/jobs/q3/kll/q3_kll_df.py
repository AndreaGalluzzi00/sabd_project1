"""
Q3 — Percentili DEP_DELAY per compagnia e fascia oraria
Implementazione con KLL sketch via DataFrame API.

A differenza della versione RDD (q3_kll_rdd.py), che fonde gli sketch map-side con
combineByKey, qui restiamo nell'API DataFrame: i valori di ogni gruppo vengono
raccolti con collect_list e lo sketch KLL viene costruito dentro una UDF che
restituisce i 4 percentili (p25/p50/p75/p90) come array<double>.

Trade-off: collect_list materializza i valori grezzi per gruppo (niente merge
map-side degli sketch). È il prezzo per usare KLL — libreria Python esterna —
con l'API dichiarativa di Spark, che non ha un aggregato KLL nativo.

Parametro k (accuratezza): default 200 → errore sul rango ≈ 1.5/k ≈ 0.75%.

Modalità:
  - Dev locale (Mac M1):  SPARK_MASTER=local[2]  (default)
  - Cluster / EC2:        SPARK_MASTER=spark://spark-master:7077
"""
import sys, os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils_output import (
    show_dataframe_result,
    save_csv_local,
    save_dataframe_hdfs,
)

from datasketches import kll_floats_sketch
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, floor, collect_list, udf,
    min as spark_min, max as spark_max,
)
from pyspark.sql.types import ArrayType, DoubleType

SPARK_MASTER            = os.getenv("SPARK_MASTER", "spark://spark-master:7077")
HDFS_INPUT              = "hdfs://namenode:9000/sabd/processed/"
HDFS_OUTPUT_PERCENTILES = "hdfs://namenode:9000/sabd/results/q3_kll_df/percentiles/"
HDFS_OUTPUT_RANGE       = "hdfs://namenode:9000/sabd/results/q3_kll_df/delay_range/"
LOCAL_OUT_PERCENTILES   = "/opt/spark/jobs/results/q3_percentiles_kll_df.csv"
LOCAL_OUT_RANGE         = "/opt/spark/jobs/results/q3_delay_range_kll_df.csv"
LOCAL_OUT               = "/opt/spark/jobs/results"

K = 200  # parametro di accuratezza KLL

os.makedirs(LOCAL_OUT, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("Q3_KLL_DataFrame_Percentiles")
    .master(SPARK_MASTER)
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print(f"Master: {SPARK_MASTER}")


@udf(returnType=ArrayType(DoubleType()))
def kll_percentiles(values):
    """Costruisce un KLL sketch sui valori del gruppo → [p25, p50, p75, p90]."""
    if not values:
        return None
    sketch = kll_floats_sketch(K)
    for v in values:
        sketch.update(float(v))
    return [
        round(float(sketch.get_quantile(0.25)), 2),
        round(float(sketch.get_quantile(0.50)), 2),
        round(float(sketch.get_quantile(0.75)), 2),
        round(float(sketch.get_quantile(0.90)), 2),
    ]


df = (
    spark.read.parquet(HDFS_INPUT)
    .filter(
        (col("OP_UNIQUE_CARRIER").isin("AA", "DL", "UA", "WN")) &
        (col("CANCELLED") == 0)
    )
    .withColumn("hour", floor(col("CRS_DEP_TIME") / 100))
)

t0 = time.time()

# collect_list ignora i NULL → niente dropna esplicito necessario.
percentiles = (
    df.groupBy("OP_UNIQUE_CARRIER", "hour")
    .agg(collect_list("DEP_DELAY").alias("vals"))
    .withColumn("p", kll_percentiles(col("vals")))
    .select(
        "OP_UNIQUE_CARRIER", "hour",
        col("p")[0].alias("p25"),
        col("p")[1].alias("p50"),
        col("p")[2].alias("p75"),
        col("p")[3].alias("p90"),
    )
    .orderBy("OP_UNIQUE_CARRIER", "hour")
)

# Min/Max: esatti, aggregazione nativa DataFrame (no approssimazione).
delay_range = (
    df.groupBy("OP_UNIQUE_CARRIER")
    .agg(
        spark_min("DEP_DELAY").alias("min_delay"),
        spark_max("DEP_DELAY").alias("max_delay"),
    )
    .orderBy("OP_UNIQUE_CARRIER")
)

percentiles.cache()
delay_range.cache()

percentile_rows = percentiles.collect()
range_rows = delay_range.collect()
elapsed = time.time() - t0

show_dataframe_result(
    result=percentiles,
    query_name=f"Q3 KLL DataFrame — Percentili DEP_DELAY per compagnia e fascia oraria (k={K})",
    elapsed=elapsed,
    n=100,
)

show_dataframe_result(
    result=delay_range,
    query_name=f"Q3 KLL DataFrame — Min/Max DEP_DELAY per compagnia (k={K})",
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

print(f"\nTempo Q3 KLL DataFrame (k={K}): {elapsed:.2f}s")
print(f"{'=' * 70}\n")

spark.stop()
