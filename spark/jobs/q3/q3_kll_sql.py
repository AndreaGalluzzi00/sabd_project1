"""
Q3 — Percentili DEP_DELAY per compagnia e fascia oraria
Implementazione con KLL sketch via Spark SQL.

Spark SQL non ha un aggregato KLL nativo: registriamo una UDF Python
(kll_percentiles) che riceve l'array dei valori di un gruppo — prodotto
dall'aggregato builtin collect_list — e restituisce [p25, p50, p75, p90].
La query SQL raggruppa per (compagnia, fascia oraria), costruisce lo sketch
nella UDF ed estrae i 4 percentili indicizzando l'array risultante (p[0]..p[3]).

Stesso trade-off della versione DataFrame (q3_kll_df.py): collect_list
materializza i valori grezzi per gruppo (niente merge map-side degli sketch).

Parametro k (accuratezza): default 200 → errore sul rango ≈ 1.5/k ≈ 0.75%.

Modalità:
  - Dev locale:  SPARK_MASTER=local[2]  (default)
  - Cluster:     SPARK_MASTER=spark://spark-master:7077
"""
import csv
import os
import time

from datasketches import kll_floats_sketch
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, DoubleType

SPARK_MASTER            = os.getenv("SPARK_MASTER", "spark://spark-master:7077")
HDFS_INPUT              = "hdfs://namenode:9000/sabd/processed/"
HDFS_OUTPUT_PERCENTILES = "hdfs://namenode:9000/sabd/results/q3_kll_sql/percentiles/"
HDFS_OUTPUT_RANGE       = "hdfs://namenode:9000/sabd/results/q3_kll_sql/delay_range/"
LOCAL_OUT_PERCENTILES   = "/opt/spark/jobs/results/q3_percentiles_kll_sql.csv"
LOCAL_OUT_RANGE         = "/opt/spark/jobs/results/q3_delay_range_kll_sql.csv"
LOCAL_OUT               = "/opt/spark/jobs/results"

K = 200  # parametro di accuratezza KLL

os.makedirs(LOCAL_OUT, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("Q3_KLL_SQL_Percentiles")
    .master(SPARK_MASTER)
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print(f"Master: {SPARK_MASTER}")


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


# Registrazione della UDF → utilizzabile direttamente nelle query SQL.
spark.udf.register("kll_percentiles", kll_percentiles, ArrayType(DoubleType()))

df = spark.read.parquet(HDFS_INPUT)
df.createOrReplaceTempView("flights")

t0 = time.time()

# Sottoquery: collect_list per gruppo → KLL → array di percentili.
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
            kll_percentiles(collect_list(DEP_DELAY)) AS p
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

# ── Output ────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Q3 KLL (Spark SQL, k={K}) — RISULTATI  (tempo esecuzione: {elapsed:.2f}s)")
print(f"{'='*70}")
print("\n=== Percentili per compagnia e fascia oraria ===")
percentiles.show(100, truncate=False)
print("\n=== Min/Max DEP_DELAY per compagnia ===")
delay_range.show(truncate=False)

# ── CSV locale ────────────────────────────────────────────────────────────────
with open(LOCAL_OUT_PERCENTILES, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(percentiles.columns)
    for row in percentile_rows:
        writer.writerow([row[c] for c in percentiles.columns])

with open(LOCAL_OUT_RANGE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(delay_range.columns)
    for row in range_rows:
        writer.writerow([row[c] for c in delay_range.columns])

print(f"\nCSV locale percentili: {LOCAL_OUT_PERCENTILES}")
print(f"CSV locale min/max:    {LOCAL_OUT_RANGE}")

# ── HDFS ──────────────────────────────────────────────────────────────────────
percentiles.coalesce(1).write.mode("overwrite").option("header", True).csv(HDFS_OUTPUT_PERCENTILES)
delay_range.coalesce(1).write.mode("overwrite").option("header", True).csv(HDFS_OUTPUT_RANGE)

print(f"CSV HDFS percentili:   {HDFS_OUTPUT_PERCENTILES}")
print(f"CSV HDFS min/max:      {HDFS_OUTPUT_RANGE}")
print(f"\nTempo Q3 KLL Spark SQL (k={K}): {elapsed:.2f}s")
print(f"{'='*70}\n")

spark.stop()
