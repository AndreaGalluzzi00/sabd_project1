import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, isnan, isnull, countDistinct, min as spark_min, max as spark_max

SPARK_MASTER = os.getenv("SPARK_MASTER", "local[2]")
HDFS_PATH    = "hdfs://namenode:9000/sabd/processed/"

spark = (
    SparkSession.builder
    .appName("explore_parquet")
    .master(SPARK_MASTER)
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

print("\n" + "="*60)
print("LETTURA PARQUET DA HDFS")
print("="*60)

df = spark.read.parquet(HDFS_PATH)

print("\n--- SCHEMA ---")
df.printSchema()

print("\n--- ROW COUNT PER FILE ---")
for month in ["202501", "202502", "202503", "202504"]:
    mm = month[4:6]  # "01".."04" — corrisponde alla partizione MONTH=mm
    path = f"{HDFS_PATH}MONTH={mm}/{month}_T_ONTIME_REPORTING.parquet"
    n = spark.read.parquet(path).count()
    print(f"  {month}: {n:,} righe")

total = df.count()
print(f"\n  TOTALE: {total:,} righe")

print("\n--- SAMPLE (5 righe) ---")
df.show(5, truncate=False)

print("\n--- NULL COUNT PER COLONNA ---")
null_counts = df.select([
    count(col(c).isNull().cast("int")).alias(c)
    for c in df.columns
])
# calcolo manuale null per colonna
from pyspark.sql.functions import sum as spark_sum
null_df = df.select([
    spark_sum(isnull(col(c)).cast("int")).alias(c)
    for c in df.columns
])
null_df.show(truncate=False)

print("\n--- CARRIERS PRESENTI ---")
df.select("OP_UNIQUE_CARRIER") \
  .distinct() \
  .orderBy("OP_UNIQUE_CARRIER") \
  .show(truncate=False)

print("\n--- RANGE TEMPORALE ---")
df.select(
    spark_min("YEAR").alias("anno_min"),
    spark_max("YEAR").alias("anno_max"),
    spark_min("MONTH").alias("mese_min"),
    spark_max("MONTH").alias("mese_max"),
    spark_min("CRS_DEP_TIME").alias("dep_time_min"),
    spark_max("CRS_DEP_TIME").alias("dep_time_max"),
).show()

print("\n--- STATISTICHE DELAY (voli non cancellati) ---")
df.filter(col("CANCELLED") == 0.0) \
  .select("DEP_DELAY", "ARR_DELAY") \
  .describe() \
  .show()

print("="*60)
print("VERIFICA COMPLETATA")
print("="*60 + "\n")

spark._sc._jvm.System.exit(0)
