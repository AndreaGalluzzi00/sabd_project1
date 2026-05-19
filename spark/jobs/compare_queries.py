"""
Confronto DataFrame API vs Spark SQL per Q1 e Q2.
Esegue entrambe le versioni, verifica che i risultati siano identici
e stampa il confronto dei tempi di esecuzione.
"""
import os
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, min as spark_min, max as spark_max,
    count, sum as spark_sum, when, round as spark_round,
    coalesce, lit, desc,
)

SPARK_MASTER = os.getenv("SPARK_MASTER", "local[2]")
HDFS_INPUT   = "hdfs://namenode:9000/sabd/processed/"

spark = (
    SparkSession.builder
    .appName("compare_df_vs_sql")
    .master(SPARK_MASTER)
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

df = spark.read.parquet(HDFS_INPUT)
df.cache()
df.createOrReplaceTempView("flights")

# ─────────────────────────────────────────────────────────────────────────────
# Q1
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("Q1 — DataFrame API")
print("="*70)
t0 = time.time()
q1_df = (
    df.filter(col("OP_UNIQUE_CARRIER").isin("AA", "DL"))
    .groupBy("OP_UNIQUE_CARRIER", "YEAR", "MONTH")
    .agg(
        count("*").alias("total_flights"),
        spark_sum(col("CANCELLED").cast("long")).alias("cancelled_flights"),
        spark_round(spark_sum("CANCELLED") / count("*") * 100, 4).alias("cancellation_rate_pct"),
        spark_round(avg(when(col("CANCELLED") == 0, col("DEP_DELAY"))), 4).alias("avg_dep_delay"),
        spark_round(spark_min(when(col("CANCELLED") == 0, col("DEP_DELAY"))), 4).alias("min_dep_delay"),
        spark_round(spark_max(when(col("CANCELLED") == 0, col("DEP_DELAY"))), 4).alias("max_dep_delay"),
    )
    .orderBy("OP_UNIQUE_CARRIER", "YEAR", "MONTH")
)
q1_df.cache()
q1_df.collect()
t1_df = time.time() - t0
q1_df.show(20, truncate=False)

print("\n" + "="*70)
print("Q1 — Spark SQL")
print("="*70)
t0 = time.time()
q1_sql = spark.sql("""
    SELECT
        OP_UNIQUE_CARRIER, YEAR, MONTH,
        COUNT(*)                                                   AS total_flights,
        SUM(CANCELLED)                                             AS cancelled_flights,
        ROUND(SUM(CANCELLED) / COUNT(*) * 100, 4)                 AS cancellation_rate_pct,
        ROUND(AVG(CASE WHEN CANCELLED = 0 THEN DEP_DELAY END), 4) AS avg_dep_delay,
        ROUND(MIN(CASE WHEN CANCELLED = 0 THEN DEP_DELAY END), 4) AS min_dep_delay,
        ROUND(MAX(CASE WHEN CANCELLED = 0 THEN DEP_DELAY END), 4) AS max_dep_delay
    FROM flights
    WHERE OP_UNIQUE_CARRIER IN ('AA', 'DL')
    GROUP BY OP_UNIQUE_CARRIER, YEAR, MONTH
    ORDER BY OP_UNIQUE_CARRIER, YEAR, MONTH
""")
q1_sql.cache()
q1_sql.collect()
t1_sql = time.time() - t0
q1_sql.show(20, truncate=False)

# Confronto risultati Q1
diff_q1 = q1_df.subtract(q1_sql).count() + q1_sql.subtract(q1_df).count()
print(f"\nQ1 — Righe diverse tra DF e SQL: {diff_q1}  {'OK' if diff_q1 == 0 else 'DIFFERENZE TROVATE'}")

# ─────────────────────────────────────────────────────────────────────────────
# Q2
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("Q2 — DataFrame API")
print("="*70)
t0 = time.time()
q2_df = (
    df.filter((col("CANCELLED") == 0) & (col("DIVERTED") == 0))
    .groupBy("OP_UNIQUE_CARRIER")
    .agg(
        count("*").alias("num_flights"),
        spark_round(avg("ARR_DELAY"), 4).alias("avg_arr_delay"),
        spark_round(avg(coalesce(col("CARRIER_DELAY"),       lit(0.0))), 4).alias("avg_carrier_delay"),
        spark_round(avg(coalesce(col("WEATHER_DELAY"),       lit(0.0))), 4).alias("avg_weather_delay"),
        spark_round(avg(coalesce(col("NAS_DELAY"),           lit(0.0))), 4).alias("avg_nas_delay"),
        spark_round(avg(coalesce(col("SECURITY_DELAY"),      lit(0.0))), 4).alias("avg_security_delay"),
        spark_round(avg(coalesce(col("LATE_AIRCRAFT_DELAY"), lit(0.0))), 4).alias("avg_late_aircraft_delay"),
    )
    .filter(col("num_flights") >= 500)
    .orderBy(desc("avg_arr_delay"))
    .limit(10)
)
q2_df.cache()
q2_df.collect()
t2_df = time.time() - t0
q2_df.show(10, truncate=False)

print("\n" + "="*70)
print("Q2 — Spark SQL")
print("="*70)
t0 = time.time()
q2_sql = spark.sql("""
    SELECT
        OP_UNIQUE_CARRIER,
        COUNT(*)                                             AS num_flights,
        ROUND(AVG(ARR_DELAY), 4)                            AS avg_arr_delay,
        ROUND(AVG(COALESCE(CARRIER_DELAY,       0.0)), 4)   AS avg_carrier_delay,
        ROUND(AVG(COALESCE(WEATHER_DELAY,       0.0)), 4)   AS avg_weather_delay,
        ROUND(AVG(COALESCE(NAS_DELAY,           0.0)), 4)   AS avg_nas_delay,
        ROUND(AVG(COALESCE(SECURITY_DELAY,      0.0)), 4)   AS avg_security_delay,
        ROUND(AVG(COALESCE(LATE_AIRCRAFT_DELAY, 0.0)), 4)   AS avg_late_aircraft_delay
    FROM flights
    WHERE CANCELLED = 0 AND DIVERTED = 0
    GROUP BY OP_UNIQUE_CARRIER
    HAVING COUNT(*) >= 500
    ORDER BY avg_arr_delay DESC
    LIMIT 10
""")
q2_sql.cache()
q2_sql.collect()
t2_sql = time.time() - t0
q2_sql.show(10, truncate=False)

# Confronto risultati Q2
diff_q2 = q2_df.subtract(q2_sql).count() + q2_sql.subtract(q2_df).count()
print(f"\nQ2 — Righe diverse tra DF e SQL: {diff_q2}  {'OK' if diff_q2 == 0 else 'DIFFERENZE TROVATE'}")

# ─────────────────────────────────────────────────────────────────────────────
# Riepilogo tempi
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("RIEPILOGO TEMPI DI ESECUZIONE")
print("="*70)
print(f"{'Query':<10} {'DataFrame API':>20} {'Spark SQL':>20} {'Differenza':>20}")
print("-"*70)
print(f"{'Q1':<10} {t1_df:>17.2f}s   {t1_sql:>17.2f}s   {t1_sql - t1_df:>+18.2f}s")
print(f"{'Q2':<10} {t2_df:>17.2f}s   {t2_sql:>17.2f}s   {t2_sql - t2_df:>+18.2f}s")
print("="*70 + "\n")

spark._sc._jvm.System.exit(0)
