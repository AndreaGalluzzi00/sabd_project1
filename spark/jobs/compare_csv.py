import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round as spark_round

SPARK_MASTER = os.getenv("SPARK_MASTER", "local[2]")

spark = (
    SparkSession.builder
    .appName("compare_csv_results_hdfs")
    .master(SPARK_MASTER)
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


def read_csv(path):
    return (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(path)
    )


def normalize(df, numeric_cols=None, decimals=4):

    numeric_cols = numeric_cols or []

    # ordine colonne stabile
    df = df.select(sorted(df.columns))

    # normalizzazione float
    for c in numeric_cols:
        if c in df.columns:
            df = df.withColumn(c, spark_round(col(c), decimals))

    return df


def compare_csv(name, df_path, sql_path, numeric_cols=None):

    print("\n" + "=" * 80)
    print(f"CONFRONTO {name}")
    print("=" * 80)

    df_api = normalize(
        read_csv(df_path),
        numeric_cols
    )

    df_sql = normalize(
        read_csv(sql_path),
        numeric_cols
    )

    print(f"Righe DataFrame API: {df_api.count()}")
    print(f"Righe Spark SQL:     {df_sql.count()}")

    only_df = df_api.exceptAll(df_sql)
    only_sql = df_sql.exceptAll(df_api)

    count_df = only_df.count()
    count_sql = only_sql.count()

    if count_df == 0 and count_sql == 0:

        print(f"{name}: CSV IDENTICI")

    else:

        print(f"{name}: DIFFERENZE TROVATE")

        print(f"Solo DataFrame API: {count_df}")
        print(f"Solo Spark SQL:     {count_sql}")

        print("\n--- SOLO DATAFRAME API ---")
        only_df.show(20, truncate=False)

        print("\n--- SOLO SPARK SQL ---")
        only_sql.show(20, truncate=False)

# ==========================================================
# Q1
# ==========================================================

compare_csv(
    name="Q1",
    df_path="hdfs://namenode:9000/sabd/results/q1/",
    sql_path="hdfs://namenode:9000/sabd/results/q1_sql/",
    numeric_cols=[
        "cancellation_rate_pct",
        "avg_dep_delay",
        "min_dep_delay",
        "max_dep_delay"
    ]
)

# ==========================================================
# Q2
# ==========================================================

compare_csv(
    name="Q2",
    df_path="hdfs://namenode:9000/sabd/results/q2/",
    sql_path="hdfs://namenode:9000/sabd/results/q2_sql/",
    numeric_cols=[
        "avg_arr_delay",
        "avg_carrier_delay",
        "avg_weather_delay",
        "avg_nas_delay",
        "avg_security_delay",
        "avg_late_aircraft_delay"
    ]
)

# ==========================================================
# Q3 Percentiles
# ==========================================================

compare_csv(
    name="Q3 Percentiles",
    df_path="hdfs://namenode:9000/sabd/results/q3/percentiles/",
    sql_path="hdfs://namenode:9000/sabd/results/q3_sql/percentiles/",
    numeric_cols=[
        "p25",
        "p50",
        "p75",
        "p90"
    ]
)


# ==========================================================
# Q3 Delay Range
# ==========================================================

compare_csv(
    name="Q3 Delay Range",
    df_path="hdfs://namenode:9000/sabd/results/q3/delay_range/",
    sql_path="hdfs://namenode:9000/sabd/results/q3_sql/delay_range/",
    numeric_cols=["min_delay", "max_delay"]
)
spark.stop()