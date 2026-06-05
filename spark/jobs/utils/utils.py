from pyspark.sql import SparkSession


def build_spark_session(app_name,master):
    default_fs = "hdfs://namenode:9000"
    log_level = "WARN"
    ui_enabled = False
    ui_port = None
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.hadoop.fs.defaultFS", default_fs)
        .config("spark.ui.enabled", "true" if ui_enabled else "false")
    )
    if ui_port is not None:
        builder = builder.config("spark.ui.port", str(ui_port))

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(log_level)
    print(f"Master: {master}")
    return spark
