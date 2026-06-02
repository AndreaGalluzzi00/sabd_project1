"""
utils.py — helper condivisi per la creazione della SparkSession.

build_spark_session() centralizza la configurazione comune a tutti i job
(Q1/Q2/Q3 + clustering): defaultFS HDFS, log level e abilitazione/porta della
Spark UI. Ogni job passa solo i parametri che lo differenziano: l'appName e,
per le versioni RDD, l'abilitazione della UI sulla porta 4040.
"""
from pyspark.sql import SparkSession


def build_spark_session(
    app_name,
    master,
    *,
    default_fs="hdfs://namenode:9000",
    ui_enabled=False,
    ui_port=None,
    log_level="WARN",
):
    """Crea (o riusa) la SparkSession con la configurazione passata.

    Parametri:
      app_name    nome dell'applicazione Spark (.appName)
      master      URL del master (.master), es. SPARK_MASTER
      default_fs  filesystem di default Hadoop (spark.hadoop.fs.defaultFS)
      ui_enabled  abilita la Spark UI (spark.ui.enabled): False per DF/SQL,
                  True per le versioni RDD
      ui_port     porta della Spark UI (spark.ui.port); impostata solo se
                  diversa da None (le versioni RDD usano "4040")
      log_level   livello di log dello SparkContext (setLogLevel)
    """
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
