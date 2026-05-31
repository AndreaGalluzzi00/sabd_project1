"""
Q1 — AA e DL: statistiche mensili DEP_DELAY e cancellation rate (gen-apr 2025)

Metriche:
  - cancellation_rate: su TUTTI i voli del mese
  - avg/min/max DEP_DELAY: solo su voli NON cancellati (CANCELLED = 0)

Modalità:
  - Dev locale (Mac M1):  SPARK_MASTER=local[2]  (default)
  - Cluster / EC2:        SPARK_MASTER=spark://spark-master:7077
"""
import csv
import os
import time
from pyspark.sql import SparkSession

SPARK_MASTER = os.getenv("SPARK_MASTER", "spark://spark-master:7077")
HDFS_INPUT   = "hdfs://namenode:9000/sabd/processed/"
HDFS_OUTPUT  = "hdfs://namenode:9000/sabd/results/q1_rdd/"
LOCAL_OUT    = "/opt/spark/jobs/results/q1_rdd.csv"

os.makedirs(os.path.dirname(LOCAL_OUT), exist_ok=True)

# La SparkSession è il punto d'ingresso "alto livello" (DataFrame/SQL).
# Dentro contiene sempre uno SparkContext, che è il punto d'ingresso
# "basso livello" da cui si creano e manipolano gli RDD.
spark = (
    SparkSession.builder
    .appName("Q1_RDD_AA_DL_monthly_stats")
    .master(SPARK_MASTER)
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print(f"Master: {SPARK_MASTER}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Lettura Parquet → RDD + filtro sui vettori AA / DL
# ─────────────────────────────────────────────────────────────────────────────
#
# Perché NON usiamo sc.textFile():
#   Il Parquet è un formato COLONNARE BINARIO (con schema, compressione, ecc.).
#   textFile() legge file di testo riga-per-riga e restituirebbe byte illeggibili.
#   Quindi leggiamo con il reader Parquet del DataFrame e poi passiamo all'RDD.
#
# Cos'è un RDD (Resilient Distributed Dataset):
#   - una collezione di oggetti IMMUTABILE, partizionata sul cluster
#     (ogni partizione viene elaborata in parallelo da un task);
#   - "resiliente" perché Spark ne ricorda la genealogia (lineage) e sa
#     ricostruire una partizione persa rieseguendo le trasformazioni.
#
# df.rdd → converte il DataFrame in un RDD di oggetti Row.
#   Una Row si comporta come una tupla/namedtuple: vi si accede per nome
#   (row["DEP_DELAY"]) o per posizione.

# Leggiamo solo le colonne che servono a Q1: meno dati = meno I/O e meno
# memoria. Essendo Parquet colonnare, la select sfrutta il "column pruning"
# (Spark legge da disco solo queste 5 colonne, non l'intera riga).
df = spark.read.parquet(HDFS_INPUT).select(
    "OP_UNIQUE_CARRIER", "YEAR", "MONTH", "CANCELLED", "DEP_DELAY"
)

# Passaggio al mondo RDD. Da qui in poi lavoriamo con trasformazioni RDD pure.
rows = df.rdd

# filter() è una TRASFORMAZIONE: NON viene eseguita subito (lazy evaluation).
# Spark si limita a registrare nel piano "filtra le righe AA/DL". Il calcolo
# vero parte solo quando invocheremo un'ACTION (es. collect/count), negli
# step successivi. Teniamo solo i due vettori richiesti da Q1.
flights = rows.filter(
    lambda r: r["OP_UNIQUE_CARRIER"] in ("AA", "DL")
)

# Verifica didattica dello Step 1 (count() è un'ACTION → forza l'esecuzione).
# Negli step finali la rimuoveremo per non scatenare un calcolo extra.
print(f"[Step 1] Voli AA/DL letti: {flights.count():,}")

spark.stop()