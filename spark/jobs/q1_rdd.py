"""
Q1 (RDD) — AA e DL: statistiche mensili DEP_DELAY e cancellation rate (gen-apr 2025)

Versione implementata con le API RDD (low-level) di Spark, da confrontare con
le versioni DataFrame (q1.py) e Spark SQL (q1_sql.py).

Metriche:
  - cancellation_rate: su TUTTI i voli del mese
  - avg/min/max DEP_DELAY: solo su voli NON cancellati (CANCELLED = 0)

Pipeline RDD (tutta lazy, una sola action finale):
  read parquet → .rdd → filter(AA/DL) → map(to_pair) → reduceByKey(combine)
  → mapValues(finalize) → collect()  ← UNICA action

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

# Intestazione del CSV: stesso ordine delle versioni DataFrame/SQL → confrontabile.
HEADER = [
    "OP_UNIQUE_CARRIER", "YEAR", "MONTH",
    "total_flights", "cancelled_flights", "cancellation_rate_pct",
    "avg_dep_delay", "min_dep_delay", "max_dep_delay",
]

os.makedirs(os.path.dirname(LOCAL_OUT), exist_ok=True)

# La SparkSession contiene lo SparkContext, punto d'ingresso per gli RDD.
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
# Funzioni della pipeline RDD
# ─────────────────────────────────────────────────────────────────────────────

def to_pair(r):
    """Row → (chiave, aggregato_parziale di UNA riga).

    CHIAVE  = (OP_UNIQUE_CARRIER, YEAR, MONTH)   ← il GROUP BY di Q1
    VALORE  = (total, cancelled, delay_sum, delay_n, delay_min, delay_max)

    Il valore ha già la forma dell'aggregato finale (combiner pattern), così
    reduceByKey può combinare due valori con la stessa funzione.
    """
    key = (r["OP_UNIQUE_CARRIER"], r["YEAR"], r["MONTH"])

    # Cancellazione: conta su TUTTI i voli. CANCELLED è double (0.0/1.0), può essere null.
    cancelled = 1 if (r["CANCELLED"] is not None and r["CANCELLED"] == 1.0) else 0

    dep = r["DEP_DELAY"]
    if cancelled == 0 and dep is not None:
        # Volo valido per le statistiche: contribuisce con il suo ritardo (1 osservazione).
        dep = float(dep)
        delay_sum, delay_n, delay_min, delay_max = dep, 1, dep, dep
    else:
        # Volo cancellato o senza ritardo: contributo NEUTRO. +inf/-inf = elementi
        # neutri di min/max, non sporcano gli estremi nello step di reduce.
        delay_sum, delay_n, delay_min, delay_max = 0.0, 0, float("inf"), float("-inf")

    return key, (1, cancelled, delay_sum, delay_n, delay_min, delay_max)


def combine(a, b):
    """Fonde due aggregati parziali della stessa chiave (associativo + commutativo)."""
    return (
        a[0] + b[0],        # total_flights
        a[1] + b[1],        # cancelled_flights
        a[2] + b[2],        # delay_sum
        a[3] + b[3],        # delay_n
        min(a[4], b[4]),    # delay_min
        max(a[5], b[5]),    # delay_max
    )


def finalize(v):
    """Aggregato grezzo → metriche finali (operazioni non associative: media e %)."""
    total, cancelled, delay_sum, delay_n, delay_min, delay_max = v

    cancellation_rate = round(cancelled / total * 100, 4)

    if delay_n > 0:
        avg_delay = round(delay_sum / delay_n, 4)
        min_delay, max_delay = delay_min, delay_max
    else:
        # Nessun volo valido: evita /0 e gli +inf/-inf neutri.
        avg_delay = min_delay = max_delay = None

    return total, cancelled, cancellation_rate, avg_delay, min_delay, max_delay


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline lazy — nessun calcolo eseguito finché non scatta l'action finale
# ─────────────────────────────────────────────────────────────────────────────
#
# Leggiamo solo le 5 colonne utili (column pruning del Parquet), poi passiamo
# all'RDD. filter/map/reduceByKey/mapValues sono tutte TRASFORMAZIONI lazy:
# qui Spark costruisce solo il piano di esecuzione (lineage).
result_rdd = (
    spark.read.parquet(HDFS_INPUT)
    .select("OP_UNIQUE_CARRIER", "YEAR", "MONTH", "CANCELLED", "DEP_DELAY")
    .rdd
    .filter(lambda r: r["OP_UNIQUE_CARRIER"] in ("AA", "DL"))  # solo AA/DL
    .map(to_pair)                                              # → (chiave, parziale)
    .reduceByKey(combine)                                      # SHUFFLE + map-side combine
    .mapValues(finalize)                                       # divisioni finali (no shuffle)
)

# cache(): chiediamo a Spark di TENERE IN MEMORIA il risultato (8 righe) dopo
# la prima action. Avremo DUE action sotto (collect per il CSV locale,
# saveAsTextFile per HDFS): senza cache, la seconda rieseguirebbe da capo
# lettura + shuffle. Con cache, la seconda riusa i dati già materializzati.
result_rdd.cache()

# ─────────────────────────────────────────────────────────────────────────────
# Action #1: collect() → CSV locale + misura del tempo di calcolo
# ─────────────────────────────────────────────────────────────────────────────
# È questa la prima e unica esecuzione "pesante" della pipeline (read + shuffle).
# Sicuro perché il risultato è minuscolo (2 vettori × 4 mesi = 8 righe).
# L'ordinamento lo facciamo in Python sul driver: su 8 righe non vale la pena
# di un sortBy distribuito (che farebbe partire un job extra di sampling).
t0 = time.time()
collected = result_rdd.collect()
elapsed = time.time() - t0

# (chiave, valore) → riga piatta, ordinata per (carrier, year, month).
rows = [
    [carrier, year, month, *metrics]
    for (carrier, year, month), metrics in sorted(collected)
]

# ─────────────────────────────────────────────────────────────────────────────
# Output a schermo + salvataggio CSV locale (dir montata → visibile sull'host)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Q1 (RDD) — RISULTATI  (tempo esecuzione: {elapsed:.2f}s)")
print(f"{'='*70}")
print("  ".join(HEADER))
for row in rows:
    print("  ".join(str(x) for x in row))

with open(LOCAL_OUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(HEADER)
    writer.writerows(rows)
print(f"\nCSV locale: {LOCAL_OUT}")

# ─────────────────────────────────────────────────────────────────────────────
# Action #2: salvataggio su HDFS (riusa il risultato in cache → niente ricalcolo)
# ─────────────────────────────────────────────────────────────────────────────
sc = spark.sparkContext

# saveAsTextFile NON sovrascrive: cancelliamo prima l'eventuale output esistente
# usando le API Hadoop FileSystem (equivalente di mode("overwrite") del writer DF).
hadoop_conf = sc._jsc.hadoopConfiguration()
fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
out_path = sc._jvm.org.apache.hadoop.fs.Path(HDFS_OUTPUT)
if fs.exists(out_path):
    fs.delete(out_path, True)  # True = ricorsivo (è una directory)

# Una riga CSV per gruppo, ordinata per chiave (sortByKey opera sui dati in cache).
# Anteponiamo l'header come prima riga via union, poi coalesce(1) per ottenere
# un UNICO file part-00000 nella directory di output (come coalesce(1) del DF writer).
header_rdd = sc.parallelize([",".join(HEADER)])
data_rdd = (
    result_rdd
    .sortByKey()
    .map(lambda kv: ",".join(str(x) for x in [kv[0][0], kv[0][1], kv[0][2], *kv[1]]))
)
header_rdd.union(data_rdd).coalesce(1).saveAsTextFile(HDFS_OUTPUT)
print(f"CSV su HDFS: {HDFS_OUTPUT}")

print(f"\nTempo Q1 (RDD API): {elapsed:.2f}s")
print(f"{'='*70}\n")

spark.stop()
