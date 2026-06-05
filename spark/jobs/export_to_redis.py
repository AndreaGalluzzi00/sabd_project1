from __future__ import annotations

import os
import sys

import redis

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "utils")))
from utils import build_spark_session  # noqa: E402

REDIS_HOST   = os.getenv("REDIS_HOST", "redis")
REDIS_PORT   = int(os.getenv("REDIS_PORT", 6379))
SPARK_MASTER = os.getenv("SPARK_MASTER", "spark://spark-master:7077")

# Directory dei risultati RDD su HDFS (prodotte dai job q*_rdd / clustering_base).
HDFS_RESULTS    = "hdfs://namenode:9000/sabd/results"
Q1_PATH         = f"{HDFS_RESULTS}/q1_rdd"
Q2_PATH         = f"{HDFS_RESULTS}/q2_rdd"
Q3_PCT_PATH     = f"{HDFS_RESULTS}/q3_rdd/percentiles"
CLUSTERING_PATH = f"{HDFS_RESULTS}/clustering_base"

# Schema dei CSV RDD su HDFS (gli output saveAsTextFile non hanno un header
#  passiamo esplicitamente le colonne).
COLS_Q1     = ["OP_UNIQUE_CARRIER", "YEAR", "MONTH", "total_flights","cancelled_flights", "cancellation_rate_pct","avg_dep_delay", "min_dep_delay", "max_dep_delay"]

COLS_Q2     = ["OP_UNIQUE_CARRIER", "num_flights", "avg_arr_delay","avg_carrier_delay", "avg_weather_delay", "avg_nas_delay","avg_security_delay", "avg_late_aircraft_delay"]

COLS_Q3_PCT = ["OP_UNIQUE_CARRIER", "hour", "p25", "p50", "p75", "p90"]

MONTH_LABELS = {"1": "Jan", "2": "Feb", "3": "Mar", "4": "Apr"}


def connect() -> redis.Redis:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r.ping()
    print(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    return r


#HDFS readers
def hdfs_exists(spark, path: str) -> bool:
    #accesso alla jvm su cui si trova spark
    jvm   = spark._jvm
    #recupera configurazione attuale di hadoop
    hconf = spark._jsc.hadoopConfiguration()
    #accede e verifica che esista
    fs    = jvm.org.apache.hadoop.fs.FileSystem.get(jvm.java.net.URI(path), hconf)
    return fs.exists(jvm.org.apache.hadoop.fs.Path(path))


def read_csv(spark, path: str, columns: list[str] | None = None) -> list[dict]:
    if not hdfs_exists(spark, path):
        print(f"  [SKIP] {path} non trovato su HDFS")
        return []

    #struttura colonne non necessaria solo per la lettura del clustering
    if columns is None:
        df = spark.read.csv(path, header=True)
        return [
            {k: ("" if v is None else v) for k, v in row.asDict().items()}
            for row in df.collect()
        ]
    #i dati sono divisi in più file quindi usa l'header definito precedentemente
    df = spark.read.csv(path, header=False)
    rows: list[dict] = []
    for row in df.collect():
        rec = {c: ("" if v is None else v) for c, v in zip(columns, row)}
        if rec.get(columns[0]) == columns[0]:   # riga header finita tra i dati quindi la saltiamo
            continue
        rows.append(rec)
    return rows


# Q1

def export_q1(r: redis.Redis, rows: list[dict]) -> int:
    written = 0
    for row in rows:
        carrier = row["OP_UNIQUE_CARRIER"]
        year    = row["YEAR"]
        month   = row["MONTH"]
        key     = f"q1:{carrier}:{year}:{month}"
        r.hset(key, mapping={
            "total_flights":          row["total_flights"],
            "cancelled_flights":      row["cancelled_flights"],
            "cancellation_rate_pct":  row["cancellation_rate_pct"],
            "avg_dep_delay":          row["avg_dep_delay"],
            "min_dep_delay":          row["min_dep_delay"],
            "max_dep_delay":          row["max_dep_delay"],
        })
        #risultato finale: q1:AA:2025:1 → {total_flights:..., cancelled_flights:..., ...}
        written += 1
    return written


# Q2

def export_q2(r: redis.Redis, rows: list[dict]) -> int:
    written = 0
    for row in rows:
        carrier = row["OP_UNIQUE_CARRIER"]
        avg_arr = float(row["avg_arr_delay"])

        # ZSET for ranking (score = avg_arr_delay)
        r.zadd("q2:ranking", {carrier: avg_arr})
        #sorted set per il ranking

        r.hset(f"q2:{carrier}", mapping={
            "num_flights":              row["num_flights"],
            "avg_arr_delay":            row["avg_arr_delay"],
            "avg_carrier_delay":        row["avg_carrier_delay"],
            "avg_weather_delay":        row["avg_weather_delay"],
            "avg_nas_delay":            row["avg_nas_delay"],
            "avg_security_delay":       row["avg_security_delay"],
            "avg_late_aircraft_delay":  row["avg_late_aircraft_delay"],
        })
        #risultato finale: q2:AA → {num_flights:..., avg_arr_delay:..., ...}
        written += 1
    return written


# Q3

def export_q3(r: redis.Redis, pct_rows: list[dict]) -> int:
    pct_written = 0
    for row in pct_rows:
        carrier = row["OP_UNIQUE_CARRIER"]
        hour    = row["hour"]
        key     = f"q3:percentiles:{carrier}:{hour}"
        r.hset(key, mapping={
            "p25": row["p25"],
            "p50": row["p50"],
            "p75": row["p75"],
            "p90": row["p90"],
        })
        pct_written += 1
    return pct_written


def export_grafana_viz(r: redis.Redis, q1_rows: list[dict], q2_rows: list[dict],
                       q3_pct_rows: list[dict]) -> None:

    # Q2
    if q1_rows:
        for row in sorted(q1_rows, key=lambda x: int(x["MONTH"])):
            carrier = row["OP_UNIQUE_CARRIER"]
            month = MONTH_LABELS.get(row["MONTH"], row["MONTH"])
            r.hset(f"q1:viz:{carrier}:avg_dep_delay", month, row["avg_dep_delay"])
            r.hset(f"q1:viz:{carrier}:cancellation_rate", month, row["cancellation_rate_pct"])
        print("  Q1 viz: q1:viz:{carrier}:avg_dep_delay  |  q1:viz:{carrier}:cancellation_rate")

        # Q2
    components = [
        "avg_carrier_delay", "avg_weather_delay", "avg_nas_delay",
        "avg_security_delay", "avg_late_aircraft_delay",
    ]
    if q2_rows:
        for row in q2_rows:
            carrier = row["OP_UNIQUE_CARRIER"]
            # avg_arr_delay non ha viz dedicata: il ranking è già nello ZSET q2:ranking.
            for comp in components:
                r.hset(f"q2:viz:{comp}", carrier, row[comp])
        print("  Q2 viz: q2:viz:avg_*_delay  (avg_arr_delay → ZSET canonico q2:ranking)")

    # Q3
    if q3_pct_rows:
        for row in q3_pct_rows:
            carrier = row["OP_UNIQUE_CARRIER"]
            hour    = f"{int(row['hour']):02d}"
            for pct in ["p25", "p50", "p75", "p90"]:
                r.hset(f"q3:viz:{carrier}:{pct}", hour, row[pct])
        print("  Q3 viz: q3:viz:{carrier}:{p25|p50|p75|p90}  (field = ora 00–23)")

CLUSTERING_VIZ_METRICS = [
    "avg_dep_delay", "avg_arr_delay", "cancellation_rate",
    "avg_carrier_delay", "avg_late_aircraft",
]



# Main

def main():
    spark = build_spark_session(app_name="export_to_redis", master=SPARK_MASTER)
    try:
        r = connect()

        print("Lettura risultati RDD da HDFS…")
        q1_rows     = read_csv(spark, Q1_PATH, COLS_Q1)
        q2_rows     = read_csv(spark, Q2_PATH, COLS_Q2)
        q3_pct_rows = read_csv(spark, Q3_PCT_PATH, COLS_Q3_PCT)
        clust_rows  = read_csv(spark, CLUSTERING_PATH)   # header DataFrame affidabile

        n = export_q1(r, q1_rows)
        print(f"Q1: {n} keys written  (pattern: q1:{{carrier}}:{{year}}:{{month}})")

        n = export_q2(r, q2_rows)
        print(f"Q2: {n} carriers written  (q2:ranking ZSET + q2:{{carrier}} HASHes)")

        pct = export_q3(r, q3_pct_rows)
        print(f"Q3: {pct} percentile keys written")

        print("\n── Grafana viz keys ")
        export_grafana_viz(r, q1_rows, q2_rows, q3_pct_rows)



        # Spot-check: print a few keys
        print("\n── Spot-check ")
        sample = r.hgetall("q1:AA:2025:1")
        print(f"q1:AA:2025:1  →  {sample}")

        ranking = r.zrevrangebyscore("q2:ranking", "+inf", "-inf", withscores=True)
        print(f"q2:ranking (top-3)  →  {ranking[:3]}")

        sample_pct = r.hgetall("q3:percentiles:AA:8")
        print(f"q3:percentiles:AA:8  →  {sample_pct}")

        viz_q3 = r.hgetall("q3:viz:AA:p50")
        print(f"q3:viz:AA:p50 (first 5 hours)  →  { {k: viz_q3[k] for k in sorted(viz_q3)[:5]} }")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
