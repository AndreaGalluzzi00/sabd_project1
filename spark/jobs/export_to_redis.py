"""
Export query results to Redis.

Reads the RDD query results **directly from HDFS** (le directory in cui i job
Spark scrivono i risultati) and writes them to Redis for the Grafana dashboards.

Leggere da HDFS — invece dei CSV locali in results/, che ne sono solo una copia
coalesce(1) — è ciò che chiede la specifica: "esportare risultati da HDFS a
storage (Redis)".

Key schema
----------
Q1  HASH  q1:{carrier}:{year}:{month}
          fields: total_flights, cancelled_flights, cancellation_rate_pct,
                  avg_dep_delay, min_dep_delay, max_dep_delay

Q2  ZSET  q2:ranking           (score = avg_arr_delay, member = carrier)
    HASH  q2:{carrier}
          fields: num_flights, avg_arr_delay, avg_carrier_delay,
                  avg_weather_delay, avg_nas_delay, avg_security_delay,
                  avg_late_aircraft_delay

Q3  HASH  q3:percentiles:{carrier}:{hour}
          fields: p25, p50, p75, p90
    HASH  q3:range:{carrier}
          fields: min_delay, max_delay

Usage (inside spark-master container):
  /opt/spark/bin/spark-submit /opt/spark/jobs/export_to_redis.py

Environment variables:
  REDIS_HOST    (default: redis)
  REDIS_PORT    (default: 6379)
  SPARK_MASTER  (default: spark://spark-master:7077)
"""

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
Q3_RNG_PATH     = f"{HDFS_RESULTS}/q3_rdd/delay_range"
CLUSTERING_PATH = f"{HDFS_RESULTS}/clustering_base"

# Schema dei CSV RDD su HDFS (gli output saveAsTextFile non hanno un header
# affidabile su più part-file → lo passiamo esplicito, vedi read_csv).
COLS_Q1     = ["OP_UNIQUE_CARRIER", "YEAR", "MONTH", "total_flights",
               "cancelled_flights", "cancellation_rate_pct",
               "avg_dep_delay", "min_dep_delay", "max_dep_delay"]
COLS_Q2     = ["OP_UNIQUE_CARRIER", "num_flights", "avg_arr_delay",
               "avg_carrier_delay", "avg_weather_delay", "avg_nas_delay",
               "avg_security_delay", "avg_late_aircraft_delay"]
COLS_Q3_PCT = ["OP_UNIQUE_CARRIER", "hour", "p25", "p50", "p75", "p90"]
COLS_Q3_RNG = ["OP_UNIQUE_CARRIER", "min_delay", "max_delay"]

MONTH_LABELS = {"1": "Jan", "2": "Feb", "3": "Mar", "4": "Apr"}


def connect() -> redis.Redis:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r.ping()
    print(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    return r


# ── HDFS readers ────────────────────────────────────────────────────────────--

def hdfs_exists(spark, path: str) -> bool:
    jvm   = spark._jvm
    hconf = spark._jsc.hadoopConfiguration()
    fs    = jvm.org.apache.hadoop.fs.FileSystem.get(jvm.java.net.URI(path), hconf)
    return fs.exists(jvm.org.apache.hadoop.fs.Path(path))


def read_csv(spark, path: str, columns: list[str] | None = None) -> list[dict]:
    """Legge una directory CSV da HDFS → lista di dict {colonna: str}.

    Gli output RDD (saveAsTextFile) scrivono l'header come semplice riga di
    testo: con più part-file Spark non sa quale sia l'header, quindi passiamo
    `columns` esplicite (header=False) e scartiamo le righe-header eventualmente
    finite tra i dati. Gli output DataFrame (clustering) hanno un header
    affidabile → columns=None.

    I valori restano stringhe (come csv.DictReader); i null diventano "".
    Ritorna [] se il path non esiste su HDFS.
    """
    if not hdfs_exists(spark, path):
        print(f"  [SKIP] {path} non trovato su HDFS")
        return []

    if columns is None:
        df = spark.read.csv(path, header=True)
        return [
            {k: ("" if v is None else v) for k, v in row.asDict().items()}
            for row in df.collect()
        ]

    df = spark.read.csv(path, header=False)
    rows: list[dict] = []
    for row in df.collect():
        rec = {c: ("" if v is None else v) for c, v in zip(columns, row)}
        if rec.get(columns[0]) == columns[0]:   # riga header finita tra i dati
            continue
        rows.append(rec)
    return rows


# ── Q1 ────────────────────────────────────────────────────────────────────────

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
        written += 1
    return written


# ── Q2 ────────────────────────────────────────────────────────────────────────

def export_q2(r: redis.Redis, rows: list[dict]) -> int:
    written = 0
    for row in rows:
        carrier = row["OP_UNIQUE_CARRIER"]
        avg_arr = float(row["avg_arr_delay"])

        # ZSET for ranking (score = avg_arr_delay)
        r.zadd("q2:ranking", {carrier: avg_arr})

        # HASH for full metrics
        r.hset(f"q2:{carrier}", mapping={
            "num_flights":              row["num_flights"],
            "avg_arr_delay":            row["avg_arr_delay"],
            "avg_carrier_delay":        row["avg_carrier_delay"],
            "avg_weather_delay":        row["avg_weather_delay"],
            "avg_nas_delay":            row["avg_nas_delay"],
            "avg_security_delay":       row["avg_security_delay"],
            "avg_late_aircraft_delay":  row["avg_late_aircraft_delay"],
        })
        written += 1
    return written


# ── Q3 ────────────────────────────────────────────────────────────────────────

def export_q3(r: redis.Redis, pct_rows: list[dict], rng_rows: list[dict]) -> tuple[int, int]:
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

    rng_written = 0
    for row in rng_rows:
        carrier = row["OP_UNIQUE_CARRIER"]
        r.hset(f"q3:range:{carrier}", mapping={
            "min_delay": row["min_delay"],
            "max_delay": row["max_delay"],
        })
        rng_written += 1

    return pct_written, rng_written


# ── Grafana visualization keys ────────────────────────────────────────────────
#
# Chiavi aggregate pensate per le dashboard Grafana (redis-datasource plugin).
# Usano HGETALL su hash aggregati invece di mille HGET separati.
#
# Schema:
#   Q1  HASH  q1:viz:{carrier}:avg_dep_delay        field=Mon  value=float
#       HASH  q1:viz:{carrier}:cancellation_rate     field=Mon  value=float
#
#   Q2  HASH  q2:viz:avg_{component}_delay           field=carrier  value=float
#             (components: carrier, weather, nas, security, late_aircraft)
#       NB: il ranking per avg_arr_delay NON ha una viz dedicata — il barchart
#           lo legge direttamente dallo ZSET canonico q2:ranking (ZRANGE WITHSCORES,
#           che il redis-datasource restituisce con lo stesso frame di HGETALL).
#
#   Q3  HASH  q3:viz:{carrier}:{pct}                field=HH   value=float
#             (pct: p25, p50, p75, p90; HH: "00".."23")
#
#   Clustering  HASH  clustering:carrier:{carrier}   fields: feature + prediction
#               HASH  clustering:assignments          {carrier → cluster_id}
#               HASH  clustering:viz:{metric}         {carrier → valore}  (5 metriche)
#               HASH  clustering:meta                 {k, n_carriers}

def export_grafana_viz(r: redis.Redis, q1_rows: list[dict], q2_rows: list[dict],
                       q3_pct_rows: list[dict]) -> None:
    """Crea chiavi Redis aggregate per le dashboard Grafana."""

    # ── Q1 ────────────────────────────────────────────────────────────────────
    if q1_rows:
        for row in q1_rows:
            carrier = row["OP_UNIQUE_CARRIER"]
            month   = MONTH_LABELS.get(row["MONTH"], row["MONTH"])
            r.hset(f"q1:viz:{carrier}:avg_dep_delay",     month, row["avg_dep_delay"])
            r.hset(f"q1:viz:{carrier}:cancellation_rate", month, row["cancellation_rate_pct"])
        print("  Q1 viz: q1:viz:{carrier}:avg_dep_delay  |  q1:viz:{carrier}:cancellation_rate")

    # ── Q2 ────────────────────────────────────────────────────────────────────
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

    # ── Q3 ────────────────────────────────────────────────────────────────────
    if q3_pct_rows:
        for row in q3_pct_rows:
            carrier = row["OP_UNIQUE_CARRIER"]
            hour    = f"{int(row['hour']):02d}"           # "00".."23" → ordine lessicografico corretto
            for pct in ["p25", "p50", "p75", "p90"]:
                r.hset(f"q3:viz:{carrier}:{pct}", hour, row[pct])
        print("  Q3 viz: q3:viz:{carrier}:{p25|p50|p75|p90}  (field = ora 00–23)")


# ── Clustering ────────────────────────────────────────────────────────────────
#
# CSV prodotto da clustering_base.py (HDFS: /sabd/results/clustering_base/):
#   OP_UNIQUE_CARRIER, total_flights, <8 feature>, prediction
#
# Chiavi Redis:
#   clustering:carrier:{carrier}    HASH  tutte le colonne (dettaglio tabella)
#   clustering:assignments          HASH  {carrier → cluster_id}          (dashboard viz)
#   clustering:viz:avg_dep_delay    HASH  {carrier → valore}              (barchart)
#   clustering:viz:cancellation_rate HASH {carrier → valore}              (barchart)
#   clustering:meta                 HASH  {k, n_carriers}

CLUSTERING_VIZ_METRICS = [
    "avg_dep_delay", "avg_arr_delay", "cancellation_rate",
    "avg_carrier_delay", "avg_late_aircraft",
]


def export_clustering(r: redis.Redis, rows: list[dict]) -> int:
    if not rows:
        return 0

    assignments: dict[str, str] = {}
    viz: dict[str, dict[str, str]] = {m: {} for m in CLUSTERING_VIZ_METRICS}

    for row in rows:
        carrier    = row["OP_UNIQUE_CARRIER"]
        cluster_id = row["prediction"]
        assignments[carrier] = cluster_id

        r.hset(f"clustering:carrier:{carrier}", mapping={
            k: row[k] for k in row if k != "OP_UNIQUE_CARRIER"
        })

        for metric in CLUSTERING_VIZ_METRICS:
            viz[metric][carrier] = row[metric]

    r.delete("clustering:assignments")
    r.hset("clustering:assignments", mapping=assignments)

    for metric, mapping in viz.items():
        r.delete(f"clustering:viz:{metric}")
        r.hset(f"clustering:viz:{metric}", mapping=mapping)

    k = max(int(v) for v in assignments.values()) + 1
    r.hset("clustering:meta", mapping={"k": k, "n_carriers": len(rows)})

    return len(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    spark = build_spark_session(app_name="export_to_redis", master=SPARK_MASTER)
    try:
        r = connect()

        print("Lettura risultati RDD da HDFS…")
        q1_rows     = read_csv(spark, Q1_PATH, COLS_Q1)
        q2_rows     = read_csv(spark, Q2_PATH, COLS_Q2)
        q3_pct_rows = read_csv(spark, Q3_PCT_PATH, COLS_Q3_PCT)
        q3_rng_rows = read_csv(spark, Q3_RNG_PATH, COLS_Q3_RNG)
        clust_rows  = read_csv(spark, CLUSTERING_PATH)   # header DataFrame affidabile

        n = export_q1(r, q1_rows)
        print(f"Q1: {n} keys written  (pattern: q1:{{carrier}}:{{year}}:{{month}})")

        n = export_q2(r, q2_rows)
        print(f"Q2: {n} carriers written  (q2:ranking ZSET + q2:{{carrier}} HASHes)")

        pct, rng = export_q3(r, q3_pct_rows, q3_rng_rows)
        print(f"Q3: {pct} percentile keys + {rng} range keys written")

        print("\n── Grafana viz keys ────────────────────────────────────────────")
        export_grafana_viz(r, q1_rows, q2_rows, q3_pct_rows)

        n = export_clustering(r, clust_rows)
        if n:
            print(f"Clustering: {n} carrier keys written  (clustering:carrier:{{carrier}} + viz)")

        # ── Spot-check: print a few keys ─────────────────────────────────────
        print("\n── Spot-check ──────────────────────────────────────────────────")
        sample = r.hgetall("q1:AA:2025:1")
        print(f"q1:AA:2025:1  →  {sample}")

        ranking = r.zrevrangebyscore("q2:ranking", "+inf", "-inf", withscores=True)
        print(f"q2:ranking (top-3)  →  {ranking[:3]}")

        sample_pct = r.hgetall("q3:percentiles:AA:8")
        print(f"q3:percentiles:AA:8  →  {sample_pct}")

        sample_rng = r.hgetall("q3:range:AA")
        print(f"q3:range:AA  →  {sample_rng}")

        viz_q1 = r.hgetall("q1:viz:AA:avg_dep_delay")
        print(f"q1:viz:AA:avg_dep_delay  →  {viz_q1}")

        viz_q3 = r.hgetall("q3:viz:AA:p50")
        print(f"q3:viz:AA:p50 (first 5 hours)  →  { {k: viz_q3[k] for k in sorted(viz_q3)[:5]} }")
        print("────────────────────────────────────────────────────────────────")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
