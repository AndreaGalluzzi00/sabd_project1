"""
Export query results to Redis.

Reads the CSVs produced by q1/q2/q3 jobs (from /opt/spark/jobs/results/)
and writes them to Redis using appropriate data structures.

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
  pip install redis
  python /opt/spark/jobs/export_to_redis.py

Environment variables:
  REDIS_HOST  (default: redis)
  REDIS_PORT  (default: 6379)
"""

import csv
import os

import redis

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

RESULTS_DIR = "/opt/spark/jobs/results"
Q1_CSV      = os.path.join(RESULTS_DIR, "q1.csv")
Q2_CSV      = os.path.join(RESULTS_DIR, "q2.csv")
Q3_PCT_CSV  = os.path.join(RESULTS_DIR, "q3_percentiles.csv")
Q3_RNG_CSV  = os.path.join(RESULTS_DIR, "q3_delay_range.csv")

MONTH_LABELS = {"1": "Jan", "2": "Feb", "3": "Mar", "4": "Apr"}


def connect() -> redis.Redis:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r.ping()
    print(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    return r


# ── Q1 ────────────────────────────────────────────────────────────────────────

def export_q1(r: redis.Redis) -> int:
    written = 0
    with open(Q1_CSV, newline="") as f:
        for row in csv.DictReader(f):
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

def export_q2(r: redis.Redis) -> int:
    written = 0
    with open(Q2_CSV, newline="") as f:
        for row in csv.DictReader(f):
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

def export_q3(r: redis.Redis) -> tuple[int, int]:
    pct_written = 0
    with open(Q3_PCT_CSV, newline="") as f:
        for row in csv.DictReader(f):
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
    with open(Q3_RNG_CSV, newline="") as f:
        for row in csv.DictReader(f):
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
#   Q2  HASH  q2:viz:avg_arr_delay                   field=carrier  value=float
#       HASH  q2:viz:avg_{component}_delay           field=carrier  value=float
#             (components: carrier, weather, nas, security, late_aircraft)
#
#   Q3  HASH  q3:viz:{carrier}:{pct}                field=HH   value=float
#             (pct: p25, p50, p75, p90; HH: "00".."23")
#
#   Q4  HASH  q4:carrier:{carrier}                  fields: cluster_id + features
#       HASH  q4:cluster:{id}                        fields: size + centroid features
#       ZSET  q4:ranking                             score=cluster_id, member=carrier

def export_grafana_viz(r: redis.Redis) -> None:
    """Crea chiavi Redis aggregate per le dashboard Grafana."""

    # ── Q1 ────────────────────────────────────────────────────────────────────
    with open(Q1_CSV, newline="") as f:
        for row in csv.DictReader(f):
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
    with open(Q2_CSV, newline="") as f:
        for row in csv.DictReader(f):
            carrier = row["OP_UNIQUE_CARRIER"]
            r.hset("q2:viz:avg_arr_delay", carrier, row["avg_arr_delay"])
            for comp in components:
                r.hset(f"q2:viz:{comp}", carrier, row[comp])
    print("  Q2 viz: q2:viz:avg_arr_delay  |  q2:viz:avg_*_delay")

    # ── Q3 ────────────────────────────────────────────────────────────────────
    with open(Q3_PCT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            carrier = row["OP_UNIQUE_CARRIER"]
            hour    = f"{int(row['hour']):02d}"           # "00".."23" → ordine lessicografico corretto
            for pct in ["p25", "p50", "p75", "p90"]:
                r.hset(f"q3:viz:{carrier}:{pct}", hour, row[pct])
    print("  Q3 viz: q3:viz:{carrier}:{p25|p50|p75|p90}  (field = ora 00–23)")


# ── Clustering ────────────────────────────────────────────────────────────────
#
# CSV prodotto da clustering.py:
#   OP_UNIQUE_CARRIER, total_flights, avg_dep_delay, avg_arr_delay,
#   cancellation_rate, avg_carrier_delay, avg_weather_delay, avg_nas_delay,
#   avg_security_delay, avg_late_aircraft, prediction
#
# Chiavi Redis:
#   clustering:carrier:{carrier}    HASH  tutte le colonne (dettaglio tabella)
#   clustering:assignments          HASH  {carrier → cluster_id}          (dashboard viz)
#   clustering:viz:avg_dep_delay    HASH  {carrier → valore}              (barchart)
#   clustering:viz:cancellation_rate HASH {carrier → valore}              (barchart)
#   clustering:meta                 HASH  {k, n_carriers}

CLUSTERING_CSV = os.path.join(RESULTS_DIR, "clustering.csv")

CLUSTERING_VIZ_METRICS = [
    "avg_dep_delay", "avg_arr_delay", "cancellation_rate",
    "avg_carrier_delay", "avg_late_aircraft",
]


def export_clustering(r: redis.Redis) -> int:
    if not os.path.exists(CLUSTERING_CSV):
        print("  [SKIP] clustering.csv non trovato — esegui prima clustering.py")
        return 0

    rows = []
    with open(CLUSTERING_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

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
    r = connect()

    n = export_q1(r)
    print(f"Q1: {n} keys written  (pattern: q1:{{carrier}}:{{year}}:{{month}})")

    n = export_q2(r)
    print(f"Q2: {n} carriers written  (q2:ranking ZSET + q2:{{carrier}} HASHes)")

    pct, rng = export_q3(r)
    print(f"Q3: {pct} percentile keys + {rng} range keys written")

    print("\n── Grafana viz keys ────────────────────────────────────────────")
    export_grafana_viz(r)

    n = export_clustering(r)
    if n:
        print(f"Clustering: {n} carrier keys written  (clustering:carrier:{{carrier}} + viz)")

    # ── Spot-check: print a few keys ─────────────────────────────────────────
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


if __name__ == "__main__":
    main()
