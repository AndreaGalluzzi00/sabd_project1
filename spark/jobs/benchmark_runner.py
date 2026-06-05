import argparse
import importlib
import os
import statistics
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "utils")))

from utils_output import (
    save_benchmark_csv_local,
    save_benchmark_hdfs,
)
from utils import build_spark_session


RESULTS_DIR = "/opt/results"
HDFS_RESULTS_DIR = "hdfs://namenode:9000/sabd/results"

DEFAULT_RUNS = 20


MODULES = {
    ("q1", "df"): "q1.q1_df",
    ("q1", "rdd"): "q1.q1_rdd",
    ("q1", "sql"): "q1.q1_sql",

    ("q2", "df"): "q2.q2_df",
    ("q2", "rdd"): "q2.q2_rdd",
    ("q2", "sql"): "q2.q2_sql",

    ("q3", "df"): "q3.q3_df",
    ("q3", "rdd"): "q3.q3_rdd",
    ("q3", "sql"): "q3.q3_sql",
    ("q3", "kll_rdd"): "q3.kll.q3_kll_rdd",
    ("q3", "tdigest_rdd"): "q3.tdigest.q3_tdigest_rdd",
}


def run_spark_job(query, api, spark):
    key = (query, api)

    if key not in MODULES:
        raise ValueError(f"Combinazione non valida: {query} {api}")

    module_name = MODULES[key]
    query_module = importlib.import_module(module_name)

    if not hasattr(query_module, "run"):
        raise AttributeError(f"Il modulo {module_name} non contiene una funzione run(spark, benchmark=True).")

    compute_time = query_module.run(spark, benchmark=True)

    if compute_time is None:
        raise RuntimeError(f"La funzione run() di {module_name} non ha restituito un tempo.")

    return float(compute_time)


def summary_stats(values):
    return {
        "min": min(values),
        "max": max(values),
        "avg": statistics.mean(values),
        "median": statistics.median(values),
        "variance": statistics.variance(values) if len(values) > 1 else 0.0,
    }


def build_summary(run_rows):
    summary_rows = []
    groups = {}

    for row in run_rows:
        key = (row["query"], row["api"])
        groups.setdefault(key, []).append(row)

    for (query, api), rows in groups.items():
        values = [float(r["compute_time"]) for r in rows]
        stats = summary_stats(values)

        summary_rows.append({
            "query": query,
            "api": api,
            "min": stats["min"],
            "max": stats["max"],
            "avg": stats["avg"],
            "median": stats["median"],
            "variance": stats["variance"],
        })

    return summary_rows


def run_benchmark(query, api, runs, spark):
    rows = []

    print("\n" + "=" * 70)
    print(f"Benchmark {query.upper()} - {api.upper()}")
    print(f"Modulo: {MODULES[(query, api)]}")
    print(f"Run misurate: {runs}")
    print("=" * 70 + "\n")

    for run_num in range(1, runs + 1):
        compute_time = run_spark_job(query, api, spark)

        row = {
            "query": query,
            "api": api,
            "run": run_num,
            "compute_time": compute_time,
        }

        rows.append(row)

        print(f"Run {run_num}/{runs} - time={compute_time:.4f}s")

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--q", choices=["q1", "q2", "q3"], required=False)
    parser.add_argument(
        "--api",
        choices=["rdd", "df", "sql", "kll_rdd", "tdigest_rdd"],
        required=False,
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--all", action="store_true")

    args = parser.parse_args()

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    if args.all:
        configs = list(MODULES.keys())
    else:
        if not args.q or not args.api:
            print("Errore: usa --q e --api oppure --all")
            sys.exit(1)

        configs = [(args.q, args.api)]

    spark = build_spark_session(
        app_name="Benchmark_Runner",
        master=os.getenv("SPARK_MASTER", "spark://spark-master:7077"),
    )

    for query, api in configs:
        rows = run_benchmark(
            query=query,
            api=api,
            runs=args.runs,
            spark=spark,
        )

        summary_rows = build_summary(rows)
        label = f"{query}_{api}"

        runs_path    = os.path.join(RESULTS_DIR, f"benchmark_runs_{label}.csv")
        summary_path = os.path.join(RESULTS_DIR, f"benchmark_summary_{label}.csv")

        save_benchmark_csv_local(
            runs_path,
            rows,
            ["query", "api", "run", "compute_time"],
        )

        save_benchmark_csv_local(
            summary_path,
            summary_rows,
            ["query", "api", "min", "max", "avg", "median", "variance"],
        )

        save_benchmark_hdfs(
            spark,
            rows,
            f"{HDFS_RESULTS_DIR}/benchmark_runs_{label}",
        )

        save_benchmark_hdfs(
            spark,
            summary_rows,
            f"{HDFS_RESULTS_DIR}/benchmark_summary_{label}",
        )

        print(f"Run salvate in: {runs_path}")
        print(f"Summary salvato in: {summary_path}")

    spark.stop()

    print("\n" + "=" * 70)
    print("Benchmark completato.")
    print(f"Risultati locali in: {RESULTS_DIR}")
    print(f"Risultati HDFS in: {HDFS_RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()