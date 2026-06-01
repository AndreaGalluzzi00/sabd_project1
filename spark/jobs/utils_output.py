import csv
import os




def show_dataframe_result(result, query_name, elapsed, n=20):
    print(f"\n{'=' * 70}")
    print(f"{query_name} — RISULTATI  (tempo esecuzione: {elapsed:.2f}s)")
    print(f"{'=' * 70}")
    result.show(n, truncate=False)


def save_csv_local(path, result, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = result.columns
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for row in rows:
            writer.writerow([row[c] for c in cols])
    print(f"CSV locale:  {path}")


def save_dataframe_hdfs(df, hdfs_output):
    df.coalesce(1).write.mode("overwrite").option("header", "true").csv(hdfs_output)
    print(f"CSV su HDFS: {hdfs_output}")


# ─────────────────────────────────────────────────────────────────────────────
# Funzioni per RDD
# ─────────────────────────────────────────────────────────────────────────────

def show_rdd_result(rows, header, query_name, elapsed):
    print(f"\n{'=' * 70}")
    print(f"{query_name} — RISULTATI  (tempo esecuzione: {elapsed:.2f}s)")
    print(f"{'=' * 70}")

    if not rows:
        print("Nessun risultato.")
        return

    table = [header] + [[str(x) for x in row] for row in rows]

    widths = [
        max(len(row[i]) for row in table)
        for i in range(len(header))
    ]

    def format_row(row):
        return "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(header))) + " |"

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print(separator)
    print(format_row(header))
    print(separator)

    for row in rows:
        print(format_row([str(x) for x in row]))
    print(separator)



def save_rdd_csv_local(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"CSV locale:  {path}")
