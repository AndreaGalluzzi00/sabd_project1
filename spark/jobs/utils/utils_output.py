import csv
import os
import time




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



def save_rdd_csv_hdfs(sc, path, header, data_rdd, row_mapper=None, coalesce_one=True):
    hadoop_conf = sc._jsc.hadoopConfiguration()
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
    out_path = sc._jvm.org.apache.hadoop.fs.Path(path)

    if fs.exists(out_path):
        fs.delete(out_path, True)

    header_rdd = sc.parallelize([",".join(header)])

    if row_mapper is not None:
        rows_rdd = data_rdd.map(row_mapper)
    else:
        rows_rdd = data_rdd

    data_csv_rdd = rows_rdd.map(
        lambda row: ",".join("" if x is None else str(x) for x in row)
    )

    output_rdd = header_rdd.union(data_csv_rdd)

    if coalesce_one:
        output_rdd = output_rdd.coalesce(1)

    t_write = time.time()
    output_rdd.saveAsTextFile(path)
    write_elapsed = time.time() - t_write

    print(f"CSV su HDFS: {path}")
    print(f"Tempo saveAsTextFile: {write_elapsed:.2f}s")

# Il tempo di salvataggio HDFS misura esclusivamente la durata della action
# saveAsTextFile; la cancellazione dell'output precedente non viene inclusa
# nella metrica di scrittura.