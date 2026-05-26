import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots_output")

os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_q1():
    df = pd.read_csv(f"{RESULTS_DIR}/q1.csv")

    for metric, filename, ylabel, title in [
        ("avg_dep_delay", "q1_avg_dep_delay.png", "Avg DEP_DELAY (min)", "Q1 — Avg DEP_DELAY mensile"),
        ("cancellation_rate_pct", "q1_cancellation_rate.png", "Cancellation Rate (%)", "Q1 — Cancellation Rate mensile"),
    ]:
        plt.figure(figsize=(8, 5))

        for carrier in ["AA", "DL"]:
            subset = df[df["OP_UNIQUE_CARRIER"] == carrier].sort_values("MONTH")
            plt.plot(subset["MONTH"], subset[metric], marker="o", label=carrier)

        plt.xlabel("Month")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(sorted(df["MONTH"].unique()))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/{filename}", dpi=150)
        plt.close()


def plot_q2():
    df = pd.read_csv(f"{RESULTS_DIR}/q2.csv")

    carriers = df["OP_UNIQUE_CARRIER"].tolist()
    arr_delays = df["avg_arr_delay"].tolist()

    plt.figure(figsize=(9, 5))
    plt.barh(carriers[::-1], arr_delays[::-1])
    plt.xlabel("Avg ARR_DELAY (min)")
    plt.title("Q2 — Top-10 compagnie per ritardo medio in arrivo")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/q2_ranking.png", dpi=150)
    plt.close()

    components = [
        "avg_carrier_delay",
        "avg_weather_delay",
        "avg_nas_delay",
        "avg_security_delay",
        "avg_late_aircraft_delay",
    ]

    x = np.arange(len(carriers))
    w = 0.15

    plt.figure(figsize=(13, 5))

    for i, c in enumerate(components):
        plt.bar(x + i * w, df[c], width=w, label=c)

    plt.xticks(x + w * 2, carriers)
    plt.ylabel("Avg Delay Component (min)")
    plt.title("Q2 — Incidenza media delle cause di ritardo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/q2_delay_components.png", dpi=150)
    plt.close()


def plot_q3():
    df = pd.read_csv(f"{RESULTS_DIR}/q3_percentiles.csv")

    carriers = sorted(df["OP_UNIQUE_CARRIER"].unique())

    for carrier in carriers:
        subset = df[df["OP_UNIQUE_CARRIER"] == carrier].sort_values("hour")

        plt.figure(figsize=(9, 5))
        for pct in ["p25", "p50", "p75", "p90"]:
            plt.plot(subset["hour"], subset[pct], marker="o", label=pct.upper())

        plt.axhline(0, linewidth=0.8, linestyle=":")
        plt.xlabel("Hour")
        plt.ylabel("DEP_DELAY (min)")
        plt.title(f"Q3 — Percentili DEP_DELAY per fascia oraria ({carrier})")
        plt.xticks(range(0, 24, 2))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/q3_percentiles_{carrier}.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    plot_q1()
    plot_q2()
    plot_q3()
    print(f"Grafici generati in: {PLOTS_DIR}")