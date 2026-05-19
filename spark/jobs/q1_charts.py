"""
Generazione grafici Q1 — da eseguire sull'host con miniforge Python.
Legge spark/jobs/results/q1.csv e produce i grafici in spark/jobs/results/.

  /Users/andreagalluzzi/miniforge3/bin/python3 spark/jobs/q1_charts.py
"""
import csv
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
CSV_PATH    = RESULTS_DIR / "q1.csv"
CHARTS_DIR  = RESULTS_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

MONTH_LABELS = {1: "Gen", 2: "Feb", 3: "Mar", 4: "Apr"}
COLORS = {"AA": "#1f77b4", "DL": "#d62728"}
CARRIERS = ["AA", "DL"]

# ── Lettura CSV ───────────────────────────────────────────────────────────────
data = {c: {} for c in CARRIERS}
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        c = row["OP_UNIQUE_CARRIER"]
        m = int(row["MONTH"])
        data[c][m] = {
            "avg_dep_delay":       float(row["avg_dep_delay"]),
            "min_dep_delay":       float(row["min_dep_delay"]),
            "max_dep_delay":       float(row["max_dep_delay"]),
            "cancellation_rate_pct": float(row["cancellation_rate_pct"]),
            "total_flights":       int(row["total_flights"]),
        }

months = sorted({m for c in data.values() for m in c})
x_labels = [MONTH_LABELS[m] for m in months]

def vals(carrier, metric):
    return [data[carrier][m][metric] for m in months]

# ── Grafico 1: DEP_DELAY medio mensile ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
for c in CARRIERS:
    ax.plot(x_labels, vals(c, "avg_dep_delay"), marker="o",
            color=COLORS[c], label=c, linewidth=2)
ax.set_title("Q1 — DEP_DELAY medio mensile (voli non cancellati)", fontsize=13)
ax.set_xlabel("Mese")
ax.set_ylabel("Ritardo medio partenza (min)")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig(CHARTS_DIR / "q1_avg_dep_delay.png", dpi=150)
plt.close(fig)
print("Salvato: q1_avg_dep_delay.png")

# ── Grafico 2: DEP_DELAY min/max con banda ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
for ax, c in zip(axes, CARRIERS):
    avg_v = vals(c, "avg_dep_delay")
    min_v = vals(c, "min_dep_delay")
    max_v = vals(c, "max_dep_delay")
    ax.fill_between(x_labels, min_v, max_v, alpha=0.15, color=COLORS[c], label="min–max")
    ax.plot(x_labels, avg_v, marker="o", color=COLORS[c], linewidth=2, label="media")
    ax.set_title(f"{c} — DEP_DELAY (min / media / max)")
    ax.set_xlabel("Mese")
    ax.set_ylabel("Minuti")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.suptitle("Q1 — Range mensile DEP_DELAY", fontsize=13)
fig.tight_layout()
fig.savefig(CHARTS_DIR / "q1_dep_delay_range.png", dpi=150)
plt.close(fig)
print("Salvato: q1_dep_delay_range.png")

# ── Grafico 3: Cancellation rate mensile ─────────────────────────────────────
x = range(len(months))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 4))
for i, c in enumerate(CARRIERS):
    offset = (i - 0.5) * width
    ax.bar([xi + offset for xi in x], vals(c, "cancellation_rate_pct"),
           width=width, color=COLORS[c], label=c, alpha=0.85)
ax.set_title("Q1 — Cancellation rate mensile (%)", fontsize=13)
ax.set_xlabel("Mese")
ax.set_ylabel("Cancellation rate (%)")
ax.set_xticks(list(x))
ax.set_xticklabels(x_labels)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig(CHARTS_DIR / "q1_cancellation_rate.png", dpi=150)
plt.close(fig)
print("Salvato: q1_cancellation_rate.png")

print(f"\nTutti i grafici in: {CHARTS_DIR}")
