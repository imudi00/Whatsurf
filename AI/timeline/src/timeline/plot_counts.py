import os
import matplotlib.pyplot as plt
from .io_utils import ensure_dir

def plot_daily_counts(counts, burst_dates, out_path):
    ensure_dir(os.path.dirname(out_path))

    x = [str(d) for d in counts.index]
    y = counts.values

    plt.figure(figsize=(12, 4))
    plt.plot(x, y, marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.title("Daily Article Counts")

    burst_set = set(burst_dates or [])
    for i, d in enumerate(x):
        if d in burst_set:
            plt.annotate("burst", (i, y[i]))

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
