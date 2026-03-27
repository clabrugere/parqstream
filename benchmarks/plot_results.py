"""Plot benchmark results from JSON files in results/.

Produces:
  - results/throughput_isolated.png  — rows/sec vs num_workers (parqstream only)
  - results/throughput_comparison.png — parqstream vs torch bar chart
  - results/gpu_utilization.png       — GPU util % bar chart (if available)

Usage:
    python benchmarks/plot_results.py
    python benchmarks/plot_results.py --results results/
"""

import argparse
import json
import os

import matplotlib.pyplot as plt


def load(results_dir: str, name: str) -> dict | None:
    path = os.path.join(results_dir, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def plot_isolated(data: dict, out_dir: str) -> None:
    results = data["results"]

    # Separate sequential vs shuffled
    seq = [r for r in results if not r["shuffle"]]
    shuf = [r for r in results if r["shuffle"]]

    # Workers sweep at bs=4k
    def workers_sweep(rs):
        return [(r["num_workers"], r["rows_per_sec"] / 1e6) for r in rs if r["batch_size"] == 4096]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # num_workers scaling
    ax = axes[0]
    sx = workers_sweep(seq)
    hx = workers_sweep(shuf)
    if sx:
        ws, ts = zip(*sorted(sx))
        ax.plot(ws, ts, "o-", label="sequential")
    if hx:
        ws, ts = zip(*sorted(hx))
        ax.plot(ws, ts, "s--", label="shuffled")
    ax.set_xlabel("num_workers")
    ax.set_ylabel("Throughput (M rows/s)")
    ax.set_title("parqstream: throughput vs num_workers (bs=4096)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # batch_size effect at w=4
    ax = axes[1]

    def bs_sweep(rs):
        return [(r["batch_size"], r["rows_per_sec"] / 1e6) for r in rs if r["num_workers"] == 4]

    sx = bs_sweep(seq)
    hx = bs_sweep(shuf)
    if sx:
        bs, ts = zip(*sorted(sx))
        ax.plot(bs, ts, "o-", label="sequential")
    if hx:
        bs, ts = zip(*sorted(hx))
        ax.plot(bs, ts, "s--", label="shuffled")
    ax.set_xlabel("batch_size")
    ax.set_ylabel("Throughput (M rows/s)")
    ax.set_title("parqstream: throughput vs batch_size (w=4)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "throughput_isolated.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def plot_comparison(data: dict, out_dir: str) -> None:
    import matplotlib.pyplot as plt

    results = data["results"]
    labels, values, colors = [], [], []

    order = [
        ("parqstream", "sequential"),
        ("torch_iterable", "sequential"),
        ("parqstream", "shuffled"),
        ("torch_map", "shuffled"),
    ]
    palette = {
        ("parqstream", "sequential"): "#1f77b4",
        ("torch_iterable", "sequential"): "#aec7e8",
        ("parqstream", "shuffled"): "#ff7f0e",
        ("torch_map", "shuffled"): "#ffbb78",
    }
    display = {
        ("parqstream", "sequential"): "parqstream\n(sequential)",
        ("torch_iterable", "sequential"): "torch iter\n(sequential)",
        ("parqstream", "shuffled"): "parqstream\n(shuffled)",
        ("torch_map", "shuffled"): "torch map\n(shuffled, full RAM)",
    }

    for key in order:
        sys_, mode = key
        match = next((r for r in results if r["system"] == sys_ and r.get("mode") == mode), None)
        if match:
            labels.append(display[key])
            values.append(match["rows_per_sec"] / 1e6)
            colors.append(palette[key])

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, width=0.5)
    ax.bar_label(bars, fmt="%.1f M/s", padding=3, fontsize=9)
    ax.set_ylabel("Throughput (M rows/s)")
    ax.set_title("parqstream vs PyTorch DataLoader")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(out_dir, "throughput_comparison.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def plot_gpu(data: dict, out_dir: str) -> None:
    import matplotlib.pyplot as plt

    results = [r for r in data["results"] if r.get("gpu_available")]
    if not results:
        print("No GPU data available, skipping GPU plot.")
        return

    labels = []
    steps_vals = []
    gpu_vals = []

    for r in results:
        sys_ = r["system"]
        shuffle = r.get("shuffle", False)
        lbl = f"{sys_}\n{'shuffled' if shuffle else 'sequential'}"
        labels.append(lbl)
        steps_vals.append(r["steps_per_sec"])
        gpu_vals.append(r.get("gpu_mean_util", 0))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.bar(labels, steps_vals, color="#1f77b4")
    ax.set_ylabel("Training steps/sec")
    ax.set_title("Steps/sec (GPU training loop)")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(labels, gpu_vals, color="#2ca02c")
    ax.set_ylabel("Mean GPU utilization (%)")
    ax.set_title("GPU utilization")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "gpu_utilization.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/")
    args = parser.parse_args()

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        import sys

        sys.exit("matplotlib not installed: pip install matplotlib")

    isolated = load(args.results, "bench_isolated")
    comparison = load(args.results, "bench_comparison")
    gpu = load(args.results, "bench_gpu")

    if isolated:
        plot_isolated(isolated, args.results)
    else:
        print("bench_isolated.json not found, skipping.")

    if comparison:
        plot_comparison(comparison, args.results)
    else:
        print("bench_comparison.json not found, skipping.")

    if gpu:
        plot_gpu(gpu, args.results)
    else:
        print("bench_gpu.json not found, skipping.")
