#!/usr/bin/env bash
set -e

echo "Sync local dev environment"
uv sync --extra bench

echo "Build and install rust core..."
maturin develop --release

echo "Run benchmarks..."
uv run python benchmarks/bench_throughput.py --data benchmarks/data --results benchmarks/results --prefetch-factor 1 --buffer-size 1000000