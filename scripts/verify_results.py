#!/usr/bin/env python3
# Author: Santosh Kumar
# Project: Prompt2GPU
# Version: 1.0

"""Recompute published Prompt2GPU summaries from the raw CSV evidence.

This is the project's zero-ambiguity / zero-manual-number check.  It loads the
raw measurement rows, recomputes means, medians, percentiles, peaks, and queue
statistics, then compares those values with the committed summary JSON files.
If any value disagrees beyond a tiny floating-point tolerance, the script
raises an error instead of printing PASS.

A reader new to Python can treat each ``verify_*`` function as an independent
check.  See ``docs/PYTHON_WALKTHROUGH.md`` for the syntax.
"""

from pathlib import Path
import csv
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TOL = 1e-8


def read_csv(name):
    """Read a CSV file and return its rows as dictionaries keyed by column name."""
    with (RESULTS / name).open(newline="") as f:
        return list(csv.DictReader(f))


def read_json(name):
    """Read a JSON file and convert it into normal Python dictionaries/lists."""
    return json.loads((RESULTS / name).read_text())


def percentile(values, p):
    """Compute a linearly interpolated percentile for result verification."""
    values = sorted(values)
    pos = (len(values) - 1) * p / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] + (values[hi] - values[lo]) * frac


def close(a, b, label):
    """Assert that two floating-point values are equal within the project tolerance."""
    if not math.isclose(float(a), float(b), rel_tol=TOL, abs_tol=TOL):
        raise AssertionError(f"{label}: {a} != {b}")


def verify_baseline(prefix, summary_name):
    """Recompute latency and throughput statistics for one baseline from its raw CSV."""
    rows = read_csv(f"{prefix}_raw.csv")
    measured = [r for r in rows if r["warmup"].lower() == "false"]
    lat = [float(r["latency_sec"]) for r in measured]
    tps = [float(r["output_tokens_per_sec"]) for r in measured]
    summary = read_json(summary_name)
    close(statistics.mean(lat), summary["latency_sec"]["mean"], f"{prefix} mean latency")
    close(statistics.median(lat), summary["latency_sec"]["median"], f"{prefix} median latency")
    close(statistics.mean(tps), summary["output_tokens_per_sec"]["mean"], f"{prefix} mean throughput")
    close(statistics.median(tps), summary["output_tokens_per_sec"]["median"], f"{prefix} median throughput")


def verify_concurrency():
    """Recompute every concurrency summary from the raw request and batch CSV files."""
    batches = read_csv("vllm_concurrency_batches.csv")
    requests = read_csv("vllm_concurrency_raw.csv")
    summary = read_json("vllm_concurrency_summary.json")
    for expected in summary["results"]:
        c = int(expected["concurrency"])
        b = [r for r in batches if int(r["concurrency"]) == c]
        q = [r for r in requests if int(r["concurrency"]) == c]
        throughput = [float(r["aggregate_tokens_per_sec"]) for r in b]
        lat = [float(r["latency_sec"]) for r in q]
        close(statistics.mean(throughput), expected["mean_aggregate_tokens_per_sec"], f"C={c} mean throughput")
        close(statistics.median(throughput), expected["median_aggregate_tokens_per_sec"], f"C={c} median throughput")
        close(statistics.mean(lat), expected["mean_request_latency_sec"], f"C={c} mean latency")
        close(percentile(lat, 95), expected["p95_request_latency_sec"], f"C={c} p95 latency")


def verify_kv():
    """Recompute every context/KV-pressure summary from raw request and batch evidence."""
    batches = read_csv("vllm_kv_pressure_batches.csv")
    requests = read_csv("vllm_kv_pressure_requests.csv")
    summary = read_json("vllm_kv_pressure_summary.json")
    for expected in summary["results"]:
        n = int(expected["actual_input_tokens"])
        b = [r for r in batches if int(r["actual_input_tokens"]) == n]
        q = [r for r in requests if int(r["actual_input_tokens"]) == n]
        throughput = [float(r["aggregate_tokens_per_sec"]) for r in b]
        lat = [float(r["latency_sec"]) for r in q]
        kv = [float(r["peak_kv_cache_usage"]) for r in b]
        waiting = [float(r["peak_requests_waiting"]) for r in b]
        queue = [float(r["average_queue_time_sec"]) for r in b]
        close(statistics.mean(throughput), expected["mean_aggregate_tokens_per_sec"], f"KV n={n} mean throughput")
        close(statistics.mean(lat), expected["mean_request_latency_sec"], f"KV n={n} mean latency")
        close(percentile(lat, 95), expected["p95_request_latency_sec"], f"KV n={n} p95 latency")
        close(max(kv), expected["max_peak_kv_cache_usage"], f"KV n={n} peak usage")
        close(max(waiting), expected["max_requests_waiting"], f"KV n={n} peak waiting")
        close(statistics.mean(queue), expected["mean_queue_time_sec"], f"KV n={n} mean queue")


def main():
    """Run all verification checks and print PASS only when every published summary matches raw data."""
    verify_baseline("hf_baseline", "hf_baseline_summary.json")
    verify_baseline("vllm_baseline", "vllm_baseline_summary.json")
    verify_concurrency()
    verify_kv()
    print("PASS: published JSON summaries recompute from committed raw CSV measurements.")


# BEGINNER NOTE: Python sets ``__name__`` to ``"__main__"`` when this
# file is run directly.  When another file imports it, this block is skipped.
# That lets us reuse functions without accidentally starting the whole program.
if __name__ == "__main__":
    main()
