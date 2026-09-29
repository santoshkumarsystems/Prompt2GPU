# Author: Santosh Kumar
# Project: Prompt2GPU
# Version: 1.0

"""Measure single-request vLLM latency and output-token throughput.

WHAT THIS PROGRAM DOES
----------------------
1. Confirms the vLLM ``/health`` endpoint returns HTTP 200.
2. Uses the same tokenizer and chat template as the HF/PyTorch baseline.
3. Sends one warm-up request and ten measured requests to the OpenAI-compatible
   ``/v1/completions`` endpoint.
4. Forces exactly 128 output tokens with ``ignore_eos=True``.
5. Measures end-to-end HTTP latency and output tokens/second.
6. Saves every run to CSV and summary statistics to JSON.

The GPU-memory number here comes from ``nvidia-smi`` and includes the vLLM
server's reserved memory pool.  It is not the same metric as PyTorch's
``max_memory_allocated`` in ``hf_baseline.py``; the README/methodology calls
out that distinction explicitly.

See ``docs/PYTHON_WALKTHROUGH.md`` if you are new to Python.
"""


import csv
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone

import requests
import torch
from transformers import AutoTokenizer

from benchmark_config import (
    MODEL,
    PROMPT,
    OUTPUT_TOKENS,
    WARMUP_RUNS,
    MEASURED_RUNS,
    VLLM_URL,
    VLLM_HEALTH_URL,
    PROJECT_ROOT,
)

BASE = str(PROJECT_ROOT)
RESULTS_DIR = f"{BASE}/results"

RAW_CSV = f"{RESULTS_DIR}/vllm_baseline_raw.csv"
SUMMARY_JSON = f"{RESULTS_DIR}/vllm_baseline_summary.json"

os.makedirs(RESULTS_DIR, exist_ok=True)


def build_prompt(tokenizer):

    """Create the exact chat-formatted prompt sent to the vLLM completions endpoint."""
    messages = [
        {"role": "user", "content": PROMPT}
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def gpu_memory_used_mib():

    """Ask ``nvidia-smi`` for current device memory usage and return it as MiB."""
    result = os.popen(
        "nvidia-smi "
        "--query-gpu=memory.used "
        "--format=csv,noheader,nounits"
    ).read().strip()

    return int(result)


def main():

    """Run the controlled single-request vLLM benchmark and save raw plus summary results."""
    print("=== Prompt2GPU : vLLM Baseline ===")

    health = requests.get(
        VLLM_HEALTH_URL,
        timeout=10,
    )

    if health.status_code != 200:
        raise RuntimeError(
            f"vLLM server unhealthy: HTTP {health.status_code}"
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    formatted_prompt = build_prompt(tokenizer)

    input_tokens = len(
        tokenizer(
            formatted_prompt,
            add_special_tokens=False,
        )["input_ids"]
    )

    print("Model:", MODEL)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Input tokens:", input_tokens)
    print("Required output tokens:", OUTPUT_TOKENS)

    payload = {
        "model": MODEL,
        "prompt": formatted_prompt,
        "temperature": 0,

        # Exact same output-work requirement as HF.
        "max_tokens": OUTPUT_TOKENS,

        # vLLM extension:
        # do not terminate early on EOS.
        "ignore_eos": True,
    }

    rows = []

    total_runs = WARMUP_RUNS + MEASURED_RUNS

    for run_index in range(total_runs):

        is_warmup = run_index < WARMUP_RUNS

        gpu_mem_before = gpu_memory_used_mib()

        start = time.perf_counter()

        response = requests.post(
            VLLM_URL,
            json=payload,
            timeout=120,
        )

        end = time.perf_counter()

        response.raise_for_status()

        result = response.json()

        output_tokens = result["usage"]["completion_tokens"]

        if output_tokens != OUTPUT_TOKENS:
            raise RuntimeError(
                f"Benchmark invalid: expected {OUTPUT_TOKENS} "
                f"output tokens but received {output_tokens}"
            )

        latency_sec = end - start
        output_tokens_per_sec = (
            output_tokens / latency_sec
        )

        gpu_mem_after = gpu_memory_used_mib()

        row = {
            "run": run_index + 1,
            "warmup": is_warmup,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_sec": latency_sec,
            "output_tokens_per_sec": output_tokens_per_sec,
            "gpu_memory_before_mib": gpu_mem_before,
            "gpu_memory_after_mib": gpu_mem_after,
        }

        rows.append(row)

        print(
            f"Run {run_index + 1:02d} | "
            f"{'WARMUP' if is_warmup else 'MEASURED'} | "
            f"{latency_sec:.4f}s | "
            f"{output_tokens_per_sec:.2f} tok/s | "
            f"GPU={gpu_mem_after} MiB"
        )

    fieldnames = list(rows[0].keys())

    with open(RAW_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    measured = [
        row for row in rows
        if not row["warmup"]
    ]

    latencies = [
        row["latency_sec"]
        for row in measured
    ]

    throughputs = [
        row["output_tokens_per_sec"]
        for row in measured
    ]

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "engine": "vLLM",
        "model": MODEL,
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "input_tokens": input_tokens,
        "output_tokens_per_request": OUTPUT_TOKENS,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,

        "latency_sec": {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "stdev": statistics.stdev(latencies)
                if len(latencies) > 1 else 0.0,
        },

        "output_tokens_per_sec": {
            "mean": statistics.mean(throughputs),
            "median": statistics.median(throughputs),
            "min": min(throughputs),
            "max": max(throughputs),
            "stdev": statistics.stdev(throughputs)
                if len(throughputs) > 1 else 0.0,
        },
    }

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== FINAL vLLM BASELINE ===")

    print(
        f"Mean latency:   "
        f"{summary['latency_sec']['mean']:.4f} sec"
    )

    print(
        f"Median latency: "
        f"{summary['latency_sec']['median']:.4f} sec"
    )

    print(
        f"Mean throughput: "
        f"{summary['output_tokens_per_sec']['mean']:.2f} tok/s"
    )

    print(
        f"Median throughput: "
        f"{summary['output_tokens_per_sec']['median']:.2f} tok/s"
    )

    print("\nRaw data:", RAW_CSV)
    print("Summary:", SUMMARY_JSON)


# BEGINNER NOTE: Python sets ``__name__`` to ``"__main__"`` when this
# file is run directly.  When another file imports it, this block is skipped.
# That lets us reuse functions without accidentally starting the whole program.
if __name__ == "__main__":
    main()
