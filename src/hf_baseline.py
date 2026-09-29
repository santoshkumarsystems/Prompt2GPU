# Author: Santosh Kumar
# Project: Prompt2GPU
# Version: 1.0

"""Run the Hugging Face / PyTorch single-request baseline.

WHAT THIS PROGRAM DOES
----------------------
1. Loads the same Qwen model used by the vLLM experiments.
2. Builds one chat-formatted prompt.
3. Runs one warm-up generation plus ten measured generations.
4. Forces exactly 128 output tokens so each run performs the same amount of
   generation work.
5. Measures latency, output tokens/second, and PyTorch GPU-memory statistics.
6. Writes raw rows to CSV and aggregate statistics to JSON.

WHY CUDA SYNCHRONIZATION APPEARS
--------------------------------
GPU operations are normally asynchronous: Python can continue before the GPU
finishes. ``torch.cuda.synchronize()`` forces Python to wait so our timer
measures completed GPU work rather than merely measuring how quickly work was
queued.

This file is intentionally verbose and explicit so a reader who is new to
Python can follow the benchmark.  See ``docs/PYTHON_WALKTHROUGH.md`` for the
syntax used throughout the project.
"""


import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark_config import (
    MODEL,
    PROMPT,
    OUTPUT_TOKENS,
    WARMUP_RUNS,
    MEASURED_RUNS,
    PROJECT_ROOT,
)

BASE = str(PROJECT_ROOT)
RESULTS_DIR = f"{BASE}/results"

RAW_CSV = f"{RESULTS_DIR}/hf_baseline_raw.csv"
SUMMARY_JSON = f"{RESULTS_DIR}/hf_baseline_summary.json"

os.makedirs(RESULTS_DIR, exist_ok=True)


def gpu_name():
    """Return the human-readable name of CUDA device 0, for example ``Tesla T4``."""
    return torch.cuda.get_device_name(0)


def build_prompt(tokenizer):
    """Convert the plain user prompt into the model's chat-template text."""
    messages = [
        {"role": "user", "content": PROMPT}
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main():

    """Execute the complete HF/PyTorch benchmark and write CSV/JSON result files."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    print("=== Prompt2GPU : HF / PyTorch Baseline ===")
    print("Model:", MODEL)
    print("GPU:", gpu_name())
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    formatted_prompt = build_prompt(tokenizer)

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    ).to("cuda")

    input_tokens = inputs["input_ids"].shape[1]

    print("Input tokens:", input_tokens)
    print("Required output tokens:", OUTPUT_TOKENS)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.float16,
        device_map="cuda",
    )

    rows = []

    total_runs = WARMUP_RUNS + MEASURED_RUNS

    for run_index in range(total_runs):

        is_warmup = run_index < WARMUP_RUNS

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,

                # Force exactly OUTPUT_TOKENS unless there is
                # an unrecoverable model/runtime error.
                min_new_tokens=OUTPUT_TOKENS,
                max_new_tokens=OUTPUT_TOKENS,
            )

        torch.cuda.synchronize()

        end = time.perf_counter()

        generated = outputs[0][input_tokens:]

        output_tokens = generated.shape[0]

        if output_tokens != OUTPUT_TOKENS:
            raise RuntimeError(
                f"Benchmark invalid: expected {OUTPUT_TOKENS} "
                f"output tokens but received {output_tokens}"
            )

        latency_sec = end - start
        output_tokens_per_sec = output_tokens / latency_sec

        peak_allocated_gb = (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
        )

        peak_reserved_gb = (
            torch.cuda.max_memory_reserved() / (1024 ** 3)
        )

        row = {
            "run": run_index + 1,
            "warmup": is_warmup,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_sec": latency_sec,
            "output_tokens_per_sec": output_tokens_per_sec,
            "peak_allocated_gb": peak_allocated_gb,
            "peak_reserved_gb": peak_reserved_gb,
        }

        rows.append(row)

        print(
            f"Run {run_index + 1:02d} | "
            f"{'WARMUP' if is_warmup else 'MEASURED'} | "
            f"{latency_sec:.4f}s | "
            f"{output_tokens_per_sec:.2f} tok/s | "
            f"peak={peak_allocated_gb:.2f} GB"
        )

    fieldnames = list(rows[0].keys())

    with open(RAW_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
        "engine": "Hugging Face / PyTorch",
        "model": MODEL,
        "gpu": gpu_name(),
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

    print("\n=== FINAL HF BASELINE ===")
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
