# Author: Santosh Kumar
# Project: Prompt2GPU
# Version: 1.0

"""Measure how vLLM behaves as simultaneous request count increases.

EXPERIMENT DESIGN
-----------------
For each concurrency level (1, 2, 4, 8, 16, 32, 64), this program:
* runs one warm-up batch;
* runs five measured batches;
* launches exactly ``concurrency`` HTTP requests at the same time;
* records every request latency;
* calculates total batch wall time and aggregate output tokens/second;
* summarizes mean/median throughput and P95 latency.

``ThreadPoolExecutor`` supplies the concurrent HTTP clients.  The threads are
not doing GPU work themselves; each thread simply waits for one network
request while the vLLM server schedules GPU work.

Raw request rows, per-batch rows, and JSON summaries are all saved.  No chart
or README result is manually typed from this script's console output.

See ``docs/PYTHON_WALKTHROUGH.md`` for a beginner tour of lists, dictionaries,
loops, functions, threads, CSV, JSON, and percentiles.
"""


import csv
import json
import math
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests
from transformers import AutoTokenizer

from benchmark_config import MODEL, PROMPT, OUTPUT_TOKENS, VLLM_URL, PROJECT_ROOT


BASE = str(PROJECT_ROOT)
RESULTS_DIR = f"{BASE}/results"

RAW_REQUEST_CSV = f"{RESULTS_DIR}/vllm_concurrency_raw.csv"
BATCH_CSV = f"{RESULTS_DIR}/vllm_concurrency_batches.csv"
SUMMARY_JSON = f"{RESULTS_DIR}/vllm_concurrency_summary.json"

CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32, 64]

WARMUP_BATCHES = 1
MEASURED_BATCHES = 5

REQUEST_TIMEOUT_SEC = 180

os.makedirs(RESULTS_DIR, exist_ok=True)


def percentile(values, percentile_value):
    """
    Linear-interpolated percentile using only Python standard library.
    Example: percentile(values, 95)
    """
    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile_value / 100.0

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower

    return (
        ordered[lower]
        + (ordered[upper] - ordered[lower]) * fraction
    )


def build_prompt():
    """Load the tokenizer, apply the Qwen chat template, and return the formatted prompt plus token count."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    messages = [
        {
            "role": "user",
            "content": PROMPT,
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    input_tokens = len(
        tokenizer(
            formatted_prompt,
            add_special_tokens=False,
        )["input_ids"]
    )

    return formatted_prompt, input_tokens


def send_request(request_id, formatted_prompt):
    """Send one HTTP inference request, validate the token count, and return that request's timing data."""
    payload = {
        "model": MODEL,
        "prompt": formatted_prompt,
        "temperature": 0,
        "max_tokens": OUTPUT_TOKENS,

        # Force identical generated work for every request.
        "ignore_eos": True,
    }

    start_time = time.perf_counter()

    response = requests.post(
        VLLM_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT_SEC,
    )

    end_time = time.perf_counter()

    response.raise_for_status()

    data = response.json()

    output_tokens = data["usage"]["completion_tokens"]

    if output_tokens != OUTPUT_TOKENS:
        raise RuntimeError(
            f"Request {request_id}: expected "
            f"{OUTPUT_TOKENS} output tokens, "
            f"got {output_tokens}"
        )

    latency_sec = end_time - start_time

    return {
        "request_id": request_id,
        "latency_sec": latency_sec,
        "output_tokens": output_tokens,
        "request_tokens_per_sec":
            output_tokens / latency_sec,
    }


def run_batch(
    concurrency,
    batch_number,
    phase,
    formatted_prompt,
):
    """Launch one group of simultaneous requests and summarize batch wall time, throughput, and latencies."""
    batch_start = time.perf_counter()

    request_results = []

    with ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:

        futures = [
            executor.submit(
                send_request,
                request_id,
                formatted_prompt,
            )
            for request_id in range(concurrency)
        ]

        for future in as_completed(futures):
            request_results.append(
                future.result()
            )

    batch_end = time.perf_counter()

    batch_wall_time_sec = (
        batch_end - batch_start
    )

    total_output_tokens = sum(
        result["output_tokens"]
        for result in request_results
    )

    aggregate_tokens_per_sec = (
        total_output_tokens
        / batch_wall_time_sec
    )

    request_latencies = [
        result["latency_sec"]
        for result in request_results
    ]

    batch_result = {
        "phase": phase,
        "concurrency": concurrency,
        "batch_number": batch_number,
        "request_count": concurrency,
        "total_output_tokens":
            total_output_tokens,
        "batch_wall_time_sec":
            batch_wall_time_sec,
        "aggregate_tokens_per_sec":
            aggregate_tokens_per_sec,
        "mean_request_latency_sec":
            statistics.mean(request_latencies),
        "median_request_latency_sec":
            statistics.median(request_latencies),
        "min_request_latency_sec":
            min(request_latencies),
        "max_request_latency_sec":
            max(request_latencies),
    }

    return request_results, batch_result


def main():

    """Sweep all configured concurrency levels, save raw/batch CSV files, and write the JSON summary."""
    print(
        "=== Prompt2GPU : "
        "vLLM Concurrency Benchmark ==="
    )

    formatted_prompt, input_tokens = (
        build_prompt()
    )

    print("Model:", MODEL)
    print("Input tokens:", input_tokens)
    print(
        "Output tokens/request:",
        OUTPUT_TOKENS,
    )
    print(
        "Concurrency levels:",
        CONCURRENCY_LEVELS,
    )
    print(
        "Warmup batches/level:",
        WARMUP_BATCHES,
    )
    print(
        "Measured batches/level:",
        MEASURED_BATCHES,
    )

    raw_request_rows = []
    measured_batch_rows = []
    final_summaries = []

    for concurrency in CONCURRENCY_LEVELS:

        print(
            f"\n================================"
        )
        print(
            f"Concurrency = {concurrency}"
        )
        print(
            f"================================"
        )

        #
        # Warmup
        #
        for warmup_index in range(
            1,
            WARMUP_BATCHES + 1
        ):
            _, warmup_result = run_batch(
                concurrency=concurrency,
                batch_number=warmup_index,
                phase="warmup",
                formatted_prompt=formatted_prompt,
            )

            print(
                f"WARMUP {warmup_index} | "
                f"{warmup_result['aggregate_tokens_per_sec']:.2f} "
                f"tok/s | "
                f"wall="
                f"{warmup_result['batch_wall_time_sec']:.3f}s"
            )

        #
        # Measured batches
        #
        concurrency_batch_results = []
        concurrency_request_latencies = []

        for measured_index in range(
            1,
            MEASURED_BATCHES + 1
        ):

            request_results, batch_result = (
                run_batch(
                    concurrency=concurrency,
                    batch_number=measured_index,
                    phase="measured",
                    formatted_prompt=formatted_prompt,
                )
            )

            concurrency_batch_results.append(
                batch_result
            )

            measured_batch_rows.append(
                batch_result
            )

            for request_result in request_results:

                latency = (
                    request_result[
                        "latency_sec"
                    ]
                )

                concurrency_request_latencies.append(
                    latency
                )

                raw_request_rows.append(
                    {
                        "concurrency":
                            concurrency,
                        "batch_number":
                            measured_index,
                        "request_id":
                            request_result[
                                "request_id"
                            ],
                        "latency_sec":
                            latency,
                        "output_tokens":
                            request_result[
                                "output_tokens"
                            ],
                        "request_tokens_per_sec":
                            request_result[
                                "request_tokens_per_sec"
                            ],
                        "batch_wall_time_sec":
                            batch_result[
                                "batch_wall_time_sec"
                            ],
                        "aggregate_tokens_per_sec":
                            batch_result[
                                "aggregate_tokens_per_sec"
                            ],
                    }
                )

            print(
                f"RUN {measured_index:02d} | "
                f"{batch_result['aggregate_tokens_per_sec']:.2f} "
                f"tok/s | "
                f"mean latency="
                f"{batch_result['mean_request_latency_sec']:.3f}s | "
                f"wall="
                f"{batch_result['batch_wall_time_sec']:.3f}s"
            )

        aggregate_throughputs = [
            batch[
                "aggregate_tokens_per_sec"
            ]
            for batch in concurrency_batch_results
        ]

        batch_wall_times = [
            batch[
                "batch_wall_time_sec"
            ]
            for batch in concurrency_batch_results
        ]

        total_measured_tokens = (
            concurrency
            * OUTPUT_TOKENS
            * MEASURED_BATCHES
        )

        total_measured_wall_time = sum(
            batch_wall_times
        )

        pooled_throughput = (
            total_measured_tokens
            / total_measured_wall_time
        )

        summary = {
            "concurrency": concurrency,

            "warmup_batches":
                WARMUP_BATCHES,

            "measured_batches":
                MEASURED_BATCHES,

            "measured_requests":
                concurrency
                * MEASURED_BATCHES,

            "mean_aggregate_tokens_per_sec":
                statistics.mean(
                    aggregate_throughputs
                ),

            "median_aggregate_tokens_per_sec":
                statistics.median(
                    aggregate_throughputs
                ),

            "min_aggregate_tokens_per_sec":
                min(
                    aggregate_throughputs
                ),

            "max_aggregate_tokens_per_sec":
                max(
                    aggregate_throughputs
                ),

            "stdev_aggregate_tokens_per_sec":
                (
                    statistics.stdev(
                        aggregate_throughputs
                    )
                    if len(
                        aggregate_throughputs
                    ) > 1
                    else 0.0
                ),

            "pooled_aggregate_tokens_per_sec":
                pooled_throughput,

            "mean_request_latency_sec":
                statistics.mean(
                    concurrency_request_latencies
                ),

            "median_request_latency_sec":
                statistics.median(
                    concurrency_request_latencies
                ),

            "p95_request_latency_sec":
                percentile(
                    concurrency_request_latencies,
                    95,
                ),

            "min_request_latency_sec":
                min(
                    concurrency_request_latencies
                ),

            "max_request_latency_sec":
                max(
                    concurrency_request_latencies
                ),
        }

        final_summaries.append(
            summary
        )

        print(
            "\nLEVEL SUMMARY"
        )

        print(
            f"Mean aggregate throughput: "
            f"{summary['mean_aggregate_tokens_per_sec']:.2f} tok/s"
        )

        print(
            f"Median aggregate throughput: "
            f"{summary['median_aggregate_tokens_per_sec']:.2f} tok/s"
        )

        print(
            f"Mean request latency: "
            f"{summary['mean_request_latency_sec']:.3f}s"
        )

        print(
            f"P95 request latency: "
            f"{summary['p95_request_latency_sec']:.3f}s"
        )

    #
    # Save request-level raw measurements
    #
    with open(
        RAW_REQUEST_CSV,
        "w",
        newline="",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "concurrency",
                "batch_number",
                "request_id",
                "latency_sec",
                "output_tokens",
                "request_tokens_per_sec",
                "batch_wall_time_sec",
                "aggregate_tokens_per_sec",
            ],
        )

        writer.writeheader()
        writer.writerows(
            raw_request_rows
        )

    #
    # Save batch-level measurements
    #
    with open(
        BATCH_CSV,
        "w",
        newline="",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "phase",
                "concurrency",
                "batch_number",
                "request_count",
                "total_output_tokens",
                "batch_wall_time_sec",
                "aggregate_tokens_per_sec",
                "mean_request_latency_sec",
                "median_request_latency_sec",
                "min_request_latency_sec",
                "max_request_latency_sec",
            ],
        )

        writer.writeheader()
        writer.writerows(
            measured_batch_rows
        )

    #
    # Save final summary
    #
    summary_document = {

        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "experiment":
            "vLLM concurrency scaling",

        "model":
            MODEL,

        "input_tokens_per_request":
            input_tokens,

        "output_tokens_per_request":
            OUTPUT_TOKENS,

        "concurrency_levels":
            CONCURRENCY_LEVELS,

        "warmup_batches_per_level":
            WARMUP_BATCHES,

        "measured_batches_per_level":
            MEASURED_BATCHES,

        "prefix_caching":
            False,

        "results":
            final_summaries,
    }

    with open(
        SUMMARY_JSON,
        "w",
    ) as json_file:

        json.dump(
            summary_document,
            json_file,
            indent=2,
        )

    print(
        "\n\n"
        "================================"
    )
    print(
        "FINAL CONCURRENCY SUMMARY"
    )
    print(
        "================================"
    )

    for summary in final_summaries:

        print(
            f"C={summary['concurrency']:2d} | "
            f"throughput="
            f"{summary['mean_aggregate_tokens_per_sec']:8.2f} tok/s | "
            f"mean latency="
            f"{summary['mean_request_latency_sec']:6.3f}s | "
            f"p95="
            f"{summary['p95_request_latency_sec']:6.3f}s"
        )

    print(
        "\nRequest-level raw data:",
        RAW_REQUEST_CSV,
    )

    print(
        "Batch-level data:",
        BATCH_CSV,
    )

    print(
        "Summary:",
        SUMMARY_JSON,
    )


# BEGINNER NOTE: Python sets ``__name__`` to ``"__main__"`` when this
# file is run directly.  When another file imports it, this block is skipped.
# That lets us reuse functions without accidentally starting the whole program.
if __name__ == "__main__":
    main()
