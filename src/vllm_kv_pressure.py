# Author: Santosh Kumar
# Project: Prompt2GPU
# Version: 1.0

"""Measure context length, KV-cache pressure, queueing, and throughput.

EXPERIMENT DESIGN
-----------------
The server stays at concurrency 80 while input context grows from 256 to 3800
tokens.  Generation remains fixed at 128 output tokens.  For every context
size, the program runs one warm-up batch and three measured batches.

A background sampler polls vLLM's Prometheus ``/metrics`` endpoint every
100 ms while requests are active.  This captures peak KV-cache usage, requests
running, requests waiting, and queue-time counters instead of looking only
after the batch has already finished.

This is intentionally described as *context + KV-cache pressure*.  Longer
prompts increase both prefill computation and KV-cache demand, so the data does
not claim that KV usage alone causes the observed slowdown.

The file is long because it preserves raw request data, batch data, and metric
time-series samples separately.  Follow the section headers from top to bottom;
``docs/PYTHON_WALKTHROUGH.md`` explains every Python construct used here.
"""


import csv
import json
import math
import os
import re
import statistics
import threading
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests
from transformers import AutoTokenizer

from benchmark_config import (
    MODEL,
    OUTPUT_TOKENS,
    VLLM_URL,
    VLLM_METRICS_URL,
    PROJECT_ROOT,
)


# ---------------------------------------------------------
# Prompt2GPU paths
# ---------------------------------------------------------

BASE = str(PROJECT_ROOT)
RESULTS_DIR = f"{BASE}/results"

REQUEST_CSV = (
    f"{RESULTS_DIR}/vllm_kv_pressure_requests.csv"
)

BATCH_CSV = (
    f"{RESULTS_DIR}/vllm_kv_pressure_batches.csv"
)

METRICS_CSV = (
    f"{RESULTS_DIR}/vllm_kv_pressure_metrics.csv"
)

SUMMARY_JSON = (
    f"{RESULTS_DIR}/vllm_kv_pressure_summary.json"
)

METRICS_URL = VLLM_METRICS_URL


# ---------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------

CONTEXT_TARGETS = [
    256,
    512,
    1024,
    2048,
    3072,
    3800,
]

CONCURRENCY = 80

WARMUP_BATCHES = 1
MEASURED_BATCHES = 3

METRIC_SAMPLE_INTERVAL_SEC = 0.10
REQUEST_TIMEOUT_SEC = 300


os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)


# ---------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------

def percentile(values, percentile_value):

    """Return a linearly interpolated percentile; Prompt2GPU uses it for P95 latency."""
    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (
        (len(ordered) - 1)
        * percentile_value
        / 100.0
    )

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower

    return (
        ordered[lower]
        + (
            ordered[upper]
            - ordered[lower]
        )
        * fraction
    )


# ---------------------------------------------------------
# Prometheus metrics parsing
# ---------------------------------------------------------

def extract_metric_value(
    metrics_text,
    metric_name,
):

    """Find one Prometheus metric in text and return its numeric value, or ``None`` if absent."""
    for line in metrics_text.splitlines():

        if line.startswith("#"):
            continue

        if (
            line.startswith(metric_name + "{")
            or line.startswith(metric_name + " ")
        ):

            try:
                return float(
                    line.rsplit(" ", 1)[1]
                )
            except Exception:
                pass

    return None


def fetch_metrics_text():

    """Download the current Prometheus text from the vLLM ``/metrics`` endpoint."""
    response = requests.get(
        METRICS_URL,
        timeout=5,
    )

    response.raise_for_status()

    return response.text


def fetch_metric_snapshot():

    """Read the small set of live vLLM metrics needed by the KV-pressure experiment."""
    text = fetch_metrics_text()

    return {

        "kv_cache_usage":
            extract_metric_value(
                text,
                "vllm:kv_cache_usage_perc",
            ),

        "requests_running":
            extract_metric_value(
                text,
                "vllm:num_requests_running",
            ),

        "requests_waiting":
            extract_metric_value(
                text,
                "vllm:num_requests_waiting",
            ),

        "queue_time_count":
            extract_metric_value(
                text,
                "vllm:request_queue_time_seconds_count",
            ),

        "queue_time_sum":
            extract_metric_value(
                text,
                "vllm:request_queue_time_seconds_sum",
            ),
    }


# ---------------------------------------------------------
# Read static KV-cache configuration directly from vLLM
# ---------------------------------------------------------

def read_cache_config():

    """Parse vLLM's static cache-configuration labels from the metrics endpoint."""
    text = fetch_metrics_text()

    cache_line = None

    for line in text.splitlines():

        if line.startswith(
            "vllm:cache_config_info{"
        ):
            cache_line = line
            break

    if cache_line is None:
        return {}

    def get_label(name):

        match = re.search(
            rf'{re.escape(name)}="([^"]*)"',
            cache_line,
        )

        if match:
            return match.group(1)

        return None

    config = {

        "block_size":
            get_label("block_size"),

        "gpu_memory_utilization":
            get_label(
                "gpu_memory_utilization"
            ),

        "kv_cache_size_tokens":
            get_label(
                "kv_cache_size_tokens"
            ),

        "kv_cache_max_concurrency":
            get_label(
                "kv_cache_max_concurrency"
            ),

        "enable_prefix_caching":
            get_label(
                "enable_prefix_caching"
            ),

        "num_gpu_blocks":
            get_label(
                "num_gpu_blocks"
            ),
    }

    return config


# ---------------------------------------------------------
# Build synthetic prompts of controlled token length
# ---------------------------------------------------------

def make_prompt_for_target_tokens(
    tokenizer,
    target_tokens,
):

    """Build synthetic chat text whose final tokenized prompt matches the requested input-token count."""
    base_content = (
        "Synthetic technical context for "
        "measuring LLM KV-cache pressure."
    )

    messages = [
        {
            "role": "user",
            "content": base_content,
        }
    ]

    base_prompt = (
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )

    base_tokens = len(
        tokenizer(
            base_prompt,
            add_special_tokens=False,
        )["input_ids"]
    )

    if target_tokens <= base_tokens:

        raise ValueError(
            f"Target {target_tokens} is too small. "
            f"Template alone uses {base_tokens} tokens."
        )

    #
    # " data" is intentionally simple synthetic filler.
    # We validate that it is one tokenizer token.
    #
    filler = " data"

    filler_ids = tokenizer(
        filler,
        add_special_tokens=False,
    )["input_ids"]

    if len(filler_ids) != 1:

        raise RuntimeError(
            "Expected filler token to encode "
            "as exactly one token."
        )

    filler_count = (
        target_tokens - base_tokens
    )

    best_prompt = None
    best_token_count = None

    #
    # Adjust until the final chat-formatted prompt
    # reaches the requested token target.
    #
    for _ in range(20):

        content = (
            base_content
            + filler * filler_count
        )

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        formatted_prompt = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        actual_tokens = len(
            tokenizer(
                formatted_prompt,
                add_special_tokens=False,
            )["input_ids"]
        )

        best_prompt = formatted_prompt
        best_token_count = actual_tokens

        difference = (
            target_tokens
            - actual_tokens
        )

        if difference == 0:
            break

        filler_count += difference

        if filler_count < 0:
            filler_count = 0

    return (
        best_prompt,
        best_token_count,
    )


# ---------------------------------------------------------
# Send one inference request
# ---------------------------------------------------------

def send_request(
    request_id,
    formatted_prompt,
    expected_input_tokens,
):

    """Send one fixed-work inference request and return its validated token counts plus latency."""
    payload = {

        "model":
            MODEL,

        "prompt":
            formatted_prompt,

        "temperature":
            0,

        "max_tokens":
            OUTPUT_TOKENS,

        "ignore_eos":
            True,
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

    usage = data["usage"]

    prompt_tokens = usage["prompt_tokens"]

    output_tokens = usage[
        "completion_tokens"
    ]

    if (
        prompt_tokens
        != expected_input_tokens
    ):
        raise RuntimeError(
            f"Request {request_id}: "
            f"expected {expected_input_tokens} "
            f"input tokens, "
            f"server reported {prompt_tokens}"
        )

    if output_tokens != OUTPUT_TOKENS:

        raise RuntimeError(
            f"Request {request_id}: "
            f"expected {OUTPUT_TOKENS} "
            f"output tokens, "
            f"got {output_tokens}"
        )

    latency_sec = (
        end_time - start_time
    )

    return {

        "request_id":
            request_id,

        "input_tokens":
            prompt_tokens,

        "output_tokens":
            output_tokens,

        "latency_sec":
            latency_sec,
    }


# ---------------------------------------------------------
# Background sampler
# ---------------------------------------------------------

def metrics_sampler(
    stop_event,
    samples,
    experiment_start,
    target_context,
    actual_input_tokens,
    batch_number,
):

    """Repeatedly sample live vLLM metrics in a background thread until the batch finishes."""
    while not stop_event.is_set():

        try:

            snapshot = (
                fetch_metric_snapshot()
            )

            samples.append(
                {
                    "context_target":
                        target_context,

                    "actual_input_tokens":
                        actual_input_tokens,

                    "batch_number":
                        batch_number,

                    "elapsed_sec":
                        (
                            time.perf_counter()
                            - experiment_start
                        ),

                    "kv_cache_usage":
                        snapshot[
                            "kv_cache_usage"
                        ],

                    "requests_running":
                        snapshot[
                            "requests_running"
                        ],

                    "requests_waiting":
                        snapshot[
                            "requests_waiting"
                        ],
                }
            )

        except Exception:
            #
            # A single failed metrics sample
            # must not destroy the inference run.
            #
            pass

        stop_event.wait(
            METRIC_SAMPLE_INTERVAL_SEC
        )


# ---------------------------------------------------------
# Run one batch
# ---------------------------------------------------------

def run_batch(
    formatted_prompt,
    target_context,
    actual_input_tokens,
    batch_number,
):

    """Run 80 concurrent requests while sampling metrics, then compute throughput, latency, KV, and queue statistics."""
    before = fetch_metric_snapshot()

    metric_samples = []

    stop_event = threading.Event()

    experiment_start = (
        time.perf_counter()
    )

    sampler_thread = threading.Thread(
        target=metrics_sampler,
        args=(
            stop_event,
            metric_samples,
            experiment_start,
            target_context,
            actual_input_tokens,
            batch_number,
        ),
        daemon=True,
    )

    sampler_thread.start()

    batch_start = time.perf_counter()

    request_results = []

    with ThreadPoolExecutor(
        max_workers=CONCURRENCY
    ) as executor:

        futures = [

            executor.submit(
                send_request,
                request_id,
                formatted_prompt,
                actual_input_tokens,
            )

            for request_id
            in range(CONCURRENCY)
        ]

        for future in as_completed(
            futures
        ):

            request_results.append(
                future.result()
            )

    batch_end = time.perf_counter()

    stop_event.set()
    sampler_thread.join(
        timeout=5
    )

    after = fetch_metric_snapshot()

    batch_wall_time_sec = (
        batch_end - batch_start
    )

    latencies = [
        result["latency_sec"]
        for result in request_results
    ]

    total_output_tokens = sum(
        result["output_tokens"]
        for result in request_results
    )

    aggregate_tokens_per_sec = (
        total_output_tokens
        / batch_wall_time_sec
    )

    kv_values = [

        sample["kv_cache_usage"]

        for sample in metric_samples

        if sample["kv_cache_usage"]
        is not None
    ]

    running_values = [

        sample["requests_running"]

        for sample in metric_samples

        if sample["requests_running"]
        is not None
    ]

    waiting_values = [

        sample["requests_waiting"]

        for sample in metric_samples

        if sample["requests_waiting"]
        is not None
    ]

    peak_kv_usage = (
        max(kv_values)
        if kv_values
        else None
    )

    peak_running = (
        max(running_values)
        if running_values
        else None
    )

    peak_waiting = (
        max(waiting_values)
        if waiting_values
        else None
    )

    before_count = (
        before["queue_time_count"]
        or 0.0
    )

    after_count = (
        after["queue_time_count"]
        or 0.0
    )

    before_sum = (
        before["queue_time_sum"]
        or 0.0
    )

    after_sum = (
        after["queue_time_sum"]
        or 0.0
    )

    queue_count_delta = (
        after_count - before_count
    )

    queue_sum_delta = (
        after_sum - before_sum
    )

    average_queue_time_sec = (

        queue_sum_delta
        / queue_count_delta

        if queue_count_delta > 0
        else 0.0
    )

    batch_result = {

        "context_target":
            target_context,

        "actual_input_tokens":
            actual_input_tokens,

        "concurrency":
            CONCURRENCY,

        "batch_number":
            batch_number,

        "total_output_tokens":
            total_output_tokens,

        "batch_wall_time_sec":
            batch_wall_time_sec,

        "aggregate_tokens_per_sec":
            aggregate_tokens_per_sec,

        "mean_request_latency_sec":
            statistics.mean(
                latencies
            ),

        "median_request_latency_sec":
            statistics.median(
                latencies
            ),

        "p95_request_latency_sec":
            percentile(
                latencies,
                95,
            ),

        "peak_kv_cache_usage":
            peak_kv_usage,

        "peak_requests_running":
            peak_running,

        "peak_requests_waiting":
            peak_waiting,

        "average_queue_time_sec":
            average_queue_time_sec,
    }

    return (
        request_results,
        batch_result,
        metric_samples,
    )


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def main():

    """Sweep all context sizes, preserve request/batch/time-series evidence, and write the final summary JSON."""
    print(
        "=== Prompt2GPU : "
        "KV-Cache Pressure Lab ==="
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL
        )
    )

    cache_config = (
        read_cache_config()
    )

    print("Model:", MODEL)
    print("Concurrency:", CONCURRENCY)
    print(
        "Output tokens/request:",
        OUTPUT_TOKENS,
    )

    print(
        "Context targets:",
        CONTEXT_TARGETS,
    )

    print(
        "\nvLLM KV-cache configuration:"
    )

    for key, value in (
        cache_config.items()
    ):
        print(
            f"  {key}: {value}"
        )

    all_request_rows = []
    all_batch_rows = []
    all_metric_rows = []

    final_summaries = []

    for target_context in (
        CONTEXT_TARGETS
    ):

        (
            formatted_prompt,
            actual_input_tokens,
        ) = make_prompt_for_target_tokens(
            tokenizer,
            target_context,
        )

        print(
            "\n"
            "================================"
        )

        print(
            f"Target context = "
            f"{target_context}"
        )

        print(
            f"Actual input tokens = "
            f"{actual_input_tokens}"
        )

        print(
            "================================"
        )

        #
        # Warmup
        #
        print(
            "Running warmup..."
        )

        run_batch(
            formatted_prompt,
            target_context,
            actual_input_tokens,
            0,
        )

        measured_batches = []
        pooled_latencies = []

        for batch_number in range(
            1,
            MEASURED_BATCHES + 1
        ):

            (
                requests_result,
                batch_result,
                metric_samples,
            ) = run_batch(
                formatted_prompt,
                target_context,
                actual_input_tokens,
                batch_number,
            )

            measured_batches.append(
                batch_result
            )

            all_batch_rows.append(
                batch_result
            )

            all_metric_rows.extend(
                metric_samples
            )

            for result in (
                requests_result
            ):

                pooled_latencies.append(
                    result[
                        "latency_sec"
                    ]
                )

                all_request_rows.append(
                    {

                        "context_target":
                            target_context,

                        "actual_input_tokens":
                            actual_input_tokens,

                        "concurrency":
                            CONCURRENCY,

                        "batch_number":
                            batch_number,

                        "request_id":
                            result[
                                "request_id"
                            ],

                        "latency_sec":
                            result[
                                "latency_sec"
                            ],

                        "output_tokens":
                            result[
                                "output_tokens"
                            ],
                    }
                )

            kv_pct = (
                batch_result[
                    "peak_kv_cache_usage"
                ]
                * 100.0

                if batch_result[
                    "peak_kv_cache_usage"
                ]
                is not None

                else float("nan")
            )

            print(
                f"RUN {batch_number:02d} | "
                f"throughput="
                f"{batch_result['aggregate_tokens_per_sec']:.2f} tok/s | "
                f"mean latency="
                f"{batch_result['mean_request_latency_sec']:.3f}s | "
                f"KV peak="
                f"{kv_pct:.1f}% | "
                f"waiting peak="
                f"{batch_result['peak_requests_waiting']}"
            )

        throughput_values = [

            batch[
                "aggregate_tokens_per_sec"
            ]

            for batch
            in measured_batches
        ]

        kv_values = [

            batch[
                "peak_kv_cache_usage"
            ]

            for batch
            in measured_batches

            if batch[
                "peak_kv_cache_usage"
            ]
            is not None
        ]

        waiting_values = [

            batch[
                "peak_requests_waiting"
            ]

            for batch
            in measured_batches

            if batch[
                "peak_requests_waiting"
            ]
            is not None
        ]

        queue_values = [

            batch[
                "average_queue_time_sec"
            ]

            for batch
            in measured_batches
        ]

        summary = {

            "context_target":
                target_context,

            "actual_input_tokens":
                actual_input_tokens,

            "concurrency":
                CONCURRENCY,

            "measured_batches":
                MEASURED_BATCHES,

            "mean_aggregate_tokens_per_sec":
                statistics.mean(
                    throughput_values
                ),

            "median_aggregate_tokens_per_sec":
                statistics.median(
                    throughput_values
                ),

            "mean_request_latency_sec":
                statistics.mean(
                    pooled_latencies
                ),

            "median_request_latency_sec":
                statistics.median(
                    pooled_latencies
                ),

            "p95_request_latency_sec":
                percentile(
                    pooled_latencies,
                    95,
                ),

            "mean_peak_kv_cache_usage":
                (
                    statistics.mean(
                        kv_values
                    )
                    if kv_values
                    else None
                ),

            "max_peak_kv_cache_usage":
                (
                    max(kv_values)
                    if kv_values
                    else None
                ),

            "max_requests_waiting":
                (
                    max(waiting_values)
                    if waiting_values
                    else None
                ),

            "mean_queue_time_sec":
                statistics.mean(
                    queue_values
                ),
        }

        final_summaries.append(
            summary
        )

        print(
            "\nLEVEL SUMMARY"
        )

        print(
            f"Mean throughput: "
            f"{summary['mean_aggregate_tokens_per_sec']:.2f} tok/s"
        )

        print(
            f"Mean latency: "
            f"{summary['mean_request_latency_sec']:.3f}s"
        )

        print(
            f"P95 latency: "
            f"{summary['p95_request_latency_sec']:.3f}s"
        )

        if (
            summary[
                "max_peak_kv_cache_usage"
            ]
            is not None
        ):

            print(
                f"Peak KV usage: "
                f"{summary['max_peak_kv_cache_usage'] * 100:.1f}%"
            )

        print(
            f"Peak waiting requests: "
            f"{summary['max_requests_waiting']}"
        )

        print(
            f"Mean queue time: "
            f"{summary['mean_queue_time_sec']:.6f}s"
        )

    # -----------------------------------------------------
    # Save request-level raw data
    # -----------------------------------------------------

    with open(
        REQUEST_CSV,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "context_target",
                "actual_input_tokens",
                "concurrency",
                "batch_number",
                "request_id",
                "latency_sec",
                "output_tokens",
            ],
        )

        writer.writeheader()
        writer.writerows(
            all_request_rows
        )

    # -----------------------------------------------------
    # Save batch-level data
    # -----------------------------------------------------

    with open(
        BATCH_CSV,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=all_batch_rows[
                0
            ].keys(),
        )

        writer.writeheader()
        writer.writerows(
            all_batch_rows
        )

    # -----------------------------------------------------
    # Save metrics time-series samples
    # -----------------------------------------------------

    with open(
        METRICS_CSV,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "context_target",
                "actual_input_tokens",
                "batch_number",
                "elapsed_sec",
                "kv_cache_usage",
                "requests_running",
                "requests_waiting",
            ],
        )

        writer.writeheader()
        writer.writerows(
            all_metric_rows
        )

    # -----------------------------------------------------
    # Save JSON summary
    # -----------------------------------------------------

    document = {

        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "experiment":
            "vLLM KV-cache pressure",

        "model":
            MODEL,

        "output_tokens_per_request":
            OUTPUT_TOKENS,

        "concurrency":
            CONCURRENCY,

        "context_targets":
            CONTEXT_TARGETS,

        "warmup_batches_per_level":
            WARMUP_BATCHES,

        "measured_batches_per_level":
            MEASURED_BATCHES,

        "metric_sample_interval_sec":
            METRIC_SAMPLE_INTERVAL_SEC,

        "cache_config":
            cache_config,

        "results":
            final_summaries,
    }

    with open(
        SUMMARY_JSON,
        "w",
    ) as file:

        json.dump(
            document,
            file,
            indent=2,
        )

    # -----------------------------------------------------
    # Final display
    # -----------------------------------------------------

    print(
        "\n\n"
        "================================"
    )

    print(
        "FINAL KV-PRESSURE SUMMARY"
    )

    print(
        "================================"
    )

    for summary in (
        final_summaries
    ):

        kv_pct = (

            summary[
                "max_peak_kv_cache_usage"
            ]
            * 100.0

            if summary[
                "max_peak_kv_cache_usage"
            ]
            is not None

            else float("nan")
        )

        print(
            f"input="
            f"{summary['actual_input_tokens']:4d} | "
            f"throughput="
            f"{summary['mean_aggregate_tokens_per_sec']:8.2f} tok/s | "
            f"mean latency="
            f"{summary['mean_request_latency_sec']:7.3f}s | "
            f"p95="
            f"{summary['p95_request_latency_sec']:7.3f}s | "
            f"KV peak="
            f"{kv_pct:6.1f}% | "
            f"waiting="
            f"{summary['max_requests_waiting']}"
        )

    print(
        "\nRequest data:",
        REQUEST_CSV,
    )

    print(
        "Batch data:",
        BATCH_CSV,
    )

    print(
        "Metrics samples:",
        METRICS_CSV,
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
