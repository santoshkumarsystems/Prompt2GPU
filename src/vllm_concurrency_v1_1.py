# Author: Santosh Kumar
# Project: Prompt2GPU
# Version: 1.0
# Experiment: V1.1 extension (original V1 remains unchanged)

"""V1.1 extension of the original vLLM concurrency benchmark.

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

V1.1 keeps the same V1 workload, warm-up/measurement batches, concurrency,
original aggregate throughput calculations, and output-token validation.
It adds optional SSE streaming instrumentation for client TTFT / TPOT / ITL,
P99 summaries, run-specific output paths, and explicit cache metadata.

Inter-token latency (ITL) is recorded only when each content SSE event can be
matched one-to-one to an output token. Otherwise we preserve chunk timings
without claiming they are token latencies. No V1 results are overwritten.
"""


import argparse
import csv
import json
import math
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
from transformers import AutoTokenizer

from benchmark_config import MODEL, PROMPT, OUTPUT_TOKENS, VLLM_URL, PROJECT_ROOT


# Output directories are versioned and run-specific. Never write to V1 results/.
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RESULTS_DIR = str(PROJECT_ROOT / "results_v1_1" / f"concurrency_{RUN_ID}")
RAW_REQUEST_CSV = f"{RESULTS_DIR}/vllm_concurrency_raw.csv"
BATCH_CSV = f"{RESULTS_DIR}/vllm_concurrency_batches.csv"
SUMMARY_JSON = f"{RESULTS_DIR}/vllm_concurrency_summary.json"
ITL_CSV = f"{RESULTS_DIR}/vllm_concurrency_itl.csv"

CONCURRENCY_LEVELS = [64]  # Original V1 full sweep is unchanged elsewhere.
WARMUP_BATCHES = 1
MEASURED_BATCHES = 5
REQUEST_TIMEOUT_SEC = 180

# Explicitly declared experiment configuration; do not assume cache settings.
MODE = "streaming"
PREFIX_CACHING = "unknown"
SERVER_VERSION = "unknown"


def configure_run(args):
    """Apply CLI overrides, keeping V1 results/ safely out of scope."""
    global RESULTS_DIR, RAW_REQUEST_CSV, BATCH_CSV, SUMMARY_JSON, ITL_CSV
    global CONCURRENCY_LEVELS, WARMUP_BATCHES, MEASURED_BATCHES
    global MODE, PREFIX_CACHING, SERVER_VERSION, VLLM_URL

    CONCURRENCY_LEVELS = [int(x.strip()) for x in args.levels.split(",")]
    if not CONCURRENCY_LEVELS or any(x < 1 for x in CONCURRENCY_LEVELS):
        raise ValueError("--levels must contain positive concurrency values")
    WARMUP_BATCHES = args.warmup_batches
    MEASURED_BATCHES = args.measured_batches
    if WARMUP_BATCHES < 0 or MEASURED_BATCHES < 1:
        raise ValueError("warmup must be >= 0; measured batches must be >= 1")
    MODE = args.mode
    PREFIX_CACHING = args.prefix_caching
    SERVER_VERSION = args.vllm_version
    VLLM_URL = args.server.rstrip("/") + "/v1/completions"
    RESULTS_DIR = str(Path(args.output_dir).expanduser().resolve())
    if (Path(RESULTS_DIR) in (PROJECT_ROOT, PROJECT_ROOT / "docs")
            or Path(RESULTS_DIR) == PROJECT_ROOT / "results"
            or (PROJECT_ROOT / "results") in Path(RESULTS_DIR).parents):
        raise ValueError("Refusing to write to repository root, docs/, or V1 results/")
    RAW_REQUEST_CSV = f"{RESULTS_DIR}/vllm_concurrency_raw.csv"
    BATCH_CSV = f"{RESULTS_DIR}/vllm_concurrency_batches.csv"
    SUMMARY_JSON = f"{RESULTS_DIR}/vllm_concurrency_summary.json"
    ITL_CSV = f"{RESULTS_DIR}/vllm_concurrency_itl.csv"

    # Refuse to overwrite even an existing V1.1 experiment; use a new run ID.
    destination = Path(RESULTS_DIR)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output directory already contains files: {destination}")
    destination.mkdir(parents=True, exist_ok=True)


def parser():
    ap = argparse.ArgumentParser(
        description="Prompt2GPU V1.1: copy of V1 concurrency benchmark with streaming latencies"
    )
    ap.add_argument("--levels", default="64", help="Comma-separated concurrency levels (default: 64)")
    ap.add_argument("--mode", choices=("streaming", "nonstreaming"), default="streaming")
    ap.add_argument("--warmup-batches", type=int, default=1)
    ap.add_argument("--measured-batches", type=int, default=5)
    ap.add_argument("--prefix-caching", choices=("on", "off", "unknown"), default="unknown",
                    help="Declared runtime setting; verify from server startup flags")
    ap.add_argument("--vllm-version", default="unknown", help="Declared installed vLLM version")
    ap.add_argument("--server", default=VLLM_URL.rsplit("/v1/completions", 1)[0],
                    help="Existing vLLM base URL; does not launch a server")
    ap.add_argument("--output-dir", default=RESULTS_DIR,
                    help="Fresh directory for this run (never V1 results/)")
    return ap


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
    """V1 workload, with streaming timing added for the V1.1 experiment."""
    payload = {
        "model": MODEL,
        "prompt": formatted_prompt,
        "temperature": 0,
        "max_tokens": OUTPUT_TOKENS,
        "ignore_eos": True,  # V1 fixed-length generation rule; unchanged.
    }

    start_wall_utc = time.time()
    start_time = time.perf_counter()

    if MODE == "nonstreaming":
        # Preserve V1's request/response timing for compatible control runs.
        response = requests.post(VLLM_URL, json=payload, timeout=REQUEST_TIMEOUT_SEC)
        end_time = time.perf_counter()
        response.raise_for_status()
        output_tokens = response.json()["usage"]["completion_tokens"]
        first_token_time = None
        last_token_time = None
        content_times = []
        content_event_wall_times = []
    else:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        first_token_time = None
        last_token_time = None
        content_times = []
        content_event_wall_times = []
        output_tokens = None
        with requests.post(VLLM_URL, json=payload, stream=True,
                           timeout=(10, REQUEST_TIMEOUT_SEC)) as response:
            response.raise_for_status()
            # chunk_size=64 balances SSE timing vs client CPU overhead for timing accuracy;
            # measurements include Python/client/network scheduling overhead.
            for line in response.iter_lines(chunk_size=64, decode_unicode=True):
                # Some servers/requests versions yield raw bytes for SSE.
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if not line or not line.startswith("data:"):
                    continue
                text = line[5:].strip()
                if text == "[DONE]":
                    break
                event = json.loads(text)
                usage = event.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    output_tokens = usage["completion_tokens"]
                for choice in event.get("choices", []):
                    if choice.get("text"):
                        now = time.perf_counter()
                        wall = time.time()
                        content_times.append(now)
                        content_event_wall_times.append(wall)
                        if first_token_time is None:
                            first_token_time = now
                        last_token_time = now
        end_time = time.perf_counter()
        if output_tokens is None:
            raise RuntimeError(
                f"Request {request_id}: streaming usage missing; "
                "check stream_options include_usage server support"
            )
        if first_token_time is None:
            raise RuntimeError(f"Request {request_id}: no streamed output content")

    if output_tokens != OUTPUT_TOKENS:
        raise RuntimeError(
            f"Request {request_id}: expected {OUTPUT_TOKENS} output tokens, "
            f"got {output_tokens}"
        )

    latency_sec = end_time - start_time
    ttft = first_token_time - start_time if first_token_time is not None else None
    tpot = ((last_token_time - first_token_time) / (output_tokens - 1)
            if last_token_time is not None and output_tokens > 1 else None)

    # SSE events are not guaranteed to be one token each. Matching event/token
    # counts is a necessary but not sufficient test for token-level ITL;
    # these values should be described as conditional SSE-based estimates.
    itl_valid = MODE == "streaming" and len(content_times) == output_tokens
    intervals = ([content_times[i] - content_times[i - 1]
                  for i in range(1, len(content_times))] if itl_valid else [])
    itl_records = ([{"interval_index": i,
                     "itl_sec": content_times[i] - content_times[i - 1],
                     "event_unix_s": content_event_wall_times[i]}
                    for i in range(1, len(content_times))] if itl_valid else [])

    return {
        "request_id": request_id,
        "latency_sec": latency_sec,
        "output_tokens": output_tokens,
        "request_tokens_per_sec": output_tokens / latency_sec,
        "start_unix_s": start_wall_utc,
        "ttft_sec": ttft,
        "tpot_sec": tpot,
        "stream_content_events": len(content_times),
        "itl_valid": itl_valid,
        "itl_mean_sec": statistics.mean(intervals) if intervals else None,
        "itl_p95_sec": percentile(intervals, 95),
        "itl_p99_sec": percentile(intervals, 99),
        "itl_records": itl_records,
    }


def run_batch(
    concurrency,
    batch_number,
    phase,
    formatted_prompt,
):
    """Launch one group of simultaneous requests and summarize batch wall time, throughput, and latencies."""
    batch_start_unix_s = time.time()
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
    batch_end_unix_s = time.time()

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
        "batch_start_unix_s": batch_start_unix_s,
        "batch_end_unix_s": batch_end_unix_s,
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


def main(argv=None):
    """Run V1 workload with V1.1 streaming instrumentation and new output files."""
    configure_run(parser().parse_args(argv))
    # Fail fast if no existing server is running (never start a second server).
    health_url = VLLM_URL.rsplit("/v1/completions", 1)[0] + "/health"
    health = requests.get(health_url, timeout=10)
    health.raise_for_status()
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

    print("Mode:", MODE)
    print("Declared prefix caching:", PREFIX_CACHING)
    print("Output directory:", RESULTS_DIR)

    raw_request_rows = []
    measured_batch_rows = []
    itl_rows = []
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
        concurrency_ttfts = []
        concurrency_tpots = []
        concurrency_itls = []
        itl_valid_requests = 0

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

                if request_result["ttft_sec"] is not None:
                    concurrency_ttfts.append(request_result["ttft_sec"])
                if request_result["tpot_sec"] is not None:
                    concurrency_tpots.append(request_result["tpot_sec"])
                if request_result["itl_valid"]:
                    itl_valid_requests += 1
                for record in request_result["itl_records"]:
                    concurrency_itls.append(record["itl_sec"])
                    itl_rows.append({
                        "concurrency": concurrency,
                        "batch_number": measured_index,
                        "request_id": request_result["request_id"],
                        **record,
                    })

                raw_request_rows.append(
                    {
                        "concurrency": concurrency,
                        "batch_number": measured_index,
                        "request_id": request_result["request_id"],
                        "latency_sec": latency,
                        "output_tokens": request_result["output_tokens"],
                        "request_tokens_per_sec": request_result["request_tokens_per_sec"],
                        "batch_wall_time_sec": batch_result["batch_wall_time_sec"],
                        "aggregate_tokens_per_sec": batch_result["aggregate_tokens_per_sec"],
                        "start_unix_s": request_result["start_unix_s"],
                        "ttft_sec": request_result["ttft_sec"],
                        "tpot_sec": request_result["tpot_sec"],
                        "stream_content_events": request_result["stream_content_events"],
                        "itl_valid": request_result["itl_valid"],
                        "itl_mean_sec": request_result["itl_mean_sec"],
                        "itl_p95_sec": request_result["itl_p95_sec"],
                        "itl_p99_sec": request_result["itl_p99_sec"],
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
            # Additional V1.1 metrics; null when not available.
            "p99_request_latency_sec": percentile(concurrency_request_latencies, 99),
            "mean_ttft_sec": statistics.mean(concurrency_ttfts) if concurrency_ttfts else None,
            "p50_ttft_sec": percentile(concurrency_ttfts, 50),
            "p95_ttft_sec": percentile(concurrency_ttfts, 95),
            "p99_ttft_sec": percentile(concurrency_ttfts, 99),
            "mean_tpot_sec": statistics.mean(concurrency_tpots) if concurrency_tpots else None,
            "p50_tpot_sec": percentile(concurrency_tpots, 50),
            "p95_tpot_sec": percentile(concurrency_tpots, 95),
            "p99_tpot_sec": percentile(concurrency_tpots, 99),
            "itl_eligible_requests": itl_valid_requests,
            "itl_total_measured_intervals": len(concurrency_itls),
            "mean_itl_sec": statistics.mean(concurrency_itls) if concurrency_itls else None,
            "p50_itl_sec": percentile(concurrency_itls, 50),
            "p95_itl_sec": percentile(concurrency_itls, 95),
            "p99_itl_sec": percentile(concurrency_itls, 99),
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
                "start_unix_s",
                "ttft_sec",
                "tpot_sec",
                "stream_content_events",
                "itl_valid",
                "itl_mean_sec",
                "itl_p95_sec",
                "itl_p99_sec",
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
                "batch_start_unix_s",
                "batch_end_unix_s",
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

    # V1.1 raw token-level intervals. Absent when stream events are batched.
    with open(ITL_CSV, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[
            "concurrency", "batch_number", "request_id",
            "interval_index", "itl_sec", "event_unix_s",
        ])
        writer.writeheader()
        writer.writerows(itl_rows)

    #
    # Save final summary
    #
    summary_document = {

        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "experiment":
            "vLLM concurrency scaling V1.1",

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

        "prefix_caching": PREFIX_CACHING,
        "prefix_caching_source": "user-declared server flag; verify from vLLM startup logs",
        "stream_mode": MODE,
        "server_url": VLLM_URL,
        "vllm_version_declared": SERVER_VERSION,
        "timing_method": "client-side perf_counter; SSE content events; usage from server",
        "itl_definition": (
            "SSE content-event inter-arrival used as conditional ITL estimate only "
            "when nonempty content event count equals completion token count. "
            "Matching counts do not prove one token per event. Otherwise ITL omitted."
        ),
        "tpot_definition": (
            "(last nonempty streamed content event - first nonempty streamed "
            "content event) / (completion_tokens - 1)"
        ),
        "reference_source": "src/vllm_concurrency.py (original V1; unchanged)",

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

    print("ITL intervals:", ITL_CSV)
    print("Summary:", SUMMARY_JSON)
    print("NOTE: Grafana/Prometheus data must be exported separately and time-aligned.")


# BEGINNER NOTE: Python sets ``__name__`` to ``"__main__"`` when this
# file is run directly.  When another file imports it, this block is skipped.
# That lets us reuse functions without accidentally starting the whole program.
if __name__ == "__main__":
    main()
