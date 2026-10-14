#!/usr/bin/env python3
# Author: Santosh Kumar
# Project: Prompt2GPU
# Version: 1.0
# Experiment: V1.1 data publication, reproducible from local private evidence

"""Validate, sanitize and stage two measured V1.1 C64 runs for public Git.

Only use real, previously measured files supplied via --evidence-root.
No GPU, downloads, web access or fabricated metrics. Full engine logs and
run commands containing user-specific paths remain private.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "initial": "v1_1_c64_20261013T050149Z",
    "repeat": "v1_1_c64_repeat_20261013T052834Z",
}
FILES = (
    "vllm_concurrency_raw.csv",
    "vllm_concurrency_batches.csv",
    "vllm_concurrency_itl.csv",
    "vllm_concurrency_summary.json",
)


def require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def percentile(numbers: list[float], p: float) -> float:
    values = sorted(numbers)
    require(bool(values), f"Cannot compute P{p} from empty values")
    i = (len(values) - 1) * p / 100
    low, high = math.floor(i), math.ceil(i)
    return values[low] + (values[high] - values[low]) * (i - low)


def approx(a: float, b: float, label: str):
    require(math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-6),
            f"{label} mismatch: calculated {a!r}, stored {b!r}")


def no_sensitive_content(path: Path):
    # Refuse data that require manual redaction/review rather than stripping it silently.
    text = path.read_text(encoding="utf-8")
    patterns = [
        r"/content/drive/", r"/" + r"home/[^/\s]+/", r"C:\\Users\\",
        r"-----BEGIN .*PRIVATE KEY-----", r"hf_[A-Za-z0-9]{20,}",
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    ]
    for pattern in patterns:
        require(not re.search(pattern, text),
                f"Potential sensitive/private string in {path.name}; manual review needed: {pattern}")


def check_run(source: Path, expected_source_hash: str) -> dict:
    require(source.is_dir(), f"Missing evidence directory: {source}")
    for name in FILES:
        require((source / "measurements" / name).is_file(), f"Missing {source.name}/{name}")
    script = source / "vllm_concurrency_v1_1.py"
    require(script.is_file(), f"Missing run's benchmark source: {script}")
    require(sha256(script) == expected_source_hash,
            f"Benchmark source for {source.name} differs from committed V1.1 source")

    summary = json.loads((source / "measurements/vllm_concurrency_summary.json").read_text())
    require(summary["stream_mode"] == "streaming", "Only streaming measurement supported")
    require(summary["prefix_caching"] == "on", "Prefix-caching config mismatch")
    require(summary["concurrency_levels"] == [64], "Concurrency mismatch")
    require(summary["measured_batches_per_level"] == 5, "Measured-batch mismatch")
    require(summary["warmup_batches_per_level"] == 1, "Warmup mismatch")
    require(summary["output_tokens_per_request"] == 128, "Output length mismatch")
    require(summary["vllm_version_declared"] == "0.31.0", "vLLM version metadata mismatch")
    r = summary["results"][0]
    require(int(r["measured_requests"]) == 320, "Expected 320 measured requests")
    raw = rows(source / "measurements/vllm_concurrency_raw.csv")
    batches = rows(source / "measurements/vllm_concurrency_batches.csv")
    itls = rows(source / "measurements/vllm_concurrency_itl.csv")
    require(len(raw) == 320, f"Expected 320 measured request rows, got {len(raw)}")
    require(len(batches) == 5, f"Expected five measured batch rows, got {len(batches)}")
    for batch in batches:
        n = int(batch["batch_number"])
        matching = [row for row in raw if int(row["batch_number"]) == n]
        require(len(matching) == 64 and int(batch["request_count"]) == 64,
                f"Batch {n} request count mismatch")
        require(all(int(row["output_tokens"]) == 128 for row in matching),
                f"Batch {n} output-token mismatch")
        actual_tokens = sum(int(row["output_tokens"]) for row in matching)
        require(actual_tokens == int(batch["total_output_tokens"]),
                f"Batch {n} total tokens inconsistent")
        wall = float(batch["batch_wall_time_sec"])
        require(wall > 0, f"Batch {n} has nonpositive wall time")
        approx(actual_tokens / wall, float(batch["aggregate_tokens_per_sec"]),
               f"Batch {n} aggregate throughput")
        approx(statistics.mean(float(row["latency_sec"]) for row in matching),
               float(batch["mean_request_latency_sec"]), f"Batch {n} mean request latency")

    latencies = [float(row["latency_sec"]) for row in raw]
    ttft = [float(row["ttft_sec"]) for row in raw if row["ttft_sec"]]
    tpot = [float(row["tpot_sec"]) for row in raw if row["tpot_sec"]]
    intervals = [float(row["itl_sec"]) for row in itls]
    eligible = sum(row["itl_valid"].lower() == "true" for row in raw)
    rates = [float(b["aggregate_tokens_per_sec"]) for b in batches]
    approx(statistics.mean(rates), r["mean_aggregate_tokens_per_sec"], "Mean throughput")
    approx(statistics.mean(latencies), r["mean_request_latency_sec"], "Mean latency")
    approx(percentile(latencies, 95), r["p95_request_latency_sec"], "P95 latency")
    approx(percentile(latencies, 99), r["p99_request_latency_sec"], "P99 latency")
    approx(percentile(ttft, 95), r["p95_ttft_sec"], "P95 TTFT")
    approx(percentile(tpot, 95), r["p95_tpot_sec"], "P95 TPOT")
    require(eligible == r["itl_eligible_requests"], "ITL eligible count mismatch")
    require(len(intervals) == r["itl_total_measured_intervals"], "ITL interval count mismatch")
    if intervals:
        approx(percentile(intervals, 95), r["p95_itl_sec"], "P95 ITL")

    telemetry_file = source / "observability/prometheus_timeseries.json"
    require(telemetry_file.is_file(), f"Missing Prometheus export in {source.name}")
    telemetry = json.loads(telemetry_file.read_text())
    require(len(telemetry["queries"]) == 12, "Expected twelve Prometheus metric queries")
    require(telemetry["benchmark_start_unix_s"] < telemetry["benchmark_end_unix_s"],
            "Invalid benchmark timeline")
    for name in (*FILES[:3],):
        no_sensitive_content(source / "measurements" / name)
    no_sensitive_content(telemetry_file)

    return {"summary": summary, "raw": raw, "batches": batches,
            "itl": itls, "telemetry": telemetry,
            "source_hash": sha256(script), "source": source,
            "full_log_hash": sha256(source / "observability/vllm_server.log")
                             if (source / "observability/vllm_server.log").is_file() else None}


def safe_summary(summary: dict) -> dict:
    return {k: v for k, v in summary.items() if k not in ("server_url",)}


def generate_findings(first: dict, repeat: dict) -> str:
    a, b = first["summary"]["results"][0], repeat["summary"]["results"][0]
    first_second_batch = first["batches"][1]
    first_other_batches = [float(x["aggregate_tokens_per_sec"]) for i,x in enumerate(first["batches"]) if i != 1]
    ttft_by_batch = {}
    for idx in range(1, 6):
        matches = [float(row["ttft_sec"]) for row in first["raw"] if int(row["batch_number"]) == idx]
        ttft_by_batch[idx] = statistics.mean(matches)
    require(first["batches"][1]["batch_number"] == "2", "Batch order incorrect")
    return f"""# Prompt2GPU V1.1 — Measured streaming C64 and JIT investigation

**Measured environment:** Google Colab NVIDIA Tesla T4, Qwen/Qwen2.5-1.5B-Instruct, FP16, vLLM 0.31.0; 69 input tokens and 128 forced completion tokens per request. Five measured batches × 64 concurrent **HTTP requests**, following one warm-up batch; prefix caching enabled in the running server. Distinguish 64 requests submitted concurrently from 64 GPU-running sequences at every instant.

**Real measurements:** From the two dated runs, reproduced and checked by `scripts/publish_v1_1.py` against their raw request, batch and ITL interval CSV files. Full unredacted engine logs are held privately and **not published**.

| Metric | Initial run | Warm-server repeat |
|---|---:|---:|
| Mean aggregate output throughput | {a['mean_aggregate_tokens_per_sec']:.2f} tokens/s | {b['mean_aggregate_tokens_per_sec']:.2f} tokens/s |
| Mean request end-to-end latency | {a['mean_request_latency_sec']:.3f} s | {b['mean_request_latency_sec']:.3f} s |
| P95 end-to-end latency | {a['p95_request_latency_sec']:.3f} s | {b['p95_request_latency_sec']:.3f} s |
| P95 client-observed TTFT | {a['p95_ttft_sec']*1000:.1f} ms | {b['p95_ttft_sec']*1000:.1f} ms |
| P95 client-estimated TPOT | {a['p95_tpot_sec']*1000:.2f} ms | {b['p95_tpot_sec']*1000:.2f} ms |
| ITL-eligible requests | {a['itl_eligible_requests']}/320 | {b['itl_eligible_requests']}/320 |

## Anomalous initial Batch 2

All five measured batches generated exactly 8,192 output tokens. Initial Batch 2 delivered **{float(first_second_batch['aggregate_tokens_per_sec']):.2f} tokens/s**, compared with **{statistics.mean(first_other_batches):.2f} tokens/s** mean for the other four initial measured batches. Mean client-observed TTFT rose from **{ttft_by_batch[1]*1000:.1f} ms** in Batch 1 to **{ttft_by_batch[2]*1000:.1f} ms** in Batch 2, while TPOT remained comparatively stable. In the private engine log, a warning at 2026-10-13 05:02:05 reported `Triton kernel JIT compilation during inference: kernel_unified_attention`. That timestamp falls within measured Batch 2 (05:02:03.634–05:02:09.330 UTC). One such warning was recorded during the initial run; none was newly recorded during the warm-server repeat.

**Inference, not proof:** a newly required Triton kernel specialization plausibly contributed to the TTFT spike. The warm-server repeat removed the anomaly, strengthening the hypothesis, but cache state, server warm state and other run-to-run variables were not independently controlled. Do **not** describe this as a measured isolated JIT causal effect or a demonstrated optimization speedup.

## How to interpret the Triton JIT annotation

Triton uses just-in-time (JIT) compilation to turn a needed GPU-kernel specialization into executable code. A model can be warmed up without covering every future kernel shape or execution configuration. A newly encountered specialization may therefore incur compilation overhead during inference, potentially delaying the first streamed token. The log identified `kernel_unified_attention`, but it did **not** measure compilation duration or prove which individual request experienced it.

In the initial run, Batch 2's mean **client-observed TTFT** was approximately **1,895 ms**, compared with approximately **350 ms** in Batch 1. Mean TPOT was roughly **28.7 ms** versus **28.4 ms**. This localizes most of the extra request latency to *before* the first token, rather than to sustained token decoding. The JIT warning falls in Batch 2; the warm-server repeat had no new warning and showed steadier batch results. **Conclusion: strong temporal evidence supporting a JIT-related latency anomaly, not an isolated causal proof.**

**Live capture versus replay:** These were real T4 GPU experiments. During execution, Prometheus scraped vLLM metrics every five seconds and Grafana visualized that data. The [GitHub Pages V1.1 dashboard](docs/v1_1.html) is a **static historical replay of the same recorded Prometheus query results**, combined with the actual client-side CSV measurements. Its charts use Plotly.js for interaction; they do not simulate inference, invent missing values, interpolate between measurements, or contact a running Grafana/vLLM server.

## Why Prometheus does not pinpoint the whole event

The 12 exported Prometheus queries (five-second steps) include running/waiting request gauges, KV-cache usage, rolling TTFT/TPOT/queue histograms, prefix-cache counters and token-throughput rates. Batches lasted roughly 4–6 seconds; individual samples may miss entire batches. One-minute rate windows and five-minute histogram windows are **not** the same as per-batch client measurements. Zero waiting at a sampled instant does not rule out transient queueing. Do not conflate server `histogram_quantile` values with client-side P95 values.

## Prefill vs decode: recorded server metric and client-derived proxy

**Prefill:** P95 prefill duration is available from the exported server-side Prometheus histogram query (five-minute rolling window). **Decode phase — client proxy P95:** for each measured streamed request we derive `(output_tokens − 1) × TPOT` (elapsed time from the first to last nonempty streamed content event); the dashboard plots the P95 of those values for each group of 64 requests. These are **real request measurements, not synthetic or simulated GPU inference**. The decode proxy includes HTTP/client/network and serving overhead, and is **not the server-side GPU decode execution duration**. We did not capture the dedicated `vllm:request_decode_time_seconds` histogram in this saved export. Client-derived decode and server prefill are not measured at equivalent boundaries.

## Streaming measurement limitations

- TTFT includes HTTP/client transport and scheduling from request submission to first nonempty SSE content event.
- Per-request TPOT is `(last content event − first content event)/(128−1)`.
- ITL is conditional **SSE content-event** spacing; counts matching 128 tokens do not prove exactly one token per SSE event.
- These are streaming-mode measurements; V1 nonstreaming comparisons are **not** a controlled cache or engine-speedup experiment.
- All time-aligned arrays in `results_v1_1/` are **historical recorded telemetry**, not a live dashboard.

## Reproduce the publication checks (CPU only)

Run from the repository root with your local evidence directory as an argument:

```bash
python3 scripts/publish_v1_1.py --evidence-root ../Prompt2GPU-evidence --check-only
python3 -m unittest discover -s tests -p 'test_concurrency_v1_1.py' -v
make verify
make audit
```

The script requires the original saved measurements, never fetches models, and refuses to generate missing data. To prepare the public results initially, omit `--check-only`. Both runs' original `command.json`, benchmark source snapshots and full engine logs remain in private local evidence. The published summary omits `server_url`; raw CSV/JSON evidence is preserved unchanged. The reproducible V1.1 benchmark source is `src/vllm_concurrency_v1_1.py`.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true",
                        help="Validate every measurement without changing repository files")
    args = parser.parse_args()
    evidence = args.evidence_root.resolve()
    source_script = ROOT / "src/vllm_concurrency_v1_1.py"
    require(source_script.is_file(), "Missing src/vllm_concurrency_v1_1.py")
    expected_hash = sha256(source_script)
    checked = {label: check_run(evidence / folder, expected_hash)
               for label, folder in RUNS.items()}
    print("PASS: two authentic C64 measured runs; raw/batches/ITL/telemetry consistent")
    for label, data in checked.items():
        result = data["summary"]["results"][0]
        print(f" {label}: 320 measured requests, mean throughput="
              f"{result['mean_aggregate_tokens_per_sec']:.2f} tok/s; "
              f"P95 TTFT={result['p95_ttft_sec']*1000:.1f} ms")
    if args.check_only:
        return

    # Evidence gate: the documented JIT warning must actually occur in the private
    # initial-run log during Batch 2. Do not generate the JIT claim without proof.
    initial_log = checked["initial"]["source"] / "observability/vllm_server.log"
    require(initial_log.is_file(), "Initial vLLM log missing; cannot verify JIT claim")
    jit_matches = [line for line in initial_log.read_text(errors="replace").splitlines()
                   if "Triton kernel JIT compilation during inference:" in line
                   and "kernel_unified_attention" in line]
    require(len(jit_matches) == 1 and "10-13 05:02:05" in jit_matches[0],
            "Expected initial-run JIT evidence not found; review manually")
    repeat_check = checked["repeat"]["source"] / "jit_repeat_check.json"
    require(repeat_check.is_file(), "Missing repeat JIT warning-count evidence")
    repeat_jit = json.loads(repeat_check.read_text())
    require(repeat_jit["jit_warnings_before"] == 1 and
            repeat_jit["jit_warnings_after"] == 1 and
            repeat_jit["new_jit_events"] == [],
            "Repeat JIT log-counter evidence differs from published claim")

    target_root = ROOT / "results_v1_1"
    require(not target_root.exists() or not list(target_root.iterdir()) or
            all((target_root / label).is_dir() for label in RUNS),
            "Unexpected results_v1_1 layout; review before overwriting")
    for label, data in checked.items():
        dest = target_root / label
        dest.mkdir(parents=True, exist_ok=True)
        for name in FILES[:3]:
            shutil.copy2(data["source"] / "measurements" / name, dest / name)
        (dest / "vllm_concurrency_summary.json").write_text(
            json.dumps(safe_summary(data["summary"]), indent=2) + "\n")
        shutil.copy2(data["source"] / "observability/prometheus_timeseries.json",
                     dest / "prometheus_timeseries.json")
        manifest = {
            "original_run_folder": RUNS[label],
            "original_benchmark_source_sha256": data["source_hash"],
            "private_engine_log_sha256": data["full_log_hash"],
            "published_measurements_sha256": {
                name: sha256(dest / name) for name in (*FILES, "prometheus_timeseries.json")
            },
            "engine_log_published": False,
            "artifact_classification": "real measured Colab T4 data; not synthetic",
        }
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    findings = generate_findings(checked["initial"], checked["repeat"])
    (ROOT / "V1_1_FINDINGS.md").write_text(findings)
    readme_file = ROOT / "README.md"
    readme = readme_file.read_text()
    marker = "## V1.1 — Streaming observability and Triton JIT investigation"
    if marker not in readme:
        before = "## View the published results — zero install"
        require(before in readme, "README insertion anchor missing; do not blindly modify")
        paragraph = (
            f"{marker}\n\n"
            "Measured follow-on C64/T4 experiments add client-observed TTFT/TPOT/conditional "
            "ITL, 12 exported Prometheus time series, and a corroborated runtime log warning "
            "during an anomalous batch. The repeat was more consistent without a new "
            "Triton JIT warning. This is correlation, **not** an isolated causal proof. "
            "See [V1.1 measured findings](V1_1_FINDINGS.md) and "
            "[`results_v1_1/`](results_v1_1/) for the original measured evidence. "
            "The existing V1.0 dashboard remains V1.0 and is not presented as V1.1 telemetry.\n\n"
        )
        readme_file.write_text(readme.replace(before, paragraph + before, 1))
    print("PASS: public artifacts staged; full logs and private paths NOT copied")
    print("Next: make verify && python3 -m unittest discover -s tests -v && make audit")
    print("Inspect `git diff` and `git status` before committing; NEVER auto-push")


if __name__ == "__main__":
    main()
