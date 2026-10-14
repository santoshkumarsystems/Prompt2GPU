#!/usr/bin/env python3
# Author: Santosh Kumar
# Project: Prompt2GPU
# Version: 1.0
# Experiment: V1.1 historical observability replay
"""Build/verify a historical-only web dashboard from V1.1 committed raw data.

Standard library only. Missing metrics remain missing; no interpolation, invented
requests, synthetic GPU measurements, network access, or GPU dependencies.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results_v1_1"
TARGET = ROOT / "docs" / "v1_1_data"
REQUIRED_METRICS = (
    "output_tokens_per_sec", "running_requests", "waiting_requests",
    "kv_cache_usage_percent", "p95_ttft_seconds", "p95_tpot_seconds",
    "p95_queue_seconds", "p95_prefill_seconds",
    "prefix_cache_hit_rate_percent", "generation_tokens_counter",
    "prefix_cache_hits_counter", "prefix_cache_queries_counter",
)
RUNS = ("initial", "repeat")


def require(value, message):
    if not value:
        raise ValueError(message)


def load_json(path: Path):
    require(path.is_file(), f"Required measured data missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path):
    require(path.is_file(), f"Required measured data missing: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def percentile(values, p):
    require(values, "No values for percentile")
    ordered = sorted(values)
    at = (len(ordered) - 1) * p / 100
    low, high = math.floor(at), math.ceil(at)
    return ordered[low] + (ordered[high] - ordered[low]) * (at - low)


def finite_float(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def decode_phase_client_proxy_ms(request):
    """Recorded first-to-last streamed content span, NOT GPU decode time."""
    count = int(request["output_tokens"])
    tpot = finite_float(request.get("tpot_sec"))
    require(count >= 2 and tpot is not None and tpot >= 0,
            "Cannot derive decode phase proxy: missing/invalid streamed TPOT")
    duration_ms = (count - 1) * tpot * 1000
    require(math.isfinite(duration_ms), "Nonfinite decode phase proxy")
    return duration_ms


def convert_run(label: str):
    folder = SOURCE / label
    # The publication script records checksums for each measured source file.
    manifest = load_json(folder / "manifest.json")
    digests = manifest.get("published_measurements_sha256", {})
    for filename in ("vllm_concurrency_raw.csv", "vllm_concurrency_batches.csv",
                     "vllm_concurrency_itl.csv", "vllm_concurrency_summary.json",
                     "prometheus_timeseries.json"):
        saved = folder / filename
        require(saved.is_file(), f"{label}: missing published source {filename}")
        require(digests.get(filename) == hashlib.sha256(saved.read_bytes()).hexdigest(),
                f"{label}: published SHA256 manifest mismatch for {filename}")
    require(manifest.get("engine_log_published") is False,
            f"{label}: private engine log publication flag unexpected")
    summary = load_json(folder / "vllm_concurrency_summary.json")
    export = load_json(folder / "prometheus_timeseries.json")
    batches = load_csv(folder / "vllm_concurrency_batches.csv")
    raw = load_csv(folder / "vllm_concurrency_raw.csv")
    require(len(batches) == 5, f"{label}: expected five measured batches")
    require(len(raw) == 320, f"{label}: expected 320 measured requests")
    require(summary.get("concurrency_levels") == [64], f"{label}: concurrency mismatch")
    require(summary.get("stream_mode") == "streaming", f"{label}: expected streaming")
    require(summary.get("prefix_caching") == "on", f"{label}: prefix cache config mismatch")
    require(summary.get("measured_batches_per_level") == 5, f"{label}: batch count mismatch")
    require(summary.get("output_tokens_per_request") == 128, f"{label}: token count mismatch")
    require(set(export.get("queries", {})) == set(REQUIRED_METRICS),
            f"{label}: Prometheus metric set unexpectedly changed")
    require(float(export["benchmark_start_unix_s"]) < float(export["benchmark_end_unix_s"]),
            f"{label}: invalid benchmark timestamps")
    start = float(export["benchmark_start_unix_s"])
    end = float(export["benchmark_end_unix_s"])
    items = []
    for number, batch in enumerate(batches, 1):
        require(int(batch["batch_number"]) == number, f"{label}: batch order mismatch")
        requests = [r for r in raw if int(r["batch_number"]) == number]
        require(len(requests) == 64, f"{label}: missing batch requests")
        ts1 = float(batch["batch_start_unix_s"])
        ts2 = float(batch["batch_end_unix_s"])
        require(start <= ts1 < ts2 <= end, f"{label}: batch outside benchmark")
        ttft = [float(r["ttft_sec"]) for r in requests if r.get("ttft_sec")]
        tpot = [float(r["tpot_sec"]) for r in requests if r.get("tpot_sec")]
        decode_ms = [decode_phase_client_proxy_ms(r) for r in requests]
        require(len(decode_ms) == 64, f"{label}: missing decode proxy observations")
        require(ttft and tpot, f"{label}: missing streaming observations")
        tokens = sum(int(r["output_tokens"]) for r in requests)
        require(tokens == 8192 == int(batch["total_output_tokens"]),
                f"{label}: batch token count mismatch")
        rate = tokens / float(batch["batch_wall_time_sec"])
        require(math.isclose(rate, float(batch["aggregate_tokens_per_sec"]), rel_tol=1e-8),
                f"{label}: batch throughput differs from source")
        items.append({
            "batch": number, "start_unix_s": ts1, "end_unix_s": ts2,
            "throughput_tok_s": rate,
            "mean_ttft_ms": statistics.mean(ttft) * 1000,
            "p95_ttft_ms": percentile(ttft, 95) * 1000,
            "mean_tpot_ms": statistics.mean(tpot) * 1000,
            "p95_tpot_ms": percentile(tpot, 95) * 1000,
            "p95_decode_phase_client_proxy_ms": percentile(decode_ms, 95),
            "mean_latency_s": float(batch["mean_request_latency_sec"]),
            "wall_time_s": float(batch["batch_wall_time_sec"]),
        })

    timeline = {}
    for name in REQUIRED_METRICS:
        query = export["queries"][name]
        values = []
        series_list = query.get("series", [])
        require(len(series_list) <= 1,
                f"{label}/{name}: unexpected multiple Prometheus series (would need explicit aggregation)")
        for series in series_list:
            for moment, value in series.get("values", []):
                t = float(moment)
                require(float(export["export_start_unix_s"]) - 1 <= t <=
                        float(export["export_end_unix_s"]) + 1,
                        f"{label}/{name}: sample outside exported range")
                values.append([t, finite_float(value)])
        require(values, f"{label}/{name}: no recorded samples")
        require(all(values[i][0] < values[i + 1][0] for i in range(len(values) - 1)),
                f"{label}/{name}: unordered samples")
        timeline[name] = {
            "promql": query["promql"],
            "samples": values,
        }
    r = summary["results"][0]
    require(math.isclose(
        float(r["mean_aggregate_tokens_per_sec"]),
        statistics.mean(b["throughput_tok_s"] for b in items), rel_tol=1e-8
    ), f"{label}: throughput summary not supported by batches")
    return {
        "classification": "RECORDED_MEASUREMENT_NOT_LIVE",
        "label": label,
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "hardware": "NVIDIA Tesla T4",
        "vllm_version": summary.get("vllm_version_declared", "unknown"),
        "prefix_caching": summary.get("prefix_caching"),
        "concurrent_http_requests": 64,
        "output_tokens_per_request": 128,
        "warmup_batches": 1, "measured_batches": 5,
        "benchmark_start_unix_s": start,
        "benchmark_end_unix_s": end,
        "export_start_unix_s": export["export_start_unix_s"],
        "export_end_unix_s": export["export_end_unix_s"],
        "prometheus_step_seconds": export["step_seconds"],
        "summary": {
            "mean_aggregate_tokens_per_sec": r["mean_aggregate_tokens_per_sec"],
            "mean_request_latency_sec": r["mean_request_latency_sec"],
            "p95_request_latency_sec": r["p95_request_latency_sec"],
            "p95_ttft_ms": r["p95_ttft_sec"] * 1000,
            "p95_tpot_ms": r["p95_tpot_sec"] * 1000,
            "p95_decode_phase_client_proxy_ms": percentile(
                [decode_phase_client_proxy_ms(request) for request in raw], 95),
            "p95_itl_ms": r["p95_itl_sec"] * 1000,
            "itl_eligible_requests": r["itl_eligible_requests"],
            "measured_requests": r["measured_requests"],
        },
        "batches": items,
        "metrics": timeline,
        "reference": "results_v1_1/" + label,
    }


def insert_navigation():
    """Keep the existing V1 landing page intact, adding only a V1.1 link."""
    page = ROOT / "docs" / "index.html"
    require(page.is_file(), "docs/index.html missing")
    original = page.read_text(encoding="utf-8")
    marker = '<h1>Prompt2GPU V1.0</h1>'
    nav = ('<p class="note"><strong>New:</strong> '
           '<a href="v1_1.html">V1.1 recorded inference observability replay</a> '
           '(initial run vs warm-server repeat; real measurements, not live telemetry).</p>')
    if nav not in original:
        require(original.count(marker) == 1, "V1 index anchor missing or ambiguous")
        page.write_text(original.replace(marker, marker + "\n" + nav, 1), encoding="utf-8")
    # The V1 build script regenerates index.html; preserve navigation on rebuild.
    v1_builder = ROOT / "scripts" / "build_dashboard.py"
    existing = v1_builder.read_text(encoding="utf-8")
    script_marker = '<h1>Prompt2GPU V1.0</h1>'
    if nav not in existing:
        require(existing.count(script_marker) == 1, "V1 builder anchor missing or ambiguous")
        v1_builder.write_text(existing.replace(script_marker, script_marker + "\n" + nav, 1), encoding="utf-8")
    # Require future make audit runs to verify dashboard provenance, not only V1.
    makefile_path = ROOT / "Makefile"
    makefile = makefile_path.read_text(encoding="utf-8")
    audit_anchor = "audit:\n\t$(PYTHON3) scripts/audit_repo.py"
    audit_command = "\n\t$(PYTHON3) scripts/build_v1_1_dashboard.py --check"
    if audit_command not in makefile:
        require(makefile.count(audit_anchor) == 1, "Makefile audit target unknown")
        makefile_path.write_text(makefile.replace(audit_anchor,
            audit_anchor + audit_command, 1), encoding="utf-8")
    readme_path = ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    link = '[V1.1 interactive historical dashboard](docs/v1_1.html)'
    if link not in readme:
        anchor = '## V1.1 — Streaming observability and Triton JIT investigation'
        require(anchor in readme, "V1.1 section missing from README")
        readme_path.write_text(readme.replace(anchor,
            anchor + '\n\n' + link + ' — playback of **recorded** T4 metrics and client-side measurements; no live server or synthetic GPU data.\n', 1),
            encoding="utf-8")


def generated_payloads():
    return {
        name: (json.dumps(convert_run(name), indent=2, allow_nan=False) + "\n").encode("utf-8")
        for name in RUNS
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Compare generated dashboard evidence byte-for-byte to committed output")
    args = parser.parse_args()
    payloads = generated_payloads()
    for name, data in payloads.items():
        out = TARGET / f"{name}.json"
        if args.check:
            require(out.is_file(), f"Dashboard export missing: {out}")
            require(out.read_bytes() == data, f"Dashboard export stale or edited: {out}")
        else:
            TARGET.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
        print(f"PASS: {name} ({len(data)} bytes) grounded in measured CSV + Prometheus")
    if args.check:
        for path in (ROOT / "docs/v1_1.html", ROOT / "docs/v1_1_dashboard.js",
                     ROOT / "docs/v1_1_dashboard.css"):
            require(path.is_file() and path.stat().st_size, f"Missing dashboard source: {path}")
        page = (ROOT / "docs/index.html").read_text(encoding="utf-8")
        v1_script = (ROOT / "scripts/build_dashboard.py").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        require('href="v1_1.html"' in page and 'href="v1_1.html"' in v1_script,
                "V1 dashboard navigation missing")
        require("$(PYTHON3) scripts/build_v1_1_dashboard.py --check" in
                (ROOT / "Makefile").read_text(encoding="utf-8"),
                "make audit does not validate V1.1 dashboard")
        require('[V1.1 interactive historical dashboard](docs/v1_1.html)' in readme,
                "README V1.1 link missing")
        print("PASS: HTML/JS/CSS exist; V1 dashboard and README link to replay")
    else:
        insert_navigation()
        print("PASS: V1 index/regen builder linked; README linked; replay exports generated")
    print("PASS: V1.1 dashboard historical replay " + ("validated" if args.check else "built"))


if __name__ == "__main__":
    main()
