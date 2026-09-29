#!/usr/bin/env python3
# Author: Santosh Kumar
# Project: Prompt2GPU
# Version: 1.0

"""Generate README plots and the self-contained HTML dashboard.

DATA-INTEGRITY RULE
-------------------
This script reads the committed Prompt2GPU result JSON/CSV files.  Benchmark
numbers are not hard-coded into chart creation.  Re-running the script after a
new benchmark regenerates the presentation from the new data.

The static PNG files are written to ``plots/`` for GitHub README display.  The
interactive Plotly dashboard is written to ``docs/index.html`` for local use or
GitHub Pages.

If you are new to Python, ``docs/PYTHON_WALKTHROUGH.md`` explains JSON loading,
functions, lists, dictionaries, f-strings, and the plotting workflow.
"""

from pathlib import Path
import json
import shutil

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.io import to_html

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
DOCS = ROOT / "docs"
DATA = DOCS / "data"

PLOTS.mkdir(exist_ok=True)
DOCS.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)


def load_json(name):
    """Load one JSON result file from ``results/`` and convert it into Python dictionaries/lists."""
    return json.loads((RESULTS / name).read_text())


def save_static_plots(hf, vllm, concurrency, kv):
    """Create the PNG charts embedded directly in the GitHub README."""
    engines = [hf["engine"], vllm["engine"]]
    throughputs = [hf["output_tokens_per_sec"]["mean"], vllm["output_tokens_per_sec"]["mean"]]
    latencies = [hf["latency_sec"]["mean"], vllm["latency_sec"]["mean"]]

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    bars = ax.bar(engines, throughputs)
    ax.set_ylabel("Output tokens / second")
    ax.set_title("Single-request throughput")
    ax.bar_label(bars, fmt="%.2f")
    fig.tight_layout()
    fig.savefig(PLOTS / "baseline_throughput.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    bars = ax.bar(engines, latencies)
    ax.set_ylabel("Mean end-to-end latency (seconds)")
    ax.set_title("Single-request latency")
    ax.bar_label(bars, fmt="%.3f")
    fig.tight_layout()
    fig.savefig(PLOTS / "baseline_latency.png", dpi=180)
    plt.close(fig)

    c = [r["concurrency"] for r in concurrency["results"]]
    t = [r["mean_aggregate_tokens_per_sec"] for r in concurrency["results"]]
    mean_lat = [r["mean_request_latency_sec"] for r in concurrency["results"]]
    p95_lat = [r["p95_request_latency_sec"] for r in concurrency["results"]]

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    ax.plot(c, t, marker="o")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Aggregate output tokens / second")
    ax.set_title("vLLM concurrency scaling")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS / "concurrency_throughput.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    ax.plot(c, mean_lat, marker="o", label="Mean latency")
    ax.plot(c, p95_lat, marker="o", label="P95 latency")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Seconds")
    ax.set_title("vLLM request latency under concurrency")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS / "concurrency_latency.png", dpi=180)
    plt.close(fig)

    x = [r["actual_input_tokens"] for r in kv["results"]]
    kv_usage = [100.0 * r["max_peak_kv_cache_usage"] for r in kv["results"]]
    kv_throughput = [r["mean_aggregate_tokens_per_sec"] for r in kv["results"]]
    queue = [r["mean_queue_time_sec"] for r in kv["results"]]

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    ax.plot(x, kv_usage, marker="o")
    ax.set_xlabel("Input tokens / request")
    ax.set_ylabel("Peak KV-cache usage (%)")
    ax.set_title("Context length vs peak KV-cache usage (concurrency 80)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS / "kv_cache_usage.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    ax.plot(x, kv_throughput, marker="o")
    ax.set_xlabel("Input tokens / request")
    ax.set_ylabel("Aggregate output tokens / second")
    ax.set_title("Context pressure vs serving throughput (concurrency 80)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS / "kv_pressure_throughput.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    ax.plot(x, queue, marker="o")
    ax.set_xlabel("Input tokens / request")
    ax.set_ylabel("Mean queue time (seconds)")
    ax.set_title("Context pressure vs queue time (concurrency 80)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS / "kv_queue_time.png", dpi=180)
    plt.close(fig)


def plotly_figures(hf, vllm, concurrency, kv):
    """Create the interactive Plotly chart objects used by the HTML dashboard."""
    figures = []

    fig = go.Figure(go.Bar(
        x=[hf["engine"], vllm["engine"]],
        y=[hf["output_tokens_per_sec"]["mean"], vllm["output_tokens_per_sec"]["mean"]],
        text=[f'{hf["output_tokens_per_sec"]["mean"]:.2f}', f'{vllm["output_tokens_per_sec"]["mean"]:.2f}'],
        textposition="outside",
    ))
    fig.update_layout(title="Single-request throughput", yaxis_title="Output tokens / second")
    figures.append(fig)

    fig = go.Figure(go.Bar(
        x=[hf["engine"], vllm["engine"]],
        y=[hf["latency_sec"]["mean"], vllm["latency_sec"]["mean"]],
        text=[f'{hf["latency_sec"]["mean"]:.3f}s', f'{vllm["latency_sec"]["mean"]:.3f}s'],
        textposition="outside",
    ))
    fig.update_layout(title="Single-request latency", yaxis_title="Seconds")
    figures.append(fig)

    c = [r["concurrency"] for r in concurrency["results"]]
    t = [r["mean_aggregate_tokens_per_sec"] for r in concurrency["results"]]
    fig = go.Figure(go.Scatter(x=c, y=t, mode="lines+markers", name="Aggregate throughput"))
    fig.update_layout(title="Concurrency scaling", xaxis_title="Concurrency", yaxis_title="Aggregate output tokens / second")
    figures.append(fig)

    mean_lat = [r["mean_request_latency_sec"] for r in concurrency["results"]]
    p95_lat = [r["p95_request_latency_sec"] for r in concurrency["results"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c, y=mean_lat, mode="lines+markers", name="Mean"))
    fig.add_trace(go.Scatter(x=c, y=p95_lat, mode="lines+markers", name="P95"))
    fig.update_layout(title="Latency under concurrency", xaxis_title="Concurrency", yaxis_title="Seconds")
    figures.append(fig)

    x = [r["actual_input_tokens"] for r in kv["results"]]
    kv_usage = [100.0 * r["max_peak_kv_cache_usage"] for r in kv["results"]]
    throughput = [r["mean_aggregate_tokens_per_sec"] for r in kv["results"]]
    queue = [r["mean_queue_time_sec"] for r in kv["results"]]

    fig = go.Figure(go.Scatter(x=x, y=kv_usage, mode="lines+markers", name="Peak KV usage"))
    fig.update_layout(title="Context length vs KV-cache usage (C=80)", xaxis_title="Input tokens / request", yaxis_title="Peak KV-cache usage (%)")
    figures.append(fig)

    fig = go.Figure(go.Scatter(x=x, y=throughput, mode="lines+markers", name="Throughput"))
    fig.update_layout(title="Context pressure vs throughput (C=80)", xaxis_title="Input tokens / request", yaxis_title="Aggregate output tokens / second")
    figures.append(fig)

    fig = go.Figure(go.Scatter(x=x, y=queue, mode="lines+markers", name="Mean queue time"))
    fig.update_layout(title="Context pressure vs queue time (C=80)", xaxis_title="Input tokens / request", yaxis_title="Mean queue time (seconds)")
    figures.append(fig)

    for f in figures:
        f.update_layout(template="plotly_white", margin=dict(l=55, r=25, t=65, b=55), height=440)
    return figures


def build_html(hf, vllm, concurrency, kv):
    """Combine measured KPIs, tables, and Plotly charts into one self-contained HTML page."""
    speedup = vllm["output_tokens_per_sec"]["mean"] / hf["output_tokens_per_sec"]["mean"]
    latency_reduction = 100.0 * (1.0 - vllm["latency_sec"]["mean"] / hf["latency_sec"]["mean"])
    c1, c64 = concurrency["results"][0], concurrency["results"][-1]
    scale = c64["mean_aggregate_tokens_per_sec"] / c1["mean_aggregate_tokens_per_sec"]
    kv_last = kv["results"][-1]

    figs = plotly_figures(hf, vllm, concurrency, kv)
    fragments = []
    for i, fig in enumerate(figs):
        fragments.append(to_html(
            fig,
            full_html=False,
            include_plotlyjs="inline" if i == 0 else False,
            config={"displaylogo": False, "responsive": True},
        ))

    conc_rows = "".join(
        f"<tr><td>{r['concurrency']}</td><td>{r['mean_aggregate_tokens_per_sec']:.2f}</td><td>{r['mean_request_latency_sec']:.3f}</td><td>{r['p95_request_latency_sec']:.3f}</td></tr>"
        for r in concurrency["results"]
    )
    kv_rows = "".join(
        f"<tr><td>{r['actual_input_tokens']}</td><td>{r['mean_aggregate_tokens_per_sec']:.2f}</td><td>{r['mean_request_latency_sec']:.3f}</td><td>{r['p95_request_latency_sec']:.3f}</td><td>{100*r['max_peak_kv_cache_usage']:.1f}%</td><td>{r['max_requests_waiting']:.0f}</td><td>{r['mean_queue_time_sec']:.3f}</td></tr>"
        for r in kv["results"]
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:,">
<title>Prompt2GPU V1.0 Dashboard</title>
<style>
:root {{ --bg:#0b1020; --card:#141b2d; --text:#edf2f7; --muted:#aab4c6; --accent:#72a7ff; --border:#263149; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--text); }}
.wrap {{ max-width:1180px; margin:0 auto; padding:32px 20px 64px; }}
h1 {{ margin:0 0 8px; font-size:2.35rem; }}
.lead {{ color:var(--muted); max-width:850px; line-height:1.55; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; margin:26px 0; }}
.card {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:18px; }}
.kpi {{ font-size:1.8rem; font-weight:750; margin-top:6px; }}
.label {{ color:var(--muted); font-size:.9rem; }}
.section {{ margin-top:28px; }}
.chart {{ background:white; border-radius:14px; overflow:hidden; margin:16px 0; }}
table {{ width:100%; border-collapse:collapse; background:var(--card); border-radius:12px; overflow:hidden; }}
th,td {{ padding:11px 12px; border-bottom:1px solid var(--border); text-align:right; }}
th:first-child,td:first-child {{ text-align:left; }}
th {{ color:#d6deec; }}
p,li {{ line-height:1.55; }}
.note {{ color:var(--muted); background:var(--card); border-left:4px solid var(--accent); padding:14px 16px; border-radius:8px; }}
a {{ color:#9cc2ff; }}
.footer {{ margin-top:42px; color:var(--muted); font-size:.9rem; }}
</style>
</head>
<body><div class="wrap">
<h1>Prompt2GPU V1.0</h1>
<p class="lead">A single-GPU LLM inference systems lab measuring a Hugging Face/PyTorch baseline against vLLM, then exploring concurrency scaling and context/KV-cache pressure on a Tesla T4 using Qwen2.5-1.5B-Instruct.</p>
<p class="lead"><strong>Published measurements:</strong> Google Colab · NVIDIA Tesla T4 · Qwen/Qwen2.5-1.5B-Instruct · FP16 · Prompt2GPU V1.0</p>
<div class="cards">
  <div class="card"><div class="label">Single-request throughput gain</div><div class="kpi">{speedup:.2f}×</div></div>
  <div class="card"><div class="label">Mean latency reduction</div><div class="kpi">{latency_reduction:.1f}%</div></div>
  <div class="card"><div class="label">C=64 aggregate throughput</div><div class="kpi">{c64['mean_aggregate_tokens_per_sec']:.0f} tok/s</div></div>
  <div class="card"><div class="label">C=1 → C=64 throughput scale</div><div class="kpi">{scale:.2f}×</div></div>
  <div class="card"><div class="label">Peak KV usage @ 3800 tokens</div><div class="kpi">{100*kv_last['max_peak_kv_cache_usage']:.1f}%</div></div>
</div>

<div class="section"><h2>1. Controlled baseline</h2>
<p>Same model, GPU, FP16 precision, 69 input tokens, 128 forced output tokens, one warm-up, and ten measured runs. The vLLM number includes HTTP request/response time; the HF/PyTorch path times in-process generation.</p>
<div class="chart">{fragments[0]}</div><div class="chart">{fragments[1]}</div></div>

<div class="section"><h2>2. Concurrency scaling</h2>
<p>Prefix caching was disabled. Each concurrency level used one warm-up batch and five measured batches.</p>
<div class="chart">{fragments[2]}</div><div class="chart">{fragments[3]}</div>
<table><thead><tr><th>Concurrency</th><th>Mean aggregate tok/s</th><th>Mean latency (s)</th><th>P95 (s)</th></tr></thead><tbody>{conc_rows}</tbody></table></div>

<div class="section"><h2>3. Context + KV-cache pressure</h2>
<p>At concurrency 80, increasing input context raises both prefill work and KV-cache demand. These measurements therefore represent combined context/scheduler/memory pressure, not a KV-only causal experiment.</p>
<div class="chart">{fragments[4]}</div><div class="chart">{fragments[5]}</div><div class="chart">{fragments[6]}</div>
<table><thead><tr><th>Input tokens</th><th>Throughput tok/s</th><th>Mean latency (s)</th><th>P95 (s)</th><th>Peak KV</th><th>Peak waiting</th><th>Mean queue (s)</th></tr></thead><tbody>{kv_rows}</tbody></table></div>

<div class="section"><h2>Evidence and reproducibility</h2>
<p class="note">Every number on this page is generated from the committed result JSON/CSV files. Run <code>python3 scripts/verify_results.py</code> to recompute summary values from raw measurements.</p>
<ul>
<li><a href="data/hf_baseline_raw.csv">HF raw CSV</a> · <a href="data/hf_baseline_summary.json">HF summary JSON</a></li>
<li><a href="data/vllm_baseline_raw.csv">vLLM raw CSV</a> · <a href="data/vllm_baseline_summary.json">vLLM summary JSON</a></li>
<li><a href="data/vllm_concurrency_raw.csv">Concurrency request data</a> · <a href="data/vllm_concurrency_batches.csv">batch data</a> · <a href="data/vllm_concurrency_summary.json">summary</a></li>
<li><a href="data/vllm_kv_pressure_requests.csv">KV-pressure request data</a> · <a href="data/vllm_kv_pressure_batches.csv">batch data</a> · <a href="data/vllm_kv_pressure_metrics.csv">metrics time series</a> · <a href="data/vllm_kv_pressure_summary.json">summary</a></li>
</ul></div>
<div class="footer">Author: Santosh Kumar · Prompt2GPU V1.0 · Generated by <code>scripts/build_dashboard.py</code> from committed Google Colab / Tesla T4 benchmark artifacts.</div>
</div></body></html>"""
    (DOCS / "index.html").write_text(html)


def copy_data():
    """Copy source result artifacts into ``docs/data`` so dashboard visitors can inspect/download them."""
    for path in RESULTS.iterdir():
        if path.is_file() and path.suffix in {".csv", ".json", ".txt"}:
            shutil.copy2(path, DATA / path.name)


def main():
    """Load measurements, rebuild all charts/dashboard files, and copy the supporting data."""
    hf = load_json("hf_baseline_summary.json")
    vllm = load_json("vllm_baseline_summary.json")
    concurrency = load_json("vllm_concurrency_summary.json")
    kv = load_json("vllm_kv_pressure_summary.json")
    save_static_plots(hf, vllm, concurrency, kv)
    copy_data()
    build_html(hf, vllm, concurrency, kv)
    print(f"Static plots: {PLOTS}")
    print(f"Dashboard: {DOCS / 'index.html'}")


# BEGINNER NOTE: Python sets ``__name__`` to ``"__main__"`` when this
# file is run directly.  When another file imports it, this block is skipped.
# That lets us reuse functions without accidentally starting the whole program.
if __name__ == "__main__":
    main()
