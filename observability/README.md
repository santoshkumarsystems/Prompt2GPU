# Prompt2GPU observability — Grafana dashboard (V1.1)

The nine-panel Grafana dashboard is a reproducible export of the dashboard
built in Google Colab with vLLM 0.31.0, Prometheus, and Grafana OSS 13.2.2.

`observability/grafana/vllm-performance.json` is version-controlled. The
Prometheus TSDB, Grafana SQLite database, logs, and runtime PIDs are NOT.

## Build dashboard JSON on WSL or another computer (no GPU, install, or venv)

```bash
python3 scripts/setup_grafana_dashboard.py
```

This regenerates the JSON deterministically; running it twice should have
no additional effect. The script uses only the Python standard library.

## Deploy to an existing Colab Grafana

If the repository is cloned or its files copied into Colab, execute from the
repository root in a Colab cell:

```python
!python3 scripts/setup_grafana_dashboard.py --provision-colab
```

Prerequisites: the **existing** Grafana installation at
`/content/prompt2gpu-observability`, the Prometheus datasource provisioned with
UID `prompt2gpu-prometheus`, and a running vLLM endpoint scraped by Prometheus.
The script installs no dependencies, does not restart vLLM or Prometheus,
and restarts Grafana *only if its dashboard provisioning provider changed*.
It binds Grafana to localhost and enables anonymous **Viewer** for this private
Colab experiment. Do not use these settings for a public Grafana instance.

The file-based provider polls for future dashboard JSON updates every 10 sec.
Open Grafana at `/d/prompt2gpu-vllm` (or browse Dashboards > Prompt2GPU).

## Panels and interpretation

- Output tokens/sec (counter derivative over 1 min)
- P95 TTFT (server-side histogram over 5 min)
- P95 TPOT (server-side per-output-token duration histogram over 5 min)
- GPU KV cache usage (0–100%)
- Running and waiting requests
- Prefix cache hit rate (hit/query token counter rates)
- P95 queue time (server-side histogram over 5 min)
- P95 prefill time (server-side histogram over 5 min)

All panels use Prometheus metrics exported by vLLM. P95 histograms and rates
can display **No data** when idle. Grafana/Prometheus measurements should not
be conflated with client-measured TTFT/TPOT from benchmark streaming responses.

Versioned experiment details (GPU, model, vLLM version, server arguments,
prefix caching, concurrency, raw data) must accompany benchmark results; this
dashboard alone does not substantiate a performance improvement claim.

## Source control

Check in `scripts/setup_grafana_dashboard.py`,
`observability/grafana/vllm-performance.json`, and this README. Do not commit
Colab runtime files, credentials, telemetry databases, logs, or generated
archives. Retain the published V1 result files unchanged.
