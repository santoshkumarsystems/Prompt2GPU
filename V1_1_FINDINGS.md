# Prompt2GPU V1.1 — Measured streaming C64 and JIT investigation

**Measured environment:** Google Colab NVIDIA Tesla T4, Qwen/Qwen2.5-1.5B-Instruct, FP16, vLLM 0.31.0; 69 input tokens and 128 forced completion tokens per request. Five measured batches × 64 concurrent **HTTP requests**, following one warm-up batch; prefix caching enabled in the running server. Distinguish 64 requests submitted concurrently from 64 GPU-running sequences at every instant.

**Real measurements:** From the two dated runs, reproduced and checked by `scripts/publish_v1_1.py` against their raw request, batch and ITL interval CSV files. Full unredacted engine logs are held privately and **not published**.

| Metric | Initial run | Warm-server repeat |
|---|---:|---:|
| Mean aggregate output throughput | 1917.87 tokens/s | 1994.69 tokens/s |
| Mean request end-to-end latency | 4.232 s | 3.962 s |
| P95 end-to-end latency | 5.574 s | 4.049 s |
| P95 client-observed TTFT | 1937.6 ms | 303.9 ms |
| P95 client-estimated TPOT | 28.82 ms | 29.83 ms |
| ITL-eligible requests | 316/320 | 320/320 |

## Anomalous initial Batch 2

All five measured batches generated exactly 8,192 output tokens. Initial Batch 2 delivered **1438.34 tokens/s**, compared with **2037.75 tokens/s** mean for the other four initial measured batches. Mean client-observed TTFT rose from **350.2 ms** in Batch 1 to **1894.5 ms** in Batch 2, while TPOT remained comparatively stable. In the private engine log, a warning at 2026-10-13 05:02:05 reported `Triton kernel JIT compilation during inference: kernel_unified_attention`. That timestamp falls within measured Batch 2 (05:02:03.634–05:02:09.330 UTC). One such warning was recorded during the initial run; none was newly recorded during the warm-server repeat.

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
