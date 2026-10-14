# Prompt2GPU V1.1 — Concurrency 64 + streaming latency

**Status:** The V1.1 streaming source was exercised on a real Tesla T4 in October 2026; two measured C64 runs were preserved. See [`V1_1_FINDINGS.md`](../V1_1_FINDINGS.md) and `results_v1_1/` after the focused release is published. Offline/mock unit tests are separate from those measurements.

## Why a separate V1.1 script?

V1 (`src/vllm_concurrency.py`) established reproducible throughput and end-to-end request latency. This source is a **copy of V1**, preserving the model, exact DMA prompt (via `benchmark_config.py`), 128-output-token contract, `ignore_eos=True`, concurrent HTTP client model, one warm-up batch, five measured batches, and original throughput and latency calculations. **The V1 source and published `results/` are unchanged.**

V1.1 adds client-side streaming time-to-first-token (TTFT), time-per-output-token (TPOT), conditional SSE-based inter-token latency (ITL), P99 percentiles, and timestamped output for correlating the benchmark with Prometheus/Grafana. You can select `--mode nonstreaming` to retain the V1 request/response measurement mode or `--mode streaming` to collect the new latencies. These modes have different client instrumentation overhead and **must not be treated as identical benchmark configurations**.

## The experiment on the existing Colab vLLM server

The server should already be running on port 8000. **Do not install vLLM or start a second server.** Existing Colab environment path: `/content/prompt2gpu-venv/bin/python`.

After making the new `src/` script available inside the Colab repository, and with Google Drive mounted, run a Colab `%%bash` cell:

```bash
%%bash
set -euo pipefail
cd /content/Prompt2GPU
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
/content/prompt2gpu-venv/bin/python -u src/vllm_concurrency_v1_1.py \
  --server http://127.0.0.1:8000 \
  --levels 64 \
  --mode streaming \
  --warmup-batches 1 \
  --measured-batches 5 \
  --prefix-caching on \
  --vllm-version 0.31.0 \
  --output-dir "/content/drive/MyDrive/Prompt2GPU/v1_1_c64_${RUN_ID}"
```

**The directory `/content/Prompt2GPU` is an example checkout location, not created by this command.** Clone/copy the updated repository to your Colab session first and change the `cd` line to its actual location. To keep V1 published files safe, every run requires a fresh output path. Default mode and level are streaming and concurrency 64.

**Duration:** Running 384 requests total (64 warm-up + 5 × 64 measured, each capped at 128 output tokens) may take several minutes on a T4. No package installation is part of this command.

### IMPORTANT: prefix caching metadata

`--prefix-caching on` **does not enable cache on the server**. It records the actual setting you confirm from the current vLLM startup arguments/logs. If unsure, use `--prefix-caching unknown`. The existing Colab vLLM server was launched with vLLM 0.31.0 where prefix caching was reported enabled. Do not compare its throughput directly with the V1 `--no-enable-prefix-caching` baseline as an isolated caching improvement: vLLM version, GPU-memory target, and streaming instrumentation differ too.

## Files saved per run

- `vllm_concurrency_raw.csv` — original V1 raw request columns **plus** TTFT, TPOT, conditional ITL summaries, and request start Unix timestamps.
- `vllm_concurrency_batches.csv` — original V1 per-batch summary **plus** batch start/end Unix timestamps for correlation with metrics.
- `vllm_concurrency_itl.csv` — timestamped SSE inter-arrival intervals only when content-event/token count matches.
- `vllm_concurrency_summary.json` — original V1 aggregate-throughput statistics **plus** streaming percentile statistics, declared configuration, and metric definitions.

Prometheus metric exports and Grafana screenshots must be saved separately from Colab and attributed to the same run. This script **does not** automatically fetch or fabricate them.

## Measurement semantics and limitations

- **TTFT**: client request start to first nonempty streamed completion text event, including network/client/queue time.
- **TPOT**: `(timestamp of last nonempty content event − first nonempty content event) / (number of completion tokens − 1)`. One scalar *per request*; P95 TPOT is calculated across requests.
- **Conditional ITL estimate**: intervals between adjacent nonempty SSE completion content events. Computed **only** if the number of content events equals the server-reported completion token count. This matching-count check does **not** prove each event contains exactly one token; treat ITL as an SSE-based estimate unless one-token-per-event emission is independently confirmed. P95/P99 ITL are pooled over eligible intervals, not per-request TPOT percentiles. If the stream merges token events, ITL is omitted rather than fabricated.
- **Aggregate output tokens/sec**: total server-reported output tokens divided by *batch* wall time, using V1's exact formula. Streaming mode can change the result via different transport/CPU overhead.
- **P99**: linear-interpolated percentile, following V1's existing percentile implementation.
- **No token-level truth from Grafana screenshots**: retain CSV/JSON as the authoritative measurement record.
- **No claim of a controlled V1 vs V1.1 speedup** until version, cache, workload, and instrumentation differences are isolated.

## Local source checks (no GPU required)

```bash
python3 -m py_compile src/vllm_concurrency_v1_1.py
python3 -m unittest discover -s tests -p 'test_concurrency_v1_1.py' -v
make verify
make audit
```

The unit tests use a mocked local HTTP server response, never launch vLLM, and never report real throughput measurements.
