# Methodology and limitations

**Author:** Santosh Kumar
**Version:** 1.0

## Design principles

Prompt2GPU V1.0 uses controlled work, saved raw measurements, repeated runs, and explicit caveats. The project avoids manually transcribed benchmark values: summaries, README plots, and the HTML dashboard are built from committed CSV/JSON files.

## Experiment 1 — controlled baseline

Both paths use Qwen/Qwen2.5-1.5B-Instruct on the same Tesla T4 with FP16, the same chat-formatted prompt (69 input tokens), and exactly 128 generated tokens. One warm-up run is excluded; ten runs are measured.

HF/PyTorch uses `model.generate()` and CUDA synchronization around the timed region. vLLM is exercised through its OpenAI-compatible HTTP completion endpoint and therefore includes local HTTP request/response overhead.

**Limitation:** this is a useful practical comparison of an in-process library baseline versus a serving engine, not a perfectly API-equivalent implementation comparison. The project reports the measured setup rather than attributing the full difference to a single internal optimization.

## Experiment 2 — concurrency scaling

Concurrency levels: 1, 2, 4, 8, 16, 32, 64. Each level uses one warm-up batch plus five measured batches. Every request generates exactly 128 output tokens. Aggregate throughput is computed as total generated output tokens divided by batch wall-clock time. Request-level mean and P95 latencies are also recorded.

Automatic prefix caching is disabled because requests intentionally share a controlled prompt; allowing prefix reuse would confound the experiment.

**Interpretation:** throughput still rises at C=64, so V1.0 does not claim a hard saturation point. The important observation is diminishing scaling efficiency accompanied by higher latency.

## Experiment 3 — context + KV-cache pressure

Concurrency is fixed at 80. Input lengths are 256, 512, 1024, 2048, 3072, and 3800 tokens; generation remains fixed at 128 output tokens. Each level uses one warm-up batch plus three measured batches.

A background sampler reads vLLM's Prometheus endpoint every 100 ms and records KV-cache usage, running requests, waiting requests, and queue-time deltas. The server reported a 297,472-token GPU KV cache with 16-token blocks and prefix caching disabled.

**Important limitation:** increasing context length simultaneously increases prefill compute and KV-cache demand. The resulting throughput/latency change cannot be attributed solely to KV-cache utilization. Likewise, queueing is influenced by scheduler capacity and compute/token budgets, not just memory capacity.

## Metrics not claimed in V1.0

- TTFT (time to first token) is not measured because the baseline scripts use non-streaming completion responses.
- HF PyTorch peak allocator memory and vLLM process/device memory are not presented as directly comparable memory metrics.
- The project does not claim cross-GPU generalization from a single Tesla T4 run.
- The project does not claim a universal vLLM speedup; results are specific to this model, hardware, software stack, and workload.
- The 100 ms Prometheus sampler may miss extremely short-lived instantaneous peaks.

## Evidence chain

```text
GPU execution
   ↓
request-level / batch-level CSV
   ↓
summary JSON
   ↓
verify_results.py
   ↓
plots + docs/index.html
```

`results/SHA256SUMS.txt` records hashes for the published measurement artifacts.
