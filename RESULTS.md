# Prompt2GPU V1.0 measured results

**Author:** Santosh Kumar
**Version:** 1.0
**Measurement environment:** Google Colab · NVIDIA Tesla T4 · Qwen/Qwen2.5-1.5B-Instruct · FP16

This document summarizes what Prompt2GPU V1.0 **observed**. It does not claim that another GPU will produce identical values. Raw CSVs, summary JSON, checksums, plots, and the interactive dashboard are committed in the repository.

## Observation 1 — single-request serving

Controlled HF/PyTorch mean output throughput was **21.41 tok/s** and mean measured latency was **6.007 s**. Controlled vLLM mean output throughput was **66.71 tok/s** and mean measured latency was **1.919 s**.

For this Tesla T4 setup, the measured vLLM output-throughput ratio was about **3.12×**.

The two paths are not identical software interfaces: vLLM timing includes HTTP request/response overhead while the HF/PyTorch path times in-process generation. That distinction is retained rather than hidden.

## Observation 2 — concurrency scaling

Aggregate vLLM output throughput increased from **66.63 tok/s at C=1** to **1645.93 tok/s at C=64**. Mean latency increased from **1.921 s** to **4.856 s**, with C=64 P95 at **5.031 s**.

V1.0 did not observe a hard throughput plateau by C=64. Scaling became progressively sublinear, which is reported as diminishing parallel efficiency rather than incorrectly labeling a saturation point.

## Observation 3 — context + KV-cache pressure

At concurrency 80, peak reported KV-cache usage increased from **10.3% at 256 input tokens** to **89.3% at 3800 input tokens**. Over the same sweep, aggregate output throughput declined from **1076.03 tok/s** to **42.19 tok/s**, mean latency increased from **9.270 s** to **233.130 s**, and mean queue time increased from **1.063 s** to **106.438 s**.

This is not presented as a KV-only causal experiment. Increasing context length simultaneously increases prefill compute, scheduler pressure, and KV-cache demand. Waiting requests were observed even when KV utilization was low, demonstrating that queueing is not explained by KV capacity alone.

## Published evidence

- `results/hf_baseline_raw.csv`
- `results/hf_baseline_summary.json`
- `results/vllm_baseline_raw.csv`
- `results/vllm_baseline_summary.json`
- `results/vllm_concurrency_raw.csv`
- `results/vllm_concurrency_batches.csv`
- `results/vllm_concurrency_summary.json`
- `results/vllm_kv_pressure_requests.csv`
- `results/vllm_kv_pressure_batches.csv`
- `results/vllm_kv_pressure_metrics.csv`
- `results/vllm_kv_pressure_summary.json`

Run `make verify` to recompute published summary values from the committed raw CSV evidence.
