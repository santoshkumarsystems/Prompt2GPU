# Recorded V1.0 environment

**Author:** Santosh Kumar
**Version:** 1.0

These values come from the committed benchmark summaries and vLLM server log.

| Component | Recorded value |
|---|---|
| GPU | Tesla T4 |
| Model | Qwen/Qwen2.5-1.5B-Instruct |
| Precision | FP16 |
| Python | 3.13.15 |
| PyTorch | 2.13.0+cu132 |
| PyTorch CUDA runtime | 13.2 |
| vLLM | 0.30.0 |
| Max model length | 4096 |
| GPU memory utilization target | 0.80 |
| Prefix caching | disabled for published vLLM concurrency/KV runs |
| Attention backend on T4 | TRITON_ATTN (server log reports FA2 unavailable on compute capability 7.5) |
| Available KV cache memory | 7.94 GiB (server log) |
| GPU KV cache | 297,472 tokens |
| KV block size | 16 tokens |
| GPU KV blocks | 18,592 |

The Hugging Face Hub warning about unauthenticated requests affects download rate limits, not the measured inference calculations after the model is loaded.
