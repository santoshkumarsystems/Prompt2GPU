# Prompt2GPU V1.0 runbook

**Author:** Santosh Kumar
**Version:** 1.0

The published V1.0 measurements were collected on **Google Colab with an NVIDIA Tesla T4**. Results on other GPUs are expected to vary.

All commands below are run from the **Prompt2GPU repository root** unless a command explicitly says otherwise. No manual virtual-environment activation step is required.

## 0. Decide what you are trying to do

If you only want to view published V1.0 results, stop here: open `docs/index.html`. No installation is required.

If you want to verify the published evidence:

```bash
make verify
```

If you want to rebuild the dashboard:

```bash
make setup-dashboard
make dashboard
```

**Expected setup time:** typically ~1–5 minutes. No NVIDIA GPU is required.

## 1. GPU reproduction preflight — mandatory

Do **not** start GPU dependency installation directly.

```bash
make preflight-gpu
```

If no NVIDIA GPU/driver is detected, Prompt2GPU stops the GPU path and tells you how to view/verify/rebuild the published results without CUDA.

## 2. Optional heavy GPU setup

Only after preflight passes:

```bash
make setup-gpu
```

Before downloading anything heavy, this command shows a warning and requires explicit `YES` confirmation.

Typical impact varies by machine/cache/network:

- **Time:** ~10–30 minutes on fast internet; 30–60+ minutes on slower Wi-Fi
- **Download:** several GB
- **Disk:** several additional GB
- **Packages:** PyTorch/CUDA runtime libraries, cuDNN, NCCL, vLLM, Transformers, Accelerate, and dependencies

No manual virtual-environment activation is required. GPU commands use `.venv-gpu/bin/python3` directly.

## 3. Verify a clean GPU

```bash
nvidia-smi
```

For benchmark isolation, avoid leaving a model loaded in an interactive notebook/kernel process.

## 4. Run the HF/PyTorch baseline

```bash
make hf-baseline
```

Outputs:

- `results/hf_baseline_raw.csv`
- `results/hf_baseline_summary.json`

## 5. Verify GPU memory was released

```bash
nvidia-smi
```

If another process still owns several GiB, stop that process or restart the runtime before starting vLLM.

## 6. Start vLLM

```bash
make start-vllm
```

The launcher writes `logs/vllm_server.log`, records its PID, and waits for `/health` to return HTTP 200. Prefix caching is disabled for the published concurrency/KV experiments.

## 7. Optional metrics inspection

```bash
.venv-gpu/bin/python3 src/inspect_vllm_metrics.py
```

This is an advanced diagnostic command; normal reproduction does not require it.

## 8. Run the vLLM baseline

```bash
make vllm-baseline
```

## 9. Run concurrency scaling

```bash
make concurrency
```

Published V1.0: C=1,2,4,8,16,32,64 with one warm-up and five measured batches per level.

## 10. Run context + KV-cache pressure

```bash
make kv-pressure
```

**This is the longest V1.0 benchmark.** Published V1.0 used concurrency 80, context lengths 256→3800, one warm-up plus three measured batches per level, and 100 ms metrics sampling. Runtime varies substantially by GPU.

## 11. Stop vLLM

```bash
make stop-vllm
```

## 12. Verify and rebuild presentation

```bash
make verify
make dashboard
```

The committed `docs/index.html` already contains the published Colab/Tesla T4 results. Rebuilding is only necessary after intentionally changing result artifacts.

## 13. Pre-push audit

```bash
make audit
```

Do not push publicly until this passes and the repository has also received a human review.
