# Author: Santosh Kumar
# Project: Prompt2GPU
# Version: 1.0

"""Shared settings used by every Prompt2GPU benchmark.

BEGINNER ROADMAP
----------------
This file does not run a benchmark by itself.  It stores constants such as the
model name, prompt, output-token count, result directories, and vLLM URLs so
all experiments use the same values.

Python ideas used here:
* ``Path`` builds filesystem paths without hard-coding Linux/Windows separators.
* Upper-case names such as ``MODEL`` are constants by convention.
* ``os.environ.get`` lets an advanced user override the vLLM host without
  editing source code.

If you are new to Python, read ``docs/PYTHON_WALKTHROUGH.md`` first.
"""

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

PROMPT = """
Explain Direct Memory Access (DMA), how it works, and why it improves
system performance. Give a technical explanation with enough detail
to continue for at least 128 generated tokens.
""".strip()

DTYPE = "float16"
OUTPUT_TOKENS = 128
WARMUP_RUNS = 1
MEASURED_RUNS = 10

VLLM_HOST = os.environ.get("PROMPT2GPU_VLLM_HOST", "http://localhost:8000")
VLLM_URL = f"{VLLM_HOST}/v1/completions"
VLLM_HEALTH_URL = f"{VLLM_HOST}/health"
VLLM_METRICS_URL = f"{VLLM_HOST}/metrics"
