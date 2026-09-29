# Author: Santosh Kumar
# Project: Prompt2GPU
# Version: 1.0

"""Print vLLM metrics related to requests, queueing, and KV cache.

The vLLM server exposes Prometheus-formatted text at ``/metrics``.  This small
utility downloads that text and prints only lines whose names are useful for
Prompt2GPU.  It is a discovery/debugging tool; it does not modify the server.

If you are new to Python, read ``docs/PYTHON_WALKTHROUGH.md`` first.
"""

import requests
from benchmark_config import VLLM_METRICS_URL

KEYWORDS = [
    "kv_cache",
    "cache_usage",
    "num_requests",
    "request_queue",
    "waiting",
    "running",
]


def main():
    """Fetch ``/metrics`` from vLLM and print only scheduling/KV-related metric lines."""
    print("=== Prompt2GPU : vLLM Metrics Inspection ===")
    response = requests.get(VLLM_METRICS_URL, timeout=5)
    response.raise_for_status()
    matches = [
        line
        for line in response.text.splitlines()
        if any(keyword in line.lower() for keyword in KEYWORDS)
    ]
    print("\n".join(matches) if matches else "No matching KV-cache or request metrics found.")


# BEGINNER NOTE: Python sets ``__name__`` to ``"__main__"`` when this
# file is run directly.  When another file imports it, this block is skipped.
# That lets us reuse functions without accidentally starting the whole program.
if __name__ == "__main__":
    main()
