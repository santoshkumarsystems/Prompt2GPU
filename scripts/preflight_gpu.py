#!/usr/bin/env python3
# Author: Santosh Kumar
# Project: Prompt2GPU
# Version: 1.0

"""Check whether this machine is ready to attempt Prompt2GPU GPU setup.

This script intentionally uses only Python's standard library.  It runs before
any heavy PyTorch/CUDA/vLLM installation so users without an NVIDIA GPU do not
waste time, bandwidth, or disk space.

Exit status:
    0 -> at least one NVIDIA GPU was detected by ``nvidia-smi``.
    0 -> the preflight itself completed; output states whether GPU setup is available.

A passing preflight means "GPU detected", not "published Tesla T4 numbers will
match".  Performance varies by GPU, driver, thermal/power limits, and software
versions.
"""

from __future__ import annotations

import shutil
import subprocess
import sys


def detect_nvidia_gpus() -> list[dict[str, str]]:
    """Return NVIDIA GPU information reported by ``nvidia-smi``.

    An empty list means that ``nvidia-smi`` is unavailable, failed, or reported
    no GPUs.  No third-party Python packages are required for this check.
    """
    if shutil.which("nvidia-smi") is None:
        return []

    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.SubprocessError, OSError):
        return []

    gpus: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            gpus.append(
                {
                    "name": parts[0],
                    "memory_mib": parts[1],
                    "driver": parts[2],
                }
            )
    return gpus


def main() -> int:
    """Print a human-readable preflight result and return an appropriate status."""
    print("=== Prompt2GPU GPU Preflight ===")
    print(f"Python: {sys.version.split()[0]}")

    gpus = detect_nvidia_gpus()
    if not gpus:
        print()
        print("STOP: No NVIDIA GPU/driver was detected by nvidia-smi.")
        print("GPU benchmark setup was NOT started.")
        print()
        print("You can still use Prompt2GPU without an NVIDIA GPU:")
        print("  - open docs/index.html to view the published Colab/Tesla T4 dashboard")
        print("  - run: make verify")
        print("  - run: make setup-dashboard   # only if you want to rebuild charts")
        print()
        print("Do not install the heavy GPU stack on this machine unless NVIDIA")
        print("hardware/driver support is intentionally added later.")
        return 0

    print(f"Detected {len(gpus)} NVIDIA GPU(s):")
    for index, gpu in enumerate(gpus):
        print(
            f"  GPU {index}: {gpu['name']} | "
            f"VRAM {gpu['memory_mib']} MiB | driver {gpu['driver']}"
        )

    print()
    print("PASS: NVIDIA hardware/driver detected.")
    print("This does not guarantee identical performance to the published Tesla T4 run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
