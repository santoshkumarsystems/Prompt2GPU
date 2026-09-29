#!/usr/bin/env python3
# Author: Santosh Kumar
# Project: Prompt2GPU
# Version: 1.0

"""Safely create the optional Prompt2GPU GPU benchmark environment.

The script runs the NVIDIA preflight *before* downloading heavy dependencies.
If no NVIDIA GPU is detected, setup stops immediately.

Even when a GPU is detected, the script prints a prominent time/download/disk
warning and requires the user to type ``YES`` before installation begins.
No virtual-environment activation command is ever required; Makefile targets
call ``.venv-gpu/bin/python3`` directly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from preflight_gpu import detect_nvidia_gpus


ROOT = Path(__file__).resolve().parents[1]
GPU_VENV = ROOT / ".venv-gpu"
GPU_PYTHON = GPU_VENV / "bin" / "python3"
REQUIREMENTS = ROOT / "requirements-gpu.txt"


def run(command: list[str]) -> None:
    """Run one setup command and stop immediately if it fails."""
    print("+", " ".join(command))
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> int:
    """Perform hardware gate, explicit confirmation, and optional GPU install."""
    print("=== Prompt2GPU Optional GPU Setup ===")

    gpus = detect_nvidia_gpus()
    if not gpus:
        print()
        print("STOP: No NVIDIA GPU/driver detected.")
        print("No PyTorch/CUDA/cuDNN/NCCL/vLLM installation was started.")
        print("Published V1.0 results are already viewable in docs/index.html.")
        print("For lightweight tools use: make setup-dashboard")
        return 0

    print("Detected NVIDIA hardware:")
    for index, gpu in enumerate(gpus):
        print(
            f"  GPU {index}: {gpu['name']} | "
            f"VRAM {gpu['memory_mib']} MiB | driver {gpu['driver']}"
        )

    print()
    print("WARNING: Full GPU benchmark setup is HEAVY and OPTIONAL.")
    print("It installs PyTorch plus CUDA runtime libraries, cuDNN, NCCL, vLLM,")
    print("Transformers, Accelerate, and supporting packages.")
    print()
    print("Typical impact (varies by platform/cache/network):")
    print("  Download: several GB")
    print("  Disk:     several additional GB")
    print("  Time:     ~10-30 min on a fast connection; 30-60+ min on slower Wi-Fi")
    print("  Internet: required")
    print()
    print("Published V1.0 measurements were collected on Google Colab / Tesla T4.")
    print("Your local results WILL vary with GPU, power/thermal limits, driver, and software.")
    print()

    answer = input("Type YES to start the heavy GPU installation, or press Enter to cancel: ").strip()
    if answer != "YES":
        print("Cancelled. No heavy GPU installation was started.")
        return 0

    if not GPU_VENV.exists():
        run([sys.executable, "-m", "venv", str(GPU_VENV)])

    run([str(GPU_PYTHON), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(GPU_PYTHON), "-m", "pip", "install", "uv"])

    uv_binary = GPU_VENV / "bin" / "uv"
    run(
        [
            str(uv_binary),
            "pip",
            "install",
            "--python",
            str(GPU_PYTHON),
            "--torch-backend=auto",
            "-r",
            str(REQUIREMENTS),
        ]
    )

    print()
    print("GPU environment ready: .venv-gpu")
    print("No manual 'source .venv-gpu/bin/activate' step is required.")
    print("Use Makefile benchmark targets, for example: make hf-baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
