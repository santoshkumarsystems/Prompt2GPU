# Author: Santosh Kumar
# Project: Prompt2GPU
# Version: 1.0

"""Ask PyTorch to release unused CUDA memory owned by this process.

BEGINNER ROADMAP
----------------
A Python process can keep GPU memory cached even after an operation finishes.
This utility runs Python garbage collection, asks PyTorch to release its CUDA
cache, synchronizes the GPU, and finally prints ``nvidia-smi`` status.

Important: one process cannot free GPU memory owned by another process.  For
example, this script cannot unload a model that is still alive inside a
separate Colab notebook kernel.

If you are new to Python, read ``docs/PYTHON_WALKTHROUGH.md`` first.
"""
import gc
import subprocess
import torch


def main():
    """Run garbage collection, release this process's PyTorch CUDA cache, and print GPU status."""
    print("=== Prompt2GPU GPU Cleanup / Diagnostic ===")
    print("Python objects reclaimed:", gc.collect())
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        torch.cuda.synchronize()
    subprocess.run([
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.free,memory.total,utilization.gpu",
        "--format=csv",
    ], check=False)


# BEGINNER NOTE: Python sets ``__name__`` to ``"__main__"`` when this
# file is run directly.  When another file imports it, this block is skipped.
# That lets us reuse functions without accidentally starting the whole program.
if __name__ == "__main__":
    main()
