# Author: Santosh Kumar
# Project: Prompt2GPU
# Version: 1.0

"""Stop the detached vLLM server started by ``start_vllm_server.py``.

The start script writes the server process ID to ``logs/vllm_server.pid``.
This script reads that number and sends SIGTERM to the whole process group so
the API server and its worker processes can shut down cleanly.

If you are new to Python, read ``docs/PYTHON_WALKTHROUGH.md`` first.
"""

import os
import signal
from benchmark_config import LOGS_DIR

PID_FILE = LOGS_DIR / "vllm_server.pid"


def main():
    """Read the saved PID, stop the vLLM process group, and remove the PID file."""
    if not PID_FILE.exists():
        print("No Prompt2GPU vLLM PID file found.")
        return
    pid = int(PID_FILE.read_text().strip())
    try:
        os.killpg(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to vLLM process group {pid}.")
    except ProcessLookupError:
        print(f"Process group {pid} is not running.")
    finally:
        PID_FILE.unlink(missing_ok=True)


# BEGINNER NOTE: Python sets ``__name__`` to ``"__main__"`` when this
# file is run directly.  When another file imports it, this block is skipped.
# That lets us reuse functions without accidentally starting the whole program.
if __name__ == "__main__":
    main()
