# Author: Santosh Kumar
# Project: Prompt2GPU
# Version: 1.0

"""Start the detached vLLM server used by Prompt2GPU V1.0.

WHAT THIS PROGRAM DOES
----------------------
* creates the log directory if needed;
* launches ``vllm serve`` as a separate operating-system process;
* sends server output to ``logs/vllm_server.log``;
* stores the process ID (PID) so ``stop_vllm_server.py`` can stop it later;
* polls ``/health`` until HTTP 200 confirms the server is ready.

Prefix caching is disabled for the published concurrency/KV experiments so
repeated prompts cannot receive an accidental cached-prefix advantage.

``subprocess.Popen`` is used instead of a normal function call because vLLM is
a long-running server process.  See ``docs/PYTHON_WALKTHROUGH.md`` for a
beginner explanation of subprocesses, exceptions, and the ``main`` pattern.
"""

import subprocess
import time

import requests

from benchmark_config import MODEL, LOGS_DIR, VLLM_HEALTH_URL

LOG_FILE = LOGS_DIR / "vllm_server.log"
PID_FILE = LOGS_DIR / "vllm_server.pid"
STARTUP_TIMEOUT_SEC = 360
POLL_INTERVAL_SEC = 5


def main():
    """Launch vLLM, record its PID/log, and wait until the health endpoint reports ready."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    with LOG_FILE.open("w") as log_file:
        server = subprocess.Popen(
            [
                "vllm",
                "serve",
                MODEL,
                "--dtype",
                "float16",
                "--gpu-memory-utilization",
                "0.80",
                "--max-model-len",
                "4096",
                "--no-enable-prefix-caching",
                "--port",
                "8000",
            ],
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    PID_FILE.write_text(str(server.pid))
    print(f"vLLM launch requested (PID {server.pid})")
    print(f"Log: {LOG_FILE}")
    print("Waiting for /health", end="", flush=True)

    deadline = time.time() + STARTUP_TIMEOUT_SEC

    while time.time() < deadline:
        if server.poll() is not None:
            print("\nServer exited before becoming healthy.")
            if LOG_FILE.exists():
                tail = LOG_FILE.read_text(errors="replace").splitlines()[-40:]
                print("\n".join(tail))
            raise SystemExit(server.returncode or 1)

        try:
            response = requests.get(VLLM_HEALTH_URL, timeout=2)
            if response.status_code == 200:
                print("\nvLLM server ready (HTTP 200).")
                return
        except requests.RequestException:
            pass

        print(".", end="", flush=True)
        time.sleep(POLL_INTERVAL_SEC)

    print("\nTimed out waiting for vLLM health check.")
    raise SystemExit(1)


# BEGINNER NOTE: Python sets ``__name__`` to ``"__main__"`` when this
# file is run directly.  When another file imports it, this block is skipped.
# That lets us reuse functions without accidentally starting the whole program.
if __name__ == "__main__":
    main()
