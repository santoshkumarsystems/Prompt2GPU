#!/usr/bin/env python3
# Author: Santosh Kumar
# Project: Prompt2GPU
# Version: 1.0

"""Build the checked-in Grafana dashboard; optionally provision in Google Colab.

Pure standard-library script. Does not install packages, start vLLM/Prometheus,
or run a GPU benchmark.

From repository root:
    python3 scripts/setup_grafana_dashboard.py

Inside Colab, with Grafana already installed and Prometheus running:
    python3 scripts/setup_grafana_dashboard.py --provision-colab

To avoid hard-coding the location of the working checkout in Colab, provision
from any directory with --dashboard-output pointing at the cloned project.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DASHBOARD = REPO / "observability" / "grafana" / "vllm-performance.json"
DEFAULT_COLAB_ROOT = Path("/content/prompt2gpu-observability")
DASHBOARD_UID = "prompt2gpu-vllm"
DATASOURCE_UID = "prompt2gpu-prometheus"

PANELS = [
    ("Output Tokens/sec", "sum(rate(vllm:generation_tokens_total[1m]))", "short"),
    (
        "P95 TTFT",
        "histogram_quantile(0.95, sum by (le) "
        "(rate(vllm:time_to_first_token_seconds_bucket[5m])))",
        "s",
    ),
    (
        "P95 TPOT",
        "histogram_quantile(0.95, sum by (le) "
        "(rate(vllm:request_time_per_output_token_seconds_bucket[5m])))",
        "s",
    ),
    ("GPU KV Cache Usage", "100 * avg(vllm:kv_cache_usage_perc)", "percent"),
    ("Running Requests", "sum(vllm:num_requests_running)", "short"),
    ("Waiting Requests", "sum(vllm:num_requests_waiting)", "short"),
    (
        "Prefix Cache Hit Rate",
        "100 * sum(rate(vllm:prefix_cache_hits_total[5m])) "
        "/ clamp_min(sum(rate(vllm:prefix_cache_queries_total[5m])), 1)",
        "percent",
    ),
    (
        "P95 Queue Time",
        "histogram_quantile(0.95, sum by (le) "
        "(rate(vllm:request_queue_time_seconds_bucket[5m])))",
        "s",
    ),
    (
        "P95 Prefill Time",
        "histogram_quantile(0.95, sum by (le) "
        "(rate(vllm:request_prefill_time_seconds_bucket[5m])))",
        "s",
    ),
]


def make_dashboard() -> dict:
    """Build deterministic JSON from the metric query definitions above."""
    panels = []
    for index, (title, expression, unit) in enumerate(PANELS):
        panels.append(
            {
                "id": index + 1,
                "title": title,
                "type": "timeseries",
                "datasource": {"type": "prometheus", "uid": DATASOURCE_UID},
                "gridPos": {
                    "x": (index % 2) * 12,
                    "y": (index // 2) * 8,
                    "w": 12,
                    "h": 8,
                },
                "targets": [
                    {"refId": "A", "expr": expression, "legendFormat": title}
                ],
                "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
                "options": {
                    "legend": {"displayMode": "list", "placement": "bottom"},
                    "tooltip": {"mode": "single"},
                },
            }
        )
    return {
        "uid": DASHBOARD_UID,
        "title": "Prompt2GPU - vLLM Inference Performance",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "5s",
        "time": {"from": "now-15m", "to": "now"},
        "panels": panels,
        "tags": ["Prompt2GPU", "vllm", "inference"],
    }


def write_if_changed(path: Path, text: str) -> bool:
    """Write only on changes, so reruns don't modify git timestamps needlessly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text() == text:
        return False
    path.write_text(text)
    return True


def build_dashboard(path: Path) -> None:
    changed = write_if_changed(path, json.dumps(make_dashboard(), indent=2) + "\n")
    print(f"Dashboard {'updated' if changed else 'unchanged'}: {path}")
    print(f"Panels: {len(PANELS)}")


def port_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def grafana_request(endpoint: str) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:3000{endpoint}", timeout=4) as response:
        return json.load(response)


def is_dashboard_ready() -> bool:
    try:
        entries = grafana_request("/api/search?query=Prompt2GPU")
        return any(item.get("uid") == DASHBOARD_UID for item in entries)
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


def find_grafana_binary(root: Path) -> Path:
    candidates = sorted(root.glob("*/bin/grafana")) + sorted(root.glob("*/bin/grafana-server"))
    if not candidates:
        raise RuntimeError(f"Grafana installation not found beneath {root}. Run the existing Colab Grafana setup first.")
    return candidates[0]


def stop_tracked_grafana(root: Path) -> None:
    """Stop only the Grafana process recorded by our Colab installation."""
    pid_path = root / "grafana.pid"
    if not pid_path.is_file():
        raise RuntimeError("Port 3000 is running but grafana.pid is missing; refusing to stop an unknown process.")
    pid = int(pid_path.read_text().strip())
    proc_cmdline = Path(f"/proc/{pid}/cmdline")
    if not proc_cmdline.is_file() or "grafana" not in proc_cmdline.read_bytes().decode(errors="replace").lower():
        raise RuntimeError(f"PID {pid} is not a recognized Grafana process; refusing to stop it.")
    print(f"Restarting tracked Grafana process PID {pid} (vLLM and Prometheus untouched)")
    os.kill(pid, signal.SIGTERM)
    for _ in range(20):
        if not port_listening(3000):
            return
        time.sleep(0.5)
    raise RuntimeError("Grafana did not release port 3000; inspect process and logs manually.")


def start_grafana(binary: Path, root: Path) -> None:
    home = binary.parent.parent
    env = os.environ.copy()
    env.update(
        {
            "GF_SERVER_HTTP_ADDR": "127.0.0.1",
            "GF_SERVER_HTTP_PORT": "3000",
            "GF_SECURITY_ALLOW_EMBEDDING": "true",
            "GF_AUTH_ANONYMOUS_ENABLED": "true",
            "GF_AUTH_ANONYMOUS_ORG_ROLE": "Viewer",
            "GF_PATHS_DATA": str(root / "grafana-data"),
            "GF_PATHS_LOGS": str(root / "grafana-logs"),
            "GF_PATHS_PLUGINS": str(root / "grafana-plugins"),
        }
    )
    command = [str(binary)]
    if binary.name == "grafana":
        command.append("server")
    command += ["--homepath", str(home)]
    with (root / "grafana.log").open("a") as log:
        proc = subprocess.Popen(
            command,
            cwd=home,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (root / "grafana.pid").write_text(str(proc.pid))
    print(f"Grafana launched (PID {proc.pid})")


def provision_colab(root: Path, dashboard_path: Path) -> None:
    """Install dashboard into the already-running private Colab Grafana."""
    if not Path("/content").is_dir():
        raise RuntimeError("--provision-colab requires a Colab-style /content environment")
    binary = find_grafana_binary(root)
    home = binary.parent.parent
    deployed_path = root / "dashboards" / "vllm-performance.json"
    deployed_path.parent.mkdir(parents=True, exist_ok=True)
    content = dashboard_path.read_text()
    dashboard_changed = write_if_changed(deployed_path, content)

    provisioning_dir = home / "conf" / "provisioning" / "dashboards"
    provider_path = provisioning_dir / "prompt2gpu.yml"
    provider_text = (
        "apiVersion: 1\n"
        "providers:\n"
        "  - name: Prompt2GPU\n"
        "    orgId: 1\n"
        "    folder: Prompt2GPU\n"
        "    type: file\n"
        "    disableDeletion: false\n"
        "    updateIntervalSeconds: 10\n"
        "    options:\n"
        f"      path: {deployed_path.parent}\n"
    )
    provider_changed = write_if_changed(provider_path, provider_text)
    print(f"Colab dashboard: {deployed_path} ({'updated' if dashboard_changed else 'unchanged'})")
    print(f"Provisioning file: {provider_path} ({'updated' if provider_changed else 'unchanged'})")

    # Provisioning provider must be loaded on Grafana startup. File-based
    # dashboards subsequently reload automatically without server restarts.
    if port_listening(3000):
        if provider_changed:
            stop_tracked_grafana(root)
            start_grafana(binary, root)
        else:
            print("Grafana already running; waiting for dashboard refresh (no restart).")
    else:
        start_grafana(binary, root)

    for _ in range(40):
        if is_dashboard_ready():
            print("Grafana dashboard found: True")
            print("Dashboard: Prompt2GPU - vLLM Inference Performance")
            print("URL: http://127.0.0.1:3000/d/prompt2gpu-vllm")
            return
        time.sleep(1)
    raise RuntimeError(
        "Dashboard not found after 40s. Inspect " + str(root / "grafana.log")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provision-colab", action="store_true", help="Deploy dashboard to installed Grafana in Colab")
    parser.add_argument("--dashboard-output", type=Path, default=DEFAULT_DASHBOARD, help="Path for generated dashboard JSON")
    parser.add_argument("--colab-root", type=Path, default=DEFAULT_COLAB_ROOT, help="Existing Colab Grafana runtime directory")
    args = parser.parse_args()
    build_dashboard(args.dashboard_output)
    if args.provision_colab:
        provision_colab(args.colab_root, args.dashboard_output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
