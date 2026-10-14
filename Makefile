PYTHON3 ?= python3
VENV := .venv
DASH_PYTHON := $(VENV)/bin/python3
GPU_VENV := .venv-gpu
GPU_PYTHON := $(GPU_VENV)/bin/python3

.PHONY: help setup setup-dashboard verify dashboard serve preflight-gpu setup-gpu hf-baseline start-vllm vllm-baseline concurrency kv-pressure stop-vllm audit clean

help:
	@echo "Prompt2GPU commands"
	@echo "  make verify           Verify published summaries from raw CSV data (no install)"
	@echo "  make serve            View committed dashboard at http://localhost:8000 (no install)"
	@echo "  make setup-dashboard  Install lightweight chart/dashboard tools (~1-5 min typical)"
	@echo "  make dashboard        Rebuild plots + docs/index.html from committed results"
	@echo "  make preflight-gpu    Check NVIDIA GPU/driver before any heavy install"
	@echo "  make setup-gpu        OPTIONAL heavy GPU setup; warns + asks before multi-GB install"
	@echo "  make audit            Run pre-push repository audit"

# Safe default setup: dashboard only. Never pulls GPU/CUDA packages.
setup: setup-dashboard

setup-dashboard:
	@echo "Expected time: typically ~1-5 minutes; depends on network/cache."
	$(PYTHON3) -m venv $(VENV)
	$(DASH_PYTHON) -m pip install --upgrade pip
	$(DASH_PYTHON) -m pip install -r requirements.txt
	@echo "Dashboard environment ready. No manual venv activation is required."

verify:
	$(PYTHON3) scripts/verify_results.py

# Do not silently install dependencies. Tell the user exactly what is missing.
dashboard:
	@test -x $(DASH_PYTHON) || (echo "Dashboard environment missing. Run 'make setup-dashboard' first (~1-5 min typical)." && exit 2)
	$(DASH_PYTHON) scripts/build_dashboard.py

serve:
	@echo "Opening committed dashboard requires no Python packages."
	@echo "Visit: http://localhost:8000"
	$(PYTHON3) -m http.server 8000 --directory docs

preflight-gpu:
	$(PYTHON3) scripts/preflight_gpu.py

setup-gpu:
	$(PYTHON3) scripts/setup_gpu.py

hf-baseline:
	@test -x $(GPU_PYTHON) || (echo "GPU environment missing. Run 'make setup-gpu' first." && exit 2)
	$(GPU_PYTHON) src/hf_baseline.py

start-vllm:
	@test -x $(GPU_PYTHON) || (echo "GPU environment missing. Run 'make setup-gpu' first." && exit 2)
	$(GPU_PYTHON) src/start_vllm_server.py

vllm-baseline:
	@test -x $(GPU_PYTHON) || (echo "GPU environment missing. Run 'make setup-gpu' first." && exit 2)
	$(GPU_PYTHON) src/vllm_baseline.py

concurrency:
	@test -x $(GPU_PYTHON) || (echo "GPU environment missing. Run 'make setup-gpu' first." && exit 2)
	$(GPU_PYTHON) src/vllm_concurrency.py

kv-pressure:
	@test -x $(GPU_PYTHON) || (echo "GPU environment missing. Run 'make setup-gpu' first." && exit 2)
	$(GPU_PYTHON) src/vllm_kv_pressure.py

stop-vllm:
	@test -x $(GPU_PYTHON) || (echo "GPU environment missing. Run 'make setup-gpu' first." && exit 2)
	$(GPU_PYTHON) src/stop_vllm_server.py

audit:
	$(PYTHON3) scripts/audit_repo.py
	$(PYTHON3) scripts/build_v1_1_dashboard.py --check

clean:
	rm -rf src/__pycache__ scripts/__pycache__
