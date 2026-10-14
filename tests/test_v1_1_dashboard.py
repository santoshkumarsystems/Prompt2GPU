"""Real-data integrity checks for the Prompt2GPU V1.1 recorded dashboard.

These tests read the committed, measured Tesla T4 files. They never run a GPU
workload and never substitute test fixtures for the measured evidence.
"""
from __future__ import annotations

import importlib.util
import csv
import re
import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_v1_1_dashboard.py"
spec = importlib.util.spec_from_file_location("v1_1_dashboard_builder", SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class HistoricalDashboardTests(unittest.TestCase):
    def test_real_measurements_match_export(self):
        for label in ("initial", "repeat"):
            with self.subTest(run=label):
                expected = builder.convert_run(label)
                published = json.loads(
                    (ROOT / "docs/v1_1_data" / f"{label}.json").read_text()
                )
                self.assertEqual(expected, published)
                self.assertEqual(published["classification"], "RECORDED_MEASUREMENT_NOT_LIVE")
                self.assertEqual(len(published["batches"]), 5)
                self.assertEqual(published["summary"]["measured_requests"], 320)
                self.assertEqual(set(published["metrics"]), set(builder.REQUIRED_METRICS))
                for metric in published["metrics"].values():
                    self.assertTrue(metric["samples"])
                    for t, value in metric["samples"]:
                        self.assertTrue(math.isfinite(t))
                        self.assertTrue(value is None or math.isfinite(value))

    def test_decode_phase_client_proxy_from_real_requests(self):
        for label in ("initial", "repeat"):
            with self.subTest(run=label):
                with (ROOT / "results_v1_1" / label / "vllm_concurrency_raw.csv").open(newline="") as f:
                    requests = list(csv.DictReader(f))
                published = json.loads((ROOT / "docs/v1_1_data" / f"{label}.json").read_text())
                self.assertEqual(len(requests), 320)
                for batch in published["batches"]:
                    group = [r for r in requests if int(r["batch_number"]) == batch["batch"]]
                    self.assertEqual(len(group), 64)
                    observed_ms = sorted((int(r["output_tokens"]) - 1) * float(r["tpot_sec"]) * 1000
                                         for r in group)
                    pos = (len(observed_ms) - 1) * .95
                    lo, hi = math.floor(pos), math.ceil(pos)
                    computed = observed_ms[lo] + (observed_ms[hi] - observed_ms[lo]) * (pos - lo)
                    self.assertAlmostEqual(batch["p95_decode_phase_client_proxy_ms"], computed, places=7)
                self.assertGreater(published["summary"]["p95_decode_phase_client_proxy_ms"], 0)
        html = (ROOT / "docs/v1_1.html").read_text()
        js = (ROOT / "docs/v1_1_dashboard.js").read_text()
        self.assertIn("Decode Phase Duration — P95", html)
        self.assertIn("not simulated", html)
        self.assertIn('drawBatch("batch-decode-proxy"', js)

    def test_decode_timeline_replay_and_clean_labels(self):
        html = (ROOT / "docs/v1_1.html").read_text(encoding="utf-8")
        js = (ROOT / "docs/v1_1_dashboard.js").read_text(encoding="utf-8")
        self.assertIn("Prefill Duration — P95", html)
        self.assertIn("Decode Phase Duration — P95", html)
        self.assertIn("Recorded inference telemetry", html)
        self.assertIn('id="metric-decode-proxy"', html)
        self.assertIn('id="decode-replay-status"', html)
        self.assertIn('function drawDecodeTimeline(){', js)
        self.assertIn('points.map(b=>relative(run,b.end_unix_s))', js)
        self.assertIn('b.p95_decode_phase_client_proxy_ms', js)
        self.assertIn('for(const name of METRICS)drawMetric(name);drawDecodeTimeline();', js)
        self.assertIn('  drawDecodeTimeline();\n  $("jit-note")', js)
        self.assertEqual(js.count('drawBatch("metric-decode-proxy"'), 0)
        self.assertIn('each point is a batch P95, not a Prometheus scrape', html)

    def test_no_private_files_and_historical_labels(self):
        for forbidden in ("vllm_server.log", "benchmark.log", "command.json"):
            self.assertFalse(list((ROOT / "docs/v1_1_data").glob(f"**/{forbidden}")))
        html = (ROOT / "docs/v1_1.html").read_text()
        js = (ROOT / "docs/v1_1_dashboard.js").read_text()
        self.assertIn("RECORDED DATA · NOT LIVE", html)
        self.assertIn("Correlation, not isolated proof of causality", html)
        self.assertIn("connectgaps:false", js)
        self.assertIn("RECORDED_MEASUREMENT_NOT_LIVE", js)
        for public_file in (ROOT / "docs/v1_1_data/initial.json", ROOT / "docs/v1_1_data/repeat.json",
                            ROOT / "docs/v1_1_dashboard.js"):
            content = public_file.read_text()
            self.assertNotIn("/content/drive/", content)
            self.assertNotIn("C:\\Users\\", content)
            self.assertNotRegex(content, re.compile(r"/home/[^/\s]+/"))
        for name in builder.REQUIRED_METRICS[:9]:
            self.assertIn('data-metric="' + name + '"', html)

    def test_navigation_is_rebuild_safe(self):
        self.assertIn('href="v1_1.html"', (ROOT / "docs/index.html").read_text())
        self.assertIn('href="v1_1.html"', (ROOT / "scripts/build_dashboard.py").read_text())
        self.assertIn('$(PYTHON3) scripts/build_v1_1_dashboard.py --check',
                      (ROOT / "Makefile").read_text())
        self.assertIn('[V1.1 interactive historical dashboard](docs/v1_1.html)',
                      (ROOT / "README.md").read_text())


if __name__ == "__main__":
    unittest.main()
