"""Offline, no-GPU tests of the copied V1.1 benchmark (mock HTTP only)."""
import csv
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))

# Tests must be runnable without installing heavy GPU/model dependencies.
try:
    import requests
except ImportError:
    fake_requests = types.ModuleType('requests')
    fake_requests.post = lambda *a, **kw: None
    fake_requests.get = lambda *a, **kw: None
    sys.modules['requests'] = fake_requests
if 'transformers' not in sys.modules:
    fake_tf = types.ModuleType('transformers')
    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model):
            return cls()
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return 'FAKE_TEST_PROMPT'
        def __call__(self, prompt, add_special_tokens):
            return {'input_ids': list(range(8))}
    fake_tf.AutoTokenizer = FakeTokenizer
    sys.modules['transformers'] = fake_tf

spec = importlib.util.spec_from_file_location('benchmark_v11', SRC / 'vllm_concurrency_v1_1.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class FakeResponse:
    status_code = 200
    def __init__(self, streaming, token_count, coalesced=False):
        self.streaming = streaming
        self.token_count = token_count
        self.coalesced = coalesced
    def raise_for_status(self):
        return None
    def __enter__(self):
        return self
    def __exit__(self, typ, val, tb):
        return None
    def json(self):
        return {'usage': {'completion_tokens': self.token_count}}
    def iter_lines(self, **kwargs):
        if self.coalesced:
            events = ['token one token two', 'token three', 'token four']
        else:
            events = ['token'] * self.token_count
        for part in events:
            yield 'data: ' + json.dumps({'choices': [{'text': part}]})
        yield 'data: ' + json.dumps({
            'choices': [], 'usage': {'completion_tokens': self.token_count}})
        yield 'data: [DONE]'

class BenchmarkTests(unittest.TestCase):
    def _run(self, mode, coalesced=False):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'fresh_run'
            with mock.patch.object(mod, 'OUTPUT_TOKENS', 4), \
                 mock.patch.object(mod, 'build_prompt', return_value=('FAKE_TEST_PROMPT', 8)), \
                 mock.patch.object(mod.requests, 'post',
                     side_effect=lambda url, json, **kwargs: FakeResponse(
                         streaming=json.get('stream', False),
                         token_count=4, coalesced=coalesced)), \
                 mock.patch.object(mod.requests, 'get', return_value=FakeResponse(False, 4)):
                mod.main(['--levels', '2', '--mode', mode,
                          '--warmup-batches', '1', '--measured-batches', '2',
                          '--prefix-caching', 'on', '--vllm-version', '0.31.0',
                          '--output-dir', str(out)])
                summary = json.loads((out / 'vllm_concurrency_summary.json').read_text())
                with (out / 'vllm_concurrency_raw.csv').open() as fd:
                    rows = list(csv.DictReader(fd))
                with (out / 'vllm_concurrency_itl.csv').open() as fd:
                    intervals = list(csv.DictReader(fd))
                return summary, rows, intervals

    def test_streaming_metrics(self):
        summary, rows, itls = self._run('streaming')
        self.assertEqual(summary['prefix_caching'], 'on')
        self.assertEqual(summary['results'][0]['measured_requests'], 4)
        self.assertEqual(summary['results'][0]['itl_eligible_requests'], 4)
        self.assertEqual(summary['results'][0]['itl_total_measured_intervals'], 12)
        self.assertIsNotNone(summary['results'][0]['p95_ttft_sec'])
        self.assertIsNotNone(summary['results'][0]['p99_tpot_sec'])
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(itls), 12)
        self.assertEqual({int(r['output_tokens']) for r in rows}, {4})

    def test_nonstreaming_does_not_invent_ttft(self):
        summary, rows, itls = self._run('nonstreaming')
        self.assertEqual(summary['results'][0]['measured_requests'], 4)
        self.assertIsNone(summary['results'][0]['p95_ttft_sec'])
        self.assertIsNone(summary['results'][0]['mean_tpot_sec'])
        self.assertEqual(itls, [])
        self.assertEqual(len(rows), 4)

    def test_merged_sse_events_no_false_itl(self):
        summary, rows, itls = self._run('streaming', coalesced=True)
        self.assertIsNotNone(summary['results'][0]['p95_ttft_sec'])
        self.assertEqual(summary['results'][0]['itl_eligible_requests'], 0)
        self.assertIsNone(summary['results'][0]['p95_itl_sec'])
        self.assertEqual(itls, [])

if __name__ == '__main__':
    unittest.main()
