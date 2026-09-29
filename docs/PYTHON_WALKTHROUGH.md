# Python walkthrough for Prompt2GPU

**Author:** Santosh Kumar
**Version:** 1.0

This guide is for a reader who has **never written Python before**. You do not
need to learn the whole language before reading Prompt2GPU. The project uses a
small, repeatable set of Python ideas, and this document explains those ideas
in the same order that you encounter them in the repository.

> If your goal is only to **run** Prompt2GPU, follow `RUNBOOK.md`. If your goal
> is to **understand the code**, start here and keep this guide open beside the
> `src/` files.

## 1. How to read a Python file

Python executes statements from top to bottom. A typical Prompt2GPU file has
this shape:

```python
import time

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def helper():
    return 42


def main():
    value = helper()
    print(value)


if __name__ == "__main__":
    main()
```

Read it in four parts:

1. **imports** — code brought in from Python or installed packages;
2. **constants/configuration** — values shared by the program;
3. **functions** — named blocks of reusable logic;
4. **main guard** — starts the program only when the file is executed directly.

Python uses **indentation** (spaces at the beginning of a line) instead of
curly braces. Everything indented under `def`, `if`, `for`, `with`, etc. belongs
to that block.

## 2. Imports

```python
import time
import statistics
from pathlib import Path
from transformers import AutoTokenizer
```

`import time` loads Python's `time` module. `from pathlib import Path` imports
only the `Path` name. `transformers` is a third-party package installed with
`pip`/`uv`.

Prompt2GPU uses:

- standard library: `csv`, `json`, `math`, `os`, `pathlib`, `statistics`,
  `subprocess`, `threading`, `time`;
- HTTP: `requests`;
- model/tokenizer: `transformers`;
- GPU/library baseline: `torch`;
- charts/dashboard: `matplotlib`, `plotly`.

## 3. Variables and constants

```python
OUTPUT_TOKENS = 128
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
```

`=` assigns a value. Python does not require a type declaration such as C's
`int output_tokens`. Upper-case names are a convention meaning "treat this as a
constant".

Common data types in this project:

```python
count = 128                  # int (integer)
latency = 1.9189             # float (decimal number)
name = "Tesla T4"            # str (text)
ready = True                 # bool (True/False)
missing = None               # no value
```

## 4. Lists

A list stores items in order:

```python
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32, 64]
```

Access one item with a zero-based index:

```python
first = CONCURRENCY_LEVELS[0]   # 1
```

Loop through every item:

```python
for concurrency in CONCURRENCY_LEVELS:
    print(concurrency)
```

## 5. Dictionaries

A dictionary stores `key: value` pairs:

```python
row = {
    "latency_sec": 1.92,
    "output_tokens": 128,
}
```

Read a value by key:

```python
latency = row["latency_sec"]
```

Prompt2GPU uses dictionaries heavily because JSON objects and CSV rows map
naturally into this structure.

## 6. Functions

```python
def gpu_name():
    return torch.cuda.get_device_name(0)
```

`def` defines a function. Parentheses hold inputs (parameters). `return` sends a
value back to the caller.

A function with parameters:

```python
def send_request(request_id, formatted_prompt):
    ...
```

Call it like:

```python
result = send_request(3, prompt)
```

Every Prompt2GPU function has a docstring immediately below `def`. In VS Code,
hovering over the function name shows that explanation.

## 7. `main()` and the main guard

At the bottom of executable files you see:

```python
if __name__ == "__main__":
    main()
```

When you run:

```bash
python3 src/vllm_baseline.py
```

Python sets `__name__` to `"__main__"`, so `main()` runs. When another file
imports that module, Python uses the module's real name and skips the block.
This prevents an import from accidentally starting a benchmark.

## 8. `if`, `else`, and validation

```python
if output_tokens != OUTPUT_TOKENS:
    raise RuntimeError("Benchmark invalid")
```

`!=` means "not equal". `raise` stops the program with an error. Prompt2GPU uses
fail-fast validation deliberately: if a benchmark does not produce exactly the
workload we specified, we do not silently publish that run.

## 9. Loops and `range`

```python
for run_index in range(total_runs):
    ...
```

If `total_runs` is 11, `range(11)` generates `0` through `10`. Python programmers
often use zero-based counters internally and print `run_index + 1` for humans.

## 10. List comprehensions

This:

```python
latencies = [row["latency_sec"] for row in measured]
```

is a compact form of:

```python
latencies = []
for row in measured:
    latencies.append(row["latency_sec"])
```

Prompt2GPU uses list comprehensions mainly to pull one column of measurements
out of many result dictionaries.

## 11. f-strings

```python
print(f"Mean latency: {mean_latency:.3f}s")
```

An `f` before a quoted string allows `{...}` expressions inside the text.
`:.3f` means "display this floating-point value with three digits after the
decimal point". Formatting changes only how the number is printed, not the
stored measurement.

## 12. Timing with `time.perf_counter()`

```python
start = time.perf_counter()
# work
end = time.perf_counter()
latency_sec = end - start
```

`perf_counter()` is a high-resolution monotonic timer suitable for measuring
elapsed time. It is better for benchmarks than wall-clock date/time.

## 13. Why `torch.cuda.synchronize()` is needed

CUDA launches work asynchronously. Without synchronization, Python can stop a
timer before the GPU has finished.

```python
torch.cuda.synchronize()
start = time.perf_counter()
outputs = model.generate(...)
torch.cuda.synchronize()
end = time.perf_counter()
```

The first synchronization clears earlier GPU work; the second waits for the
measured work to finish.

## 14. HTTP requests and vLLM

vLLM runs as a server. The benchmark sends JSON over HTTP:

```python
response = requests.post(
    VLLM_URL,
    json=payload,
    timeout=120,
)
response.raise_for_status()
result = response.json()
```

- `requests.post` sends an HTTP POST request;
- `json=payload` converts a Python dictionary into JSON;
- `timeout` prevents waiting forever;
- `raise_for_status()` raises an error for HTTP failures;
- `response.json()` converts returned JSON into Python data.

## 15. Warm-up versus measured runs

GPU/model systems often perform one-time setup on first use. Prompt2GPU keeps
warm-up rows but excludes them from published means:

```python
is_warmup = run_index < WARMUP_RUNS
```

Later:

```python
measured = [row for row in rows if not row["warmup"]]
```

This makes the measurement policy explicit and auditable.

## 16. CSV and JSON outputs

CSV preserves row-oriented raw measurements:

```python
with open(RAW_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
```

`with` automatically closes the file even if an error occurs.

JSON preserves nested summaries:

```python
with open(SUMMARY_JSON, "w") as f:
    json.dump(summary, f, indent=2)
```

The dashboard reads these committed files instead of embedding benchmark
numbers in source code.

## 17. Mean, median, standard deviation, and P95

Examples:

```python
statistics.mean(latencies)
statistics.median(latencies)
statistics.stdev(latencies)
```

P95 means 95% of measured values are at or below approximately that latency.
Prompt2GPU implements a small linear-interpolation `percentile()` helper so the
calculation is explicit and reproducible without another dependency.

## 18. Concurrency with `ThreadPoolExecutor`

The concurrency experiment creates many simultaneous HTTP clients:

```python
with ThreadPoolExecutor(max_workers=concurrency) as executor:
    futures = [
        executor.submit(send_request, request_id, formatted_prompt)
        for request_id in range(concurrency)
    ]
```

A **thread** here does not run the LLM. It waits on one HTTP request. The vLLM
server decides how requests are batched and scheduled on the GPU.

`executor.submit(...)` starts work and returns a `Future`, which represents a
result that will arrive later. `as_completed(futures)` lets the program collect
results in whichever order the requests finish.

## 19. Aggregate throughput versus per-request throughput

For a batch of concurrent requests:

```python
aggregate_tokens_per_sec = total_output_tokens / batch_wall_time_sec
```

Do **not** add individual requests' tokens/second values. Aggregate throughput
asks: "How many output tokens did the server deliver to all users during the
actual wall-clock batch duration?"

## 20. Background metric sampling

The KV-pressure experiment needs metrics while inference is active. It starts a
background thread:

```python
sampler_thread = threading.Thread(..., daemon=True)
sampler_thread.start()
```

The thread repeatedly downloads `/metrics` until the main benchmark sets a
`threading.Event`:

```python
stop_event.set()
```

This captures transient peaks that would disappear if we looked only after all
requests had completed.

## 21. Prometheus text parsing

A metric line looks like:

```text
vllm:kv_cache_usage_perc{engine="0",...} 0.576
```

`extract_metric_value()` skips comment lines, finds the metric name, takes the
final whitespace-separated field, and converts it with `float(...)`.

A value of `0.576` means 57.6% KV-cache usage.

## 22. Subprocesses and the vLLM server

`start_vllm_server.py` launches an external command:

```python
server = subprocess.Popen([...])
```

`Popen` returns immediately while vLLM keeps running. Prompt2GPU redirects its
stdout/stderr to a log file and stores `server.pid` so another script can stop
it later.

The start script does **not** assume launch means ready. It polls `/health` and
returns only after HTTP 200.

## 23. Exceptions

Examples:

```python
try:
    response = requests.get(...)
except requests.RequestException:
    pass
```

`try` runs code that may fail. `except` handles a known failure type. Prompt2GPU
uses exceptions differently depending on purpose:

- a transient health poll may fail while vLLM is still starting, so retry;
- invalid token count makes the benchmark invalid, so raise and stop;
- one missed metrics sample should not destroy a long inference run, so skip
  that sample and continue.

## 24. Project-relative paths

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
```

`__file__` is the current Python file. `.resolve()` gives its absolute path.
`.parents[1]` moves up from `src/file.py` to the repository root. This is why a
user can clone Prompt2GPU anywhere instead of editing a personal path such as
`/content/drive/MyDrive/...`.

## 25. Environment-variable override

```python
VLLM_HOST = os.environ.get(
    "PROMPT2GPU_VLLM_HOST",
    "http://localhost:8000",
)
```

By default the server is local. An advanced user can override it without code
changes:

```bash
export PROMPT2GPU_VLLM_HOST=http://10.0.0.25:8000
```

## 26. Dashboard generation

`scripts/build_dashboard.py` performs this pipeline:

```text
results/*.json + results/*.csv
              ↓
        Python dictionaries
              ↓
      Matplotlib PNG charts
              ↓
       Plotly HTML charts
              ↓
plots/*.png + docs/index.html
```

The benchmark scripts measure. The dashboard script presents. Keeping these
separate prevents visualization code from affecting benchmark timing.

## 27. Verification script

`scripts/verify_results.py` independently recomputes published statistics from
raw CSVs. It intentionally does not trust the summary JSON simply because the
file exists.

Conceptually:

```text
raw CSV measurements
        ↓ recompute
calculated statistics
        ↓ compare
committed summary JSON
        ↓
PASS or error
```

This is one of Prompt2GPU's most important open-source reproducibility rules.

## 28. File-by-file reading order

If you are learning Python from this project, use this order:

1. `src/benchmark_config.py` — constants, strings, `Path`.
2. `src/inspect_vllm_metrics.py` — one HTTP GET and a simple list filter.
3. `src/clear_gpu.py` — function call, `if`, `try/except`, subprocess.
4. `src/hf_baseline.py` — loops, dictionaries, CSV/JSON, PyTorch timing.
5. `src/vllm_baseline.py` — HTTP POST plus the same result pattern.
6. `src/start_vllm_server.py` — subprocess and health polling.
7. `src/stop_vllm_server.py` — PID file and process signals.
8. `src/vllm_concurrency.py` — threads/Futures and aggregate throughput.
9. `src/vllm_kv_pressure.py` — background sampling and Prometheus parsing.
10. `scripts/verify_results.py` — independent recomputation/checking.
11. `scripts/build_dashboard.py` — plotting and HTML generation.

## 29. How to experiment safely

Make changes on a branch and preserve the committed published results:

```bash
git switch -c experiment/my-change
```

If you change benchmark methodology, write new output files or clearly version
the experiment. Do not overwrite published measurements and continue using old
README claims.

## 30. The most important principle

Prompt2GPU favors **explicit code over clever code**. A few extra lines are
acceptable when they make methodology easier to audit. A reader should be able
to answer:

- What workload is being sent?
- What exactly is timed?
- How many warm-ups and measured runs are used?
- What data is raw and what data is summarized?
- Which process owns the GPU memory?
- How is concurrency created?
- How are metrics sampled?
- Where did every chart number come from?

If the code or documentation leaves any of those ambiguous, treat that as a
bug in the open-source project.
