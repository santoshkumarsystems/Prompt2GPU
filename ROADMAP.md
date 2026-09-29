# Prompt2GPU roadmap

**Author:** Santosh Kumar
**Version:** 1.0

## V1.0 — Single-GPU inference systems — complete

Goal: understand one GPU deeply before distributing the workload.

- Controlled HF/PyTorch vs vLLM baseline
- Exact input/output token control
- Concurrency scaling and P95 latency
- vLLM scheduler behavior
- Live Prometheus metrics
- Context + KV-cache pressure
- Queue-time analysis
- Raw evidence, verification script, checksums
- Static README plots and interactive dashboard

## V2 — Multi-GPU inference systems

Goal: move from one GPU to multiple GPUs and measure communication/topology costs rather than simply enabling tensor parallelism.

### Core experiments

1. **1 GPU vs 2+ GPUs**
   - same model/workload where feasible
   - throughput, TTFT, inter-token latency, end-to-end latency
   - per-GPU memory and utilization
   - scaling efficiency

2. **Tensor parallelism**
   - vLLM `tensor_parallel_size`
   - model sharding across devices
   - per-layer / per-rank memory behavior
   - communication overhead versus compute saved

3. **NCCL collectives lab**
   - AllReduce
   - AllGather
   - ReduceScatter
   - Broadcast
   - latency/bandwidth versus message size

4. **Topology-aware measurements**
   - GPU↔GPU connectivity
   - PCIe versus NVLink when hardware is available
   - `nvidia-smi topo -m`
   - communication bottlenecks and locality

5. **Serving under multi-GPU concurrency**
   - throughput scaling
   - tail latency
   - KV-memory distribution
   - scheduler behavior

### V2 deliverable

A second dashboard section that answers: *When does adding another GPU help, and when does communication become the bottleneck?*

## V3 — Production inference infrastructure

Goal: operate LLM inference as a production service rather than a single benchmark process.

### Platform

- Docker/OCI image
- Kubernetes deployment
- NVIDIA device plugin / GPU resource requests
- replicas and service load balancing
- configuration and secrets separation

### Reliability and traffic management

- admission control
- request queue limits
- retries / timeouts
- readiness and liveness probes
- graceful shutdown
- overload behavior

### Autoscaling

- throughput/latency-driven scaling
- GPU utilization and queue depth signals
- scale-out versus batch-size tradeoffs
- cold-start measurements

### Observability

- Prometheus metrics
- Grafana dashboard
- request rate, queue depth, TTFT, inter-token latency, P95/P99
- GPU utilization/memory
- KV-cache utilization
- error rates and SLOs

### Failure experiments

- kill a serving pod during traffic
- GPU-serving process crash/restart
- overloaded queue
- model-load failure
- replica recovery and request impact

### V3 deliverable

A production-style architecture plus repeatable experiments answering: *How does an inference service behave under load, scale-out, and failure?*
