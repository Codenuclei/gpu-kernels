# ⚡ gpu-kernels

High-efficiency, hand-optimized GPU math kernels for machine learning inference written from scratch in CUDA C++.

Built and benchmarked directly on silicon: **NVIDIA GeForce RTX 2050 (Ampere SM 8.6, 16 SMs, 4GB VRAM, 64-bit GDDR6 bus) — CUDA 13.3**.

---

## 🏛️ Project Architecture & Completion Status

| Challenge | Module | Status | Highlights |
|---|---|---|---|
| **00** | [`00_warmup/`](file:///home/codenuclei/gpu-kernels/00_warmup/) | **Complete** | Vector Add (Naive, Grid-Stride, `float4`, `half2`), Coalescing vs Stride Analysis, Transpose Bank Conflicts. |
| **01** | [`01_softmax/`](file:///home/codenuclei/gpu-kernels/01_softmax/) | **Complete** | Online Softmax (Milakov-Gimelshein), Warp-level reductions, Split-KV Stage 1/2, PTX Streaming. |
| **02** | [`02_reduce/`](file:///home/codenuclei/gpu-kernels/02_reduce/) | **Complete** | 7-Stage Reduction Hierarchy (Interleaved $\to$ Warp Shuffle), Blelloch & Hillis-Steele Prefix Scan. |
| **03** | [`03_gemm/`](file:///home/codenuclei/gpu-kernels/03_gemm/) | **Complete** | Naive $\to$ Shared Memory Tiled $\to$ 2D Register Tiled (1.34 TFLOPS) $\to$ Ampere WMMA Tensor Cores (3.20 TFLOPS). |
| **04** | [`04_attention/`](file:///home/codenuclei/gpu-kernels/04_attention/) | **Complete** | Naive Multi-Head Attention, Fused FlashAttention (Causal/Non-Causal), Flash-Decoding (Split-KV across 16 SMs). |
| **Bench** | [`bench/`](file:///home/codenuclei/gpu-kernels/bench/) | **Complete** | Unified multi-kernel benchmark harness & automated verification suite. |
| **Common** | [`common/`](file:///home/codenuclei/gpu-kernels/common/) | **Complete** | Hard-crash error checking (`CUDA_CHECK`), warp & block primitives, vectorized loaders, CPU verification. |

---

## 📊 Silicon Audit: RTX 2050 Performance Matrix

The following results were measured directly on local hardware via `make bench && ./bench/benchmark_suite`:

```
=========================================================================================
                    FULL GPU KERNEL SUITE BENCHMARK & HARDWARE AUDIT                    
=========================================================================================
+------------------------------+--------------------+----------------------+--------+
| Module / Kernel              | Latency            | Measured Throughput  | Status |
+------------------------------+--------------------+----------------------+--------+
| 00_warmup: Vector Add float4 |   2.442 ms         |  82.46 GB/s (DRAM)   | PASS   |
| 00_warmup: Transpose Pad     |   1.624 ms         |  82.67 GB/s (DRAM)   | PASS   |
| 02_reduce: Warp Shuffle Sum  |   0.781 ms         |  85.96 GB/s (DRAM)   | PASS   |
| 03_gemm: 2D Register-Tiled   |   1.598 ms         | 1343.62 GFLOPS (FP32)| PASS   |
| 03_gemm: WMMA Tensor Cores   |   5.366 ms         |   3.20 TFLOPS (FP16) | PASS   |
| 04_attention: Flash-Decoding |  0.0385 ms (38.5 us)| Split-KV (16 SMs)   | PASS   |
+------------------------------+--------------------+----------------------+--------+
```

---

## 🛠️ Build & Quick Start

### Build All Modules
```bash
make all
```

### Run Unified Benchmark Suite
```bash
make bench
./bench/benchmark_suite
```
Or via Python:
```bash
python3 bench/run_benchmarks.py
```

### Module-Specific Targets
```bash
make warmup     # Builds vector_add, memory_coalescing, matrix_transpose, strided_copy
make reduce     # Builds reduce (7 stages) and prefix_scan
make gemm       # Builds gemm_tiled, gemm_reg_tiled, gemm_tensor_core
make attention  # Builds mha_naive, flash_attention, flash_decoding
```

---

## 📐 Core Engineering Rules

1. **No black-box libraries for kernel compute**: Kernels are implemented strictly from scratch in CUDA C++.
2. **First-principles progression**: Every kernel follows `naive baseline -> memory/register optimization -> hardware acceleration`.
3. **Rigorous validation**: 100% of kernels verify numerical ground truth against CPU references with strict tolerance boundaries.
