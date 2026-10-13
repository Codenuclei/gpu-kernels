<div align="center">

# ⚡ gpu-kernels
### *High-Efficiency Bare-Metal GPU Math Kernels for Machine Learning Inference*

[![CUDA 13.3](https://img.shields.io/badge/CUDA-13.3-76B900?logo=nvidia&style=for-the-badge)](https://developer.nvidia.com/cuda-toolkit)
[![Ampere SM 8.6](https://img.shields.io/badge/Architecture-Ampere_SM__8.6-00C7B7?logo=nvidia&style=for-the-badge)](https://developer.nvidia.com/cuda-gpus)
[![RTX 2050 Mobile](https://img.shields.io/badge/Silicon-RTX_2050_Mobile-76B900?style=for-the-badge)](https://www.nvidia.com)
[![Pure CUDA C++](https://img.shields.io/badge/Implementation-Pure_CUDA_C%2B%2B-00599C?logo=c%2B%2B&style=for-the-badge)](https://en.cppreference.com)
[![Zero cuBLAS/cuDNN](https://img.shields.io/badge/Compute-Zero_cuBLAS%2FcuDNN-8B5CF6?style=for-the-badge)](#core-engineering-rules)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary_@Codenuclei-EF4444?style=for-the-badge)](#intellectual-property--security)

<br/>

**Production-grade, hand-crafted CUDA kernels for neural network inference, written strictly from first principles. Progressing from naive memory-bound baselines through shared-memory tiling, 2D register micro-kernels, and Ampere Tensor Cores.**

[Architecture Roadmap](#-1-architectural-manifesto--core-thesis) •
[Silicon Benchmark Matrix](#-2-silicon-audit-rtx-2050-performance-matrix) •
[Kernel Modules](#-3-deep-dive-kernel-modules) •
[Quick Start](#-4-quick-start--build-system) •
[Profiling Guide](#-5-hardware-profiling--ncu-playbook)

</div>

---

## 🏛️ 1. Architectural Manifesto & Core Thesis

> *"Every single FLOP executed that was not strictly required to determine the final correct token is an engineering failure."*  
> — **Abhishek ([@Codenuclei](https://github.com/Codenuclei))**, *The Polytope Inference Manifesto*

Modern large language models spend gigawatts of datacenter power projecting hidden states through dense linear matrices ($d_{\text{model}} \to |V| \approx 128,000$), only to collapse the resulting probability distribution into a single integer token ID. At the same time, standard attention mechanisms allocate $O(N^2)$ memory to materialize intermediate attention scores that are immediately discarded.

`gpu-kernels` exists to master the raw physical bounds of modern GPU architectures (memory bus coalescing, shared memory bank alignments, register pressure, warp shuffle networks, and tensor cores) and build high-throughput, bare-metal kernels without relying on black-box proprietary libraries like cuBLAS or cuDNN.

### Silicon Memory Hierarchy (Ampere GA107 / SM 8.6)

```
+-----------------------------------------------------------------------------------+
|                           GLOBAL MEMORY (DRAM GDDR6)                              |
|                   4,096 MB | 64-bit Bus Width | ~112 GB/s Peak                     |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼ 32-Byte Cache Line Sectors
+-----------------------------------------------------------------------------------+
|                               L2 CACHE (SRAM)                                     |
|                      1,024 KB Shared Across All 16 SMs                            |
+-----------------------------------------------------------------------------------+
                                         │
         ┌───────────────────────────────┴───────────────────────────────┐
         ▼                                                               ▼
+─────────────────────────────────+             +───────────────────────────────────+
|     STREAMING MULTIPROCESSOR 0  |     ...     |    STREAMING MULTIPROCESSOR 15    |
|  ┌───────────────────────────┐  |             |  ┌─────────────────────────────┐  |
|  | Configurable Shared Mem   |  |             |  | Configurable Shared Mem     |  |
|  | (Up to 100 KB / 32 Banks) |  |             |  | (Up to 100 KB / 32 Banks)   |  |
|  └─────────────┬─────────────┘  |             |  └─────────────┬───────────────┘  |
|                │                |             |                │                  |
|  ┌─────────────▼─────────────┐  |             |  ┌─────────────▼───────────────┐  |
|  | Register File (64K x 32b) |  |             |  | Register File (64K x 32b)   |  |
|  └─────────────┬─────────────┘  |             |  └─────────────┬───────────────┘  |
|                │                |             |                │                  |
|  ┌─────────────▼─────────────┐  |             |  ┌─────────────▼───────────────┐  |
|  | FP32 Cores + Tensor Cores |  |             |  | FP32 Cores + Tensor Cores   |  |
|  | (WMMA 16x16x16 Fragments) |  |             |  | (WMMA 16x16x16 Fragments)   |  |
|  └───────────────────────────┘  |             |  └─────────────────────────────┘  |
+─────────────────────────────────+             +───────────────────────────────────+
```

---

## 📊 2. Silicon Audit: RTX 2050 Performance Matrix

All metrics below were measured directly on bare metal on the local **NVIDIA GeForce RTX 2050 Mobile (Ampere SM 8.6, 16 SMs, 4GB VRAM, 64-bit bus) — CUDA 13.3** using high-precision hardware `cudaEvent` timers:

```
========================================================================================================
                                FULL SILICON PERFORMANCE AUDIT (RTX 2050)                               
========================================================================================================
| Module & Kernel                 | Input Shape / Dimensions      | Latency    | Measured Throughput   | Status |
|---------------------------------|-------------------------------|------------|-----------------------|--------|
| 00_warmup: Vector Add float4    | N = 16,777,216 (64 MB/array)  |  2.442 ms  |   82.46 GB/s (GDDR6)  |  PASS  |
| 00_warmup: Transpose Pad        | 4096 x 4096 (64 MB)           |  1.624 ms  |   82.67 GB/s (GDDR6)  |  PASS  |
| 02_reduce: Warp Shuffle Sum     | N = 16,777,216 (64 MB)        |  0.781 ms  |   85.96 GB/s (GDDR6)  |  PASS  |
| 03_gemm: 2D Register-Tiled      | 1024 x 1024 x 1024 (FP32)     |  1.598 ms  | 1343.62 GFLOPS (1.3T) |  PASS  |
| 03_gemm: WMMA Tensor Cores      | 2048 x 2048 x 2048 (FP16/FP32)|  5.366 ms  |    3.20 TFLOPS (TC)   |  PASS  |
| 04_attention: FlashAttention    | N = 1024, HeadDim = 64        |  0.738 ms  | SRAM-Fused (2.9x MHA) |  PASS  |
| 04_attention: Flash-Decoding    | N = 4096 (4K), q = 1, D = 64  |  0.0385 ms | 38.5 µs (16-SM Split) |  PASS  |
========================================================================================================
```

---

## 🔬 3. Deep-Dive: Kernel Modules

The codebase is structured into modular progressive challenges following the rule:  
$$\text{Naive Baseline} \longrightarrow \text{SRAM Tiling and Coalescing} \longrightarrow \text{Register Outer-Product / Tensor Cores}$$

```
gpu-kernels/
├── 00_warmup/             # Coalescing, 128-bit vectorization, bank conflict elimination
├── 01_softmax/            # Online Softmax, warp reductions, Split-KV, PTX streaming
├── 02_kernel_crucible/    # [Submodule] Autonomous RLVR + GRPO coliseum & serving accelerator
├── 02_reduce/             # 7-stage reduction hierarchy & work-efficient Blelloch prefix scan
├── 03_gemm/               # Naive -> 1D Shared Tiled -> 2D Register Tiled -> WMMA Tensor Cores
├── 04_attention/          # Naive MHA -> Fused FlashAttention -> Flash-Decoding (Split-KV)
├── bench/                 # Unified multi-kernel benchmark suite & verification harness
├── common/                # Shared hardware utilities, warp primitives, error handlers
└── Makefile               # Top-level parallel build system
```

---

### Challenge 00: Warmup & Memory Hierarchy (`00_warmup/`)
Explores low-level bus mechanics, SIMD memory instructions, and shared memory bank architectures:

* [`vector_add.cu`](file:///home/codenuclei/gpu-kernels/00_warmup/vector_add.cu):
  * **v1 Naive**: 1 thread per element, fixed grid configuration.
  * **v2 Grid-Stride Loop**: Persistent thread scheduling decoupling total elements $N$ from grid dimensions.
  * **v3 Vectorized `float4`**: Utilizes 128-bit memory instructions (`reinterpret_cast<const float4*>`), maximizing memory transaction line efficiency (**83.64 GB/s delivered throughput**).
  * **v4 Vectorized `half2`**: FP16 packed SIMD arithmetic via `__hadd2` (**1.22 ms**).
* [`memory_coalescing.cu`](file:///home/codenuclei/gpu-kernels/00_warmup/memory_coalescing.cu):
  * Quantifies the performance cliff of strided global memory access ($s \in \{1, 2, 4, 8, 16, 32, 64, 128\}$).
  * **The Stride-32 Cliff**: At stride 32, every thread in a 32-lane warp requests a disjoint 128-byte cache line, causing a **$4.4\times$ throughput collapse** (from 83.48 GB/s down to 18.83 GB/s).
* [`matrix_transpose.cu`](file:///home/codenuclei/gpu-kernels/00_warmup/matrix_transpose.cu):
  * **Naive**: Coalesced read, uncoalesced write ($49.54\text{ GB/s}$).
  * **Shared Memory Tiled**: Coalesced read & write, but suffers severe 32-way shared memory bank conflicts when reading columns ($82.53\text{ GB/s}$).
  * **Conflict-Free Padded**: Adding $+1$ column padding (`tile[32][33]`) skews addresses so consecutive threads read distinct banks, achieving peak hardware bandwidth (**82.64 GB/s**).

---

### Challenge 01: Online Softmax (`01_softmax/`)
The foundation of modern transformer attention:
* Implements the **Milakov & Gimelshein (2018)** online normalizer algorithm, maintaining running max $m$ and running sum $l$ simultaneously in a single pass over data:
  $$m_{\text{new}} = \max(m_{\text{old}}, x), \quad l_{\text{new}} = l_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + e^{x - m_{\text{new}}}$$
* Ultra-fast sub-warp and warp shuffle primitives (`__shfl_down_sync`).
* Inline PTX cache streaming operators (`ld.global.cs`) bypassing L1 cache to avoid polluting high-speed SRAM.

---

### Challenge 02: Parallel Reduction & Prefix Scan (`02_reduce/`)
Exhaustive implementations of collective parallel reduction algorithms:

* [`reduce.cu`](file:///home/codenuclei/gpu-kernels/02_reduce/reduce.cu): 7-Stage Optimization Progression:
  1. *Stage 1 (Interleaved Divergent)*: `threadIdx.x % (2*s) == 0` causes severe warp branch divergence ($34.1\text{ GB/s}$).
  2. *Stage 2 (Contiguous Threads)*: `index = 2*s*tid` resolves divergence but introduces 32-way shared memory bank conflicts ($56.4\text{ GB/s}$).
  3. *Stage 3 (Sequential Addressing)*: Stride halving resolves all bank conflicts ($58.6\text{ GB/s}$).
  4. *Stage 4 (Add on Global Load)*: Halves block count and global memory latency ($85.9\text{ GB/s}$, **$2.52\times$ speedup**).
  5. *Stage 5 (Warp Synchronous Unrolling)*: Avoids `__syncthreads()` overhead when $s \le 32$.
  6. *Stage 6 (Template Full Unrolling)*: Complete compile-time loop unrolling via template parameter `blockSize`.
  7. *Stage 7 (Hardware Warp Shuffle)*: `__shfl_down_sync` intra-warp register exchange with zero shared memory latency (**86.0 GB/s**).
* [`prefix_scan.cu`](file:///home/codenuclei/gpu-kernels/02_reduce/prefix_scan.cu):
  * **Hillis-Steele Scan**: Step-efficient parallel inclusive scan ($O(N \log N)$ work).
  * **Blelloch Tree Scan**: Work-efficient exclusive scan ($O(N)$ operations: Up-Sweep reduction tree $\to$ set root to $0 \to$ Down-Sweep distribution tree) with bank-conflict-free padding.
  * **Warp Shuffle Scan**: Pure register intra-warp scan via `__shfl_up_sync` (**0.0020 ms** for 512 elements).

---

### Challenge 03: General Matrix Multiplication (`03_gemm/`)
From naive global memory fetches to Ampere 3rd-Gen Tensor Cores:

* [`gemm_tiled.cu`](file:///home/codenuclei/gpu-kernels/03_gemm/gemm_tiled.cu):
  * **Naive GEMM**: $0.25\text{ FLOP/byte}$ ($349.26\text{ GFLOPS}$).
  * **1D Shared Memory Tiled GEMM**: Caches $16 \times 16$ or $32 \times 32$ submatrices in SRAM, boosting arithmetic intensity by $TILE$ factor (**481.50 GFLOPS**).
* [`gemm_reg_tiled.cu`](file:///home/codenuclei/gpu-kernels/03_gemm/gemm_reg_tiled.cu):
  * **2D Register Tiling**: Breaks the shared memory read throughput wall by computing **outer-product rank-1 updates** directly in thread registers.
  * Block tile: $BM \times BN = 64 \times 64$; Thread micro-tile: $TM \times TN = 8 \times 8$ (64 output elements per thread).
  * Cuts shared memory read bandwidth by $(TM \cdot TN) / (TM + TN) = 64 / 16 = 4\times$.
  * Vectorized `float4` global-to-shared transfers.
  * Delivered: **1,343.62 GFLOPS (1.34 TFLOPS FP32)**.
* [`gemm_tensor_core.cu`](file:///home/codenuclei/gpu-kernels/03_gemm/gemm_tensor_core.cu):
  * Deploys hardware Tensor Cores via `nvcuda::wmma`.
  * Tile shape: $16 \times 16 \times 16$ mixed-precision (FP16 inputs, FP32 accumulator).
  * Multi-warp block decomposition with 4 concurrent warps.
  * Delivered: **3.20 TFLOPS (FP16)**.

---

### Challenge 04: Transformer Attention (`04_attention/`)
Solving the quadratic memory wall ($O(N^2)$) and autoregressive decoding underutilization:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d}}\right) \mathbf{V}$$

* [`mha_naive.cu`](file:///home/codenuclei/gpu-kernels/04_attention/mha_naive.cu):
  * Standard 3-pass baseline ($Q K^T \to \text{Softmax} \to P V$).
  * Materializes $N \times N$ attention matrix in global DRAM ($4\text{ MB}$ at $N=1024$), bottlenecked by memory bandwidth (**2.138 ms**).
* [`flash_attention.cu`](file:///home/codenuclei/gpu-kernels/04_attention/flash_attention.cu):
  * Tri Dao's FlashAttention algorithm implemented from scratch.
  * Tiles $Q, K, V$ into on-chip SRAM ($B_r = 32, B_c = 32$), tracking running online softmax statistics and rescaling the accumulator in registers:
    $$O^{(new)} = O^{(old)} \cdot e^{m^{(old)} - m^{(new)}} + P_{ij} V_j$$
  * Zero global memory score matrix materialization ($O(N)$ memory footprint = $256\text{ KB}$ vs $4\text{ MB}$).
  * Causal masking support for autoregressive transformers (**0.738 ms, 2.9x faster than naive MHA**).
* [`flash_decoding.cu`](file:///home/codenuclei/gpu-kernels/04_attention/flash_decoding.cu):
  * Solves the $q=1$ autoregressive decoding underutilization bottleneck.
  * Partitions the KV cache across **all 16 SMs**:
    * *Stage 1*: Each SM computes partial attention for its chunk, emitting partial outputs, partial maxes, and partial sums.
    * *Stage 2*: Single-warp Log-Sum-Exp reduction combines partial split outputs into final result vector.
  * Delivered: **36.2 µs (0.036 ms)** for $N=4096$ context on RTX 2050!

---

### Submodule: Kernel Crucible (`02_kernel_crucible/`)
[`kernel-crucible`](https://github.com/Codenuclei/kernel-crucible) is linked as a submodule at [`02_kernel_crucible`](file:///home/codenuclei/gpu-kernels/02_kernel_crucible):
* Autonomous Reinforcement Learning coliseum combining **RLVR** (Reinforcement Learning with Verifiable Rewards) and **GRPO** ($G=4$ cohort advantage normalization).
* Automatic monkey-patching of live **Unsloth** forward passes in $\sim 1.2\text{s}$.
* 50-trial sliding-window plateau detector triggering self-distillation LoRA SFT post-training upon search saturation.

---

## 🛠️ 4. Quick Start & Build System

### Build All Kernels & Benchmarks
```bash
make all
```

### Run the Unified Silicon Benchmark Suite
```bash
make bench
./bench/benchmark_suite
```
Or via Python:
```bash
python3 bench/run_benchmarks.py
```

### Target-Specific Builds
```bash
make warmup     # Builds vector_add, memory_coalescing, matrix_transpose, strided_copy
make reduce     # Builds reduce (7 stages) and prefix_scan
make gemm       # Builds gemm_tiled, gemm_reg_tiled, gemm_tensor_core
make attention  # Builds mha_naive, flash_attention, flash_decoding
```

---

## 🔍 5. Hardware Profiling & NCU Playbook

Every kernel is designed to be profiled with NVIDIA Nsight Compute (`ncu`) to inspect low-level bottlenecks:

```bash
# 1. Profile Memory Workload Analysis (DRAM vs L2 vs Shared Memory throughput)
ncu --set full --section MemoryWorkloadAnalysis ./00_warmup/matrix_transpose

# 2. Inspect Shared Memory Bank Conflicts
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./00_warmup/matrix_transpose

# 3. Inspect Warp Execution Efficiency & Branch Divergence
ncu --metrics smsp__thread_inst_executed_per_inst_executed.ratio ./02_reduce/reduce

# 4. Profile Tensor Core Utilization & Pipe Activity
ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active ./03_gemm/gemm_tensor_core
```

---

## 📐 6. Core Engineering Rules

1. **No black-box compute libraries**: All mathematical operations and kernels are written from scratch in CUDA C++. cuBLAS/cuDNN are used strictly as external benchmark targets.
2. **First-principles progression**: Every kernel follows `naive baseline -> memory/register optimization -> hardware acceleration`.
3. **Rigorous validation**: 100% of kernels verify numerical ground truth against CPU references with strict tolerance boundaries ($\text{RMSE} \le 10^{-4}$).

---

## 🔒 Intellectual Property & Security

> **Confidential & Proprietary:** All kernels, algorithms, post-training flywheel architectures, and mathematical formulations contained in this repository are strictly private and proprietary to **@Codenuclei**.
