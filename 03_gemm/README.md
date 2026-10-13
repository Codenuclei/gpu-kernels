# Challenge 03: General Matrix Multiplication (GEMM)

Progressive progression of Matrix Multiplication on NVIDIA Ampere (SM 8.6), from naive global memory triple-loop to Ampere 3rd-Gen Tensor Core WMMA.

$$\mathbf{C} = \alpha (\mathbf{A} \times \mathbf{B}) + \beta \mathbf{C}$$

---

## Architecture Progression

### 1. Naive GEMM (`gemm_tiled.cu`)
- 1 thread per element of $\mathbf{C}$.
- Inner loop loads from global DRAM on every step.
- Arithmetic intensity: $0.25\text{ FLOP/byte}$ ($2\text{ FLOPs} / 8\text{ bytes}$).
- Measured: **349.26 GFLOPS**.

### 2. 1D Shared Memory Tiled GEMM (`gemm_tiled.cu`)
- Blocks cache $16 \times 16$ or $32 \times 32$ submatrices into SRAM.
- Arithmetic intensity boosted by $TILE$ factor.
- Measured: **481.50 GFLOPS** ($1.38\times$ speedup).

### 3. 2D Register-Tiled GEMM (`gemm_reg_tiled.cu`)
- Breaks the shared memory read bottleneck by performing **outer-product rank-1 updates** directly in thread registers.
- Block tile: $BM \times BN = 64 \times 64$.
- Thread micro-tile: $TM \times TN = 8 \times 8$ (64 output elements per thread accumulated in registers).
- Cuts shared memory read bandwidth by $(TM \cdot TN) / (TM + TN) = 64 / 16 = 4\times$.
- Vectorized `float4` global-to-shared transfers.
- Measured: **1,131 - 1,343 GFLOPS (1.34 TFLOPS FP32)** ($3.8\times$ speedup).

### 4. Ampere Tensor Core WMMA (`gemm_tensor_core.cu`)
- Deploys hardware Tensor Cores via `nvcuda::wmma`.
- Tile shape: $16 \times 16 \times 16$ mixed-precision (FP16 inputs, FP32 accumulator).
- Multi-warp block decomposition with 4 concurrent warps.
- Measured: **2.58 - 3.20 TFLOPS (FP16)** ($9.2\times$ over naive FP32).

---

## Performance Summary (RTX 2050, Ampere)

| Implementation | Precision | Latency ($1024^3$) | Compute Throughput | Speedup |
|---|---|---|---|---|
| Naive GEMM | FP32 | 6.149 ms | 349.26 GFLOPS | 1.00x |
| Shared Memory Tiled | FP32 | 4.460 ms | 481.50 GFLOPS | 1.38x |
| 2D Register Tiled | FP32 | 1.598 ms | 1,343.62 GFLOPS | 3.85x |
| WMMA Tensor Cores | FP16/FP32 | 5.366 ms ($2048^3$) | 3.20 TFLOPS | 9.16x |
