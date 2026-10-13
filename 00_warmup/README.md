# Challenge 00: GPU Warmup & Memory Hierarchy Fundamentals

First-principles explorations of GPU memory access patterns, warp execution dynamics, coalescing, and shared memory bank architecture on NVIDIA Ampere (RTX 2050, SM 8.6).

---

## Kernels Implemented

### 1. Vector Addition (`vector_add.cu`)
- **v1 Naive**: 1 thread per element, fixed grid.
- **v2 Grid-Stride Loop**: Decouples problem size $N$ from grid dimensions. Enables persistent thread execution with optimal SM occupancy and cache locality.
- **v3 Vectorized `float4`**: Uses 128-bit memory instructions (`reinterpret_cast<const float4*>`), saturating GDDR6 memory bus transaction lines.
- **v4 Vectorized `half2` (FP16)**: Demonstrates packed SIMD arithmetic using `__hadd2` instructions.

### 2. Memory Coalescing & Stride Benchmarks (`memory_coalescing.cu`)
- Compares sequential / coalesced loads against strided access ($s \in \{1, 2, 4, 8, 16, 32, 64, 128\}$).
- **The Stride 32 Cliff**: Demonstrates how a stride of 32 causes every thread in a 32-lane warp to touch a distinct 128-byte cache line, causing a $4.4\times$ memory throughput collapse (from 83.5 GB/s down to 18.8 GB/s).

### 3. Matrix Transpose & Bank Conflicts (`matrix_transpose.cu`)
- **v1 Naive**: Coalesced read, uncoalesced strided write ($49.5\text{ GB/s}$).
- **v2 Shared Memory Tiled**: Coalesced read and write, but suffers 32-way shared memory bank conflicts when reading columns ($82.5\text{ GB/s}$).
- **v3 Conflict-Free Padding**: Adding $+1$ column padding (`tile[32][33]`) shifts consecutive warp access to disjoint banks, achieving maximum hardware throughput ($82.6\text{ GB/s}$).

---

## Hardware Measurements (RTX 2050, SM 8.6)

| Kernel | Configuration | Latency | Bandwidth | Peak Efficiency |
|---|---|---|---|---|
| Vector Add Naive | $N = 2^{24}$ (64 MB) | 2.407 ms | 83.64 GB/s | 99.6% |
| Vector Add float4 | $N = 2^{24}$ (64 MB) | 2.434 ms | 82.73 GB/s | 99.2% |
| Vector Add half2 | $N = 2^{24}$ (32 MB) | 1.223 ms | 82.33 GB/s | 99.0% |
| Transpose Naive | $4096 \times 4096$ | 2.710 ms | 49.54 GB/s | 59.3% |
| Transpose Padded | $4096 \times 4096$ | 1.624 ms | 82.64 GB/s | 99.1% |
