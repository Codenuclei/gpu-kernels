# gpu-kernels

Optimized GPU math kernels for ML inference. Written from scratch in CUDA C++.

**Hardware:** RTX 2050 (SM 8.6, 4GB VRAM, Ampere) — CUDA 13.3

## Structure

```
00_warmup/      — vector ops, memory patterns, coalescing
01_softmax/     — online softmax (the gateway kernel)
02_reduce/      — parallel reduction, prefix scan
03_gemm/        — matrix multiply: naive → tiled → tensor core
04_attention/   — fused multi-head attention
bench/          — benchmarking utilities
common/         — shared headers, timing, error checking
```

## Rules

1. No cuBLAS/cuDNN for the kernel itself. Benchmark against them.
2. Profile every kernel with `ncu`. Know your bottleneck.
3. Each kernel gets: naive version → optimized version → benchmark comparison.
