# Benchmarking Suite & Automated Verification Harness

Unified benchmarking utilities and performance auditing for all CUDA kernels.

---

## Usage

### 1. Unified C++ Benchmark Suite
Compile and run the full end-to-end hardware audit:
```bash
make bench
./bench/benchmark_suite
```

### 2. Python Runner
```bash
python3 bench/run_benchmarks.py
```

### 3. Individual Kernel Benchmarks
You can also build and benchmark any specific challenge:
```bash
# Warmup
make 00_warmup/vector_add && ./00_warmup/vector_add
make 00_warmup/memory_coalescing && ./00_warmup/memory_coalescing
make 00_warmup/matrix_transpose && ./00_warmup/matrix_transpose

# Reduction
make 02_reduce/reduce && ./02_reduce/reduce
make 02_reduce/prefix_scan && ./02_reduce/prefix_scan

# GEMM
make 03_gemm/gemm_tiled && ./03_gemm/gemm_tiled
make 03_gemm/gemm_reg_tiled && ./03_gemm/gemm_reg_tiled
make 03_gemm/gemm_tensor_core && ./03_gemm/gemm_tensor_core

# Attention
make 04_attention/mha_naive && ./04_attention/mha_naive
make 04_attention/flash_attention && ./04_attention/flash_attention
make 04_attention/flash_decoding && ./04_attention/flash_decoding
```
