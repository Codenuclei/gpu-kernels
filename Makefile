NVCC ?= nvcc
ARCH ?= sm_86
CXXFLAGS ?= -O3 -arch=$(ARCH) -I. --use_fast_math -Xcompiler -Wall

WARMUP_BINS = 00_warmup/vector_add 00_warmup/memory_coalescing 00_warmup/matrix_transpose 00_warmup/strided_copy
REDUCE_BINS = 02_reduce/reduce 02_reduce/prefix_scan
GEMM_BINS = 03_gemm/gemm_tiled 03_gemm/gemm_reg_tiled 03_gemm/gemm_tensor_core
ATTN_BINS = 04_attention/mha_naive 04_attention/flash_attention 04_attention/flash_decoding
BENCH_BINS = bench/benchmark_suite

all: warmup reduce gemm attention bench

warmup: $(WARMUP_BINS)

reduce: $(REDUCE_BINS)

gemm: $(GEMM_BINS)

attention: $(ATTN_BINS)

bench: $(BENCH_BINS)

00_warmup/%: 00_warmup/%.cu common/cuda_utils.cuh
	$(NVCC) $(CXXFLAGS) $< -o $@

02_reduce/%: 02_reduce/%.cu common/cuda_utils.cuh
	$(NVCC) $(CXXFLAGS) $< -o $@

03_gemm/%: 03_gemm/%.cu common/cuda_utils.cuh
	$(NVCC) $(CXXFLAGS) $< -o $@

04_attention/%: 04_attention/%.cu common/cuda_utils.cuh
	$(NVCC) $(CXXFLAGS) $< -o $@

bench/%: bench/%.cu common/cuda_utils.cuh
	$(NVCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(WARMUP_BINS) $(REDUCE_BINS) $(GEMM_BINS) $(ATTN_BINS) $(BENCH_BINS)
	rm -f *.o *.so

.PHONY: all warmup reduce gemm attention bench clean
