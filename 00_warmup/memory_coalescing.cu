/*
 * ============================================================================
 * CHALLENGE 00: WARMUP — Memory Coalescing & Stride Benchmarks
 * ============================================================================
 *
 * Demonstrates:
 *   1. Warp memory coalescing across 32-byte cache line sectors
 *   2. The performance cliff of strided access (stride 1 -> stride 32 warp serialization)
 *   3. Effective vs useful memory bandwidth quantification on Ampere
 *
 * Hardware: RTX 2050 (SM 8.6, 64-bit bus, Ampere)
 */

#include "common/cuda_utils.cuh"
#include <cstdio>
#include <vector>

// Every element is accessed once, but neighbouring threads in a warp
// touch addresses separated by `stride` elements.
__global__ void benchmark_stride_read(const float* __restrict__ in,
                                      float* __restrict__ out,
                                      int n,
                                      int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int rows = n / stride;
    int idx = (i % rows) * stride + (i / rows);
    out[i] = in[idx];
}

// Coalesced baseline for comparison
__global__ void benchmark_coalesced_copy(const float* __restrict__ in,
                                         float* __restrict__ out,
                                         int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i];
    }
}

int main() {
    print_gpu_info();

    const int N = 1 << 24; // 16M floats = 64 MB
    const size_t bytes = N * sizeof(float);
    printf("Memory Coalescing Benchmark on N = %d elements (%.2f MB)\n\n",
           N, bytes / (1024.0 * 1024.0));

    std::vector<float> h_in(N);
    std::vector<float> h_out(N);
    init_random(h_in.data(), N, 0.0f, 100.0f);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    const int block_size = 256;
    const int grid_size = ceil_div(N, block_size);

    // Warm-up & baseline coalesced timing
    float ms_coalesced = time_kernel([&] {
        benchmark_coalesced_copy<<<grid_size, block_size>>>(d_in, d_out, N);
    });
    double bw_coalesced = (2.0 * bytes) / (ms_coalesced * 1e6); // Read + Write

    printf("========================================================================\n");
    printf("Stride | Latency (ms) | Bandwidth (GB/s) | Relative Slowdown | Efficiency\n");
    printf("------------------------------------------------------------------------\n");
    printf("  Seq  |   %7.3f ms |       %6.2f GB/s |             1.00x |    100.0%%\n",
           ms_coalesced, bw_coalesced);

    int test_strides[] = {1, 2, 4, 8, 16, 32, 64, 128};
    for (int stride : test_strides) {
        float ms = time_kernel([&] {
            benchmark_stride_read<<<grid_size, block_size>>>(d_in, d_out, N, stride);
        });

        double bw = (2.0 * bytes) / (ms * 1e6);
        double slowdown = ms / ms_coalesced;
        double efficiency = (bw / bw_coalesced) * 100.0;

        printf("   %3d |   %7.3f ms |       %6.2f GB/s |           %6.2fx |    %5.1f%%\n",
               stride, ms, bw, slowdown, efficiency);
    }
    printf("========================================================================\n");
    printf("\nNote: At stride 32, every thread in a 32-lane warp requests a distinct\n"
           "128-byte cache line, causing 32x memory bus transaction amplification.\n\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
