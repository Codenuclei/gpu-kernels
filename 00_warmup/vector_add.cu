/*
 * ============================================================================
 * CHALLENGE 00: WARMUP — Vector Addition
 * ============================================================================
 *
 * Demonstrates:
 *   1. Naive element-wise addition (1 thread per element)
 *   2. Grid-stride loops (decoupling problem size from grid dimension)
 *   3. Vectorized float4 memory loads (128-bit memory bus saturation)
 *   4. Vectorized half2 (FP16) SIMD addition
 *
 * Hardware: RTX 2050 (SM 8.6, 64-bit bus, Ampere)
 */

#include "common/cuda_utils.cuh"
#include <cstdio>
#include <vector>

// ----------------------------------------------------------------------------
// Version 1: Naive (1 thread per element)
// ----------------------------------------------------------------------------
__global__ void vector_add_naive(const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 float* __restrict__ c,
                                 int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// ----------------------------------------------------------------------------
// Version 2: Grid-Stride Loop
// ----------------------------------------------------------------------------
__global__ void vector_add_grid_stride(const float* __restrict__ a,
                                       const float* __restrict__ b,
                                       float* __restrict__ c,
                                       int n) {
    int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        c[idx] = a[idx] + b[idx];
    }
}

// ----------------------------------------------------------------------------
// Version 3: Vectorized float4 (128-bit loads)
// ----------------------------------------------------------------------------
__global__ void vector_add_float4(const float4* __restrict__ a,
                                  const float4* __restrict__ b,
                                  float4* __restrict__ c,
                                  int n4) {
    int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n4; idx += stride) {
        float4 va = a[idx];
        float4 vb = b[idx];
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        c[idx] = vc;
    }
}

// ----------------------------------------------------------------------------
// Version 4: Vectorized half2 (FP16)
// ----------------------------------------------------------------------------
__global__ void vector_add_half2(const half2* __restrict__ a,
                                 const half2* __restrict__ b,
                                 half2* __restrict__ c,
                                 int n2) {
    int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n2; idx += stride) {
        c[idx] = __hadd2(a[idx], b[idx]);
    }
}

// ----------------------------------------------------------------------------
// CPU Ground Truth Reference
// ----------------------------------------------------------------------------
void cpu_vector_add(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    print_gpu_info();

    // 16 Million floats = 64 MB per vector (Read A + Read B + Write C = 192 MB per pass)
    const int N = 1 << 24;
    const size_t bytes = N * sizeof(float);
    printf("Benchmarking Vector Addition on N = %d elements (%.2f MB/vector)\n\n",
           N, bytes / (1024.0 * 1024.0));

    // Allocate host memory
    std::vector<float> h_a(N);
    std::vector<float> h_b(N);
    std::vector<float> h_c_cpu(N);
    std::vector<float> h_c_gpu(N);

    init_random(h_a.data(), N, -1.0f, 1.0f);
    init_random(h_b.data(), N, -1.0f, 1.0f);
    cpu_vector_add(h_a.data(), h_b.data(), h_c_cpu.data(), N);

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    const int block_size = 256;
    const int num_blocks = ceil_div(N, block_size);
    const int sm_count = 16;
    const int persistent_blocks = sm_count * 4; // 64 blocks saturated on 16 SMs

    printf("--- Correctness Verification ---\n");

    // 1. Naive
    vector_add_naive<<<num_blocks, block_size>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    verify_close(h_c_cpu.data(), h_c_gpu.data(), N, 1e-4f, 1e-4f, "v1_naive");

    // 2. Grid-stride
    CUDA_CHECK(cudaMemset(d_c, 0, bytes));
    vector_add_grid_stride<<<persistent_blocks, block_size>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    verify_close(h_c_cpu.data(), h_c_gpu.data(), N, 1e-4f, 1e-4f, "v2_grid_stride");

    // 3. Float4
    CUDA_CHECK(cudaMemset(d_c, 0, bytes));
    vector_add_float4<<<persistent_blocks, block_size>>>(
        reinterpret_cast<const float4*>(d_a),
        reinterpret_cast<const float4*>(d_b),
        reinterpret_cast<float4*>(d_c),
        N / 4);
    CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    verify_close(h_c_cpu.data(), h_c_gpu.data(), N, 1e-4f, 1e-4f, "v3_float4");

    printf("\n--- Performance Benchmarks ---\n");
    // Bandwidth calculation: 3 * N * 4 bytes / time (seconds)
    auto calc_bandwidth = [&](float ms) {
        return (3.0 * N * sizeof(float)) / (ms * 1e6); // GB/s
    };

    float ms_naive = time_kernel([&] {
        vector_add_naive<<<num_blocks, block_size>>>(d_a, d_b, d_c, N);
    });
    printf("v1_naive        : %6.3f ms | Bandwidth: %6.2f GB/s\n", ms_naive, calc_bandwidth(ms_naive));

    float ms_grid = time_kernel([&] {
        vector_add_grid_stride<<<persistent_blocks, block_size>>>(d_a, d_b, d_c, N);
    });
    printf("v2_grid_stride  : %6.3f ms | Bandwidth: %6.2f GB/s\n", ms_grid, calc_bandwidth(ms_grid));

    float ms_float4 = time_kernel([&] {
        vector_add_float4<<<persistent_blocks, block_size>>>(
            reinterpret_cast<const float4*>(d_a),
            reinterpret_cast<const float4*>(d_b),
            reinterpret_cast<float4*>(d_c),
            N / 4);
    });
    printf("v3_float4       : %6.3f ms | Bandwidth: %6.2f GB/s | Speedup: %.2fx\n",
           ms_float4, calc_bandwidth(ms_float4), ms_naive / ms_float4);

    // FP16 Benchmark
    half *d_ha, *d_hb, *d_hc;
    const size_t h_bytes = N * sizeof(half);
    CUDA_CHECK(cudaMalloc(&d_ha, h_bytes));
    CUDA_CHECK(cudaMalloc(&d_hb, h_bytes));
    CUDA_CHECK(cudaMalloc(&d_hc, h_bytes));
    CUDA_CHECK(cudaMemset(d_ha, 0, h_bytes));
    CUDA_CHECK(cudaMemset(d_hb, 0, h_bytes));

    float ms_half2 = time_kernel([&] {
        vector_add_half2<<<persistent_blocks, block_size>>>(
            reinterpret_cast<const half2*>(d_ha),
            reinterpret_cast<const half2*>(d_hb),
            reinterpret_cast<half2*>(d_hc),
            N / 2);
    });
    double bw_half = (3.0 * N * sizeof(half)) / (ms_half2 * 1e6);
    printf("v4_half2 (FP16) : %6.3f ms | Bandwidth: %6.2f GB/s\n", ms_half2, bw_half);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_ha));
    CUDA_CHECK(cudaFree(d_hb));
    CUDA_CHECK(cudaFree(d_hc));

    return 0;
}
