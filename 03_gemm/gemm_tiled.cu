/*
 * ============================================================================
 * CHALLENGE 03: GEMM — Naive to Shared Memory Tiled Matrix Multiplication
 * ============================================================================
 *
 * Demonstrates:
 *   1. Naive GEMM: O(MNK) global memory fetches, 0.25 FLOP/byte
 *   2. Shared Memory Tiled GEMM (16x16 and 32x32 tiles):
 *      Caches submatrices in SRAM, boosting arithmetic intensity by TILE factor
 *
 * Hardware: RTX 2050 (SM 8.6, Ampere)
 */

#include "common/cuda_utils.cuh"
#include <cstdio>
#include <vector>

// ----------------------------------------------------------------------------
// Version 1: Naive GEMM
// ----------------------------------------------------------------------------
__global__ void gemm_naive(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        #pragma unroll 4
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ----------------------------------------------------------------------------
// Version 2: Shared Memory Tiled GEMM (TILE_SIZE = 16)
// ----------------------------------------------------------------------------
template <int TILE_SIZE>
__global__ void gemm_tiled(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int N, int K) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;
    int num_tiles = ceil_div(K, TILE_SIZE);

    for (int t = 0; t < num_tiles; t++) {
        // Coalesced load of A and B tiles into shared memory
        int k_A = t * TILE_SIZE + threadIdx.x;
        int k_B = t * TILE_SIZE + threadIdx.y;

        s_A[threadIdx.y][threadIdx.x] = (row < M && k_A < K) ? A[row * K + k_A] : 0.0f;
        s_B[threadIdx.y][threadIdx.x] = (k_B < K && col < N) ? B[k_B * N + col] : 0.0f;

        __syncthreads();

        // Multiply sub-matrices from shared memory
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ----------------------------------------------------------------------------
// CPU Ground Truth Reference
// ----------------------------------------------------------------------------
void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    print_gpu_info();

    // 1024 x 1024 matrices (2 * 1024^3 = 2.15 GFLOPs)
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    const size_t bytes_A = M * K * sizeof(float);
    const size_t bytes_B = K * N * sizeof(float);
    const size_t bytes_C = M * N * sizeof(float);

    printf("GEMM Benchmark [%d x %d] * [%d x %d] = [%d x %d]\n", M, K, K, N, M, N);
    double total_flops = 2.0 * static_cast<double>(M) * N * K;
    printf("Total Computation: %.2f GFLOPs\n\n", total_flops / 1e9);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_cpu(M * N);
    std::vector<float> h_C_gpu(M * N);

    init_random(h_A.data(), M * K, -1.0f, 1.0f);
    init_random(h_B.data(), K * N, -1.0f, 1.0f);

    printf("Computing CPU reference on sample 128x128 submatrix for quick verification...\n");
    // Verify first 128x128 elements against CPU
    const int V_M = 128, V_N = 128, V_K = 128;
    std::vector<float> v_A(V_M * V_K), v_B(V_K * V_N), v_C_cpu(V_M * V_N), v_C_gpu(V_M * V_N);
    init_random(v_A.data(), V_M * V_K, -1.0f, 1.0f);
    init_random(v_B.data(), V_K * V_N, -1.0f, 1.0f);
    cpu_gemm(v_A.data(), v_B.data(), v_C_cpu.data(), V_M, V_N, V_K);

    float *d_vA, *d_vB, *d_vC;
    CUDA_CHECK(cudaMalloc(&d_vA, V_M * V_K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vB, V_K * V_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vC, V_M * V_N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_vA, v_A.data(), V_M * V_K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vB, v_B.data(), V_K * V_N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 v_block16(16, 16);
    dim3 v_grid16(ceil_div(V_N, 16), ceil_div(V_M, 16));

    gemm_naive<<<v_grid16, v_block16>>>(d_vA, d_vB, d_vC, V_M, V_N, V_K);
    CUDA_CHECK(cudaMemcpy(v_C_gpu.data(), d_vC, V_M * V_N * sizeof(float), cudaMemcpyDeviceToHost));
    verify_close(v_C_cpu.data(), v_C_gpu.data(), V_M * V_N, 1e-3f, 1e-3f, "v1_naive");

    gemm_tiled<16><<<v_grid16, v_block16>>>(d_vA, d_vB, d_vC, V_M, V_N, V_K);
    CUDA_CHECK(cudaMemcpy(v_C_gpu.data(), d_vC, V_M * V_N * sizeof(float), cudaMemcpyDeviceToHost));
    verify_close(v_C_cpu.data(), v_C_gpu.data(), V_M * V_N, 1e-3f, 1e-3f, "v2_tiled_16");

    CUDA_CHECK(cudaFree(d_vA));
    CUDA_CHECK(cudaFree(d_vB));
    CUDA_CHECK(cudaFree(d_vC));

    // Full 1024x1024 performance benchmark
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    printf("\n--- Performance Benchmarks [1024 x 1024] ---\n");
    auto calc_gflops = [&](float ms) {
        return (total_flops / 1e9) / (ms / 1e3);
    };

    dim3 block16(16, 16);
    dim3 grid16(ceil_div(N, 16), ceil_div(M, 16));

    dim3 block32(32, 32);
    dim3 grid32(ceil_div(N, 32), ceil_div(M, 32));

    float ms_naive = time_kernel([&] {
        gemm_naive<<<grid16, block16>>>(d_A, d_B, d_C, M, N, K);
    }, 2, 10);
    printf("v1_naive        : %7.3f ms | %6.2f GFLOPS\n", ms_naive, calc_gflops(ms_naive));

    float ms_tile16 = time_kernel([&] {
        gemm_tiled<16><<<grid16, block16>>>(d_A, d_B, d_C, M, N, K);
    }, 5, 20);
    printf("v2_tiled_16     : %7.3f ms | %6.2f GFLOPS | Speedup: %.2fx\n",
           ms_tile16, calc_gflops(ms_tile16), ms_naive / ms_tile16);

    float ms_tile32 = time_kernel([&] {
        gemm_tiled<32><<<grid32, block32>>>(d_A, d_B, d_C, M, N, K);
    }, 5, 20);
    printf("v3_tiled_32     : %7.3f ms | %6.2f GFLOPS | Speedup: %.2fx\n",
           ms_tile32, calc_gflops(ms_tile32), ms_naive / ms_tile32);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
