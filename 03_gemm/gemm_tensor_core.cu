/*
 * ============================================================================
 * CHALLENGE 03: GEMM — Ampere Tensor Core WMMA Matrix Multiplication
 * ============================================================================
 *
 * Demonstrates:
 *   1. Hardware Tensor Core acceleration on Ampere SM 8.6
 *   2. nvcuda::wmma API: 16x16x16 matrix multiply-accumulate fragments
 *   3. Mixed-precision: FP16 inputs with FP32 accumulator
 *   4. Multi-warp block decomposition (4 warps computing 32x32 tiles)
 *
 * Hardware: RTX 2050 (SM 8.6, Ampere 3rd-Gen Tensor Cores)
 */

#include "common/cuda_utils.cuh"
#include <cstdio>
#include <vector>
#include <mma.h>

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Block config: 2 warps along M, 2 warps along N -> 4 warps = 128 threads
// Block computes (2 * 16) x (2 * 16) = 32 x 32 elements of C
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 2;
constexpr int BLOCK_THREADS = (WARPS_M * WARPS_N) * WARP_SIZE; // 128 threads

__global__ void gemm_wmma_fp16(const half* __restrict__ A,
                               const half* __restrict__ B,
                               float* __restrict__ C,
                               int M, int N, int K) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_m = warp_id / WARPS_N;
    int warp_n = warp_id % WARPS_N;

    int c_row = blockIdx.y * (WARPS_M * WMMA_M) + warp_m * WMMA_M;
    int c_col = blockIdx.x * (WARPS_N * WMMA_N) + warp_n * WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        if (c_row < M && k < K) {
            wmma::load_matrix_sync(a_frag, A + c_row * K + k, K);
        }
        if (k < K && c_col < N) {
            wmma::load_matrix_sync(b_frag, B + k * N + c_col, N);
        }
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    if (c_row < M && c_col < N) {
        wmma::store_matrix_sync(C + c_row * N + c_col, c_frag, N, wmma::mem_row_major);
    }
}

// CPU Reference for verification
void cpu_gemm_fp16_ref(const half* A, const half* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    print_gpu_info();

    const int M = 2048;
    const int N = 2048;
    const int K = 2048;

    printf("Ampere Tensor Core WMMA GEMM [%d x %d x %d]\n\n", M, N, K);

    const size_t bytes_A = M * K * sizeof(half);
    const size_t bytes_B = K * N * sizeof(half);
    const size_t bytes_C = M * N * sizeof(float);

    std::vector<half> h_A(M * K);
    std::vector<half> h_B(K * N);
    std::vector<float> h_C(M * N);

    for (int i = 0; i < M * K; i++) {
        h_A[i] = __float2half((rand() % 100) / 100.0f - 0.5f);
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = __float2half((rand() % 100) / 100.0f - 0.5f);
    }

    // Ground truth verification on sample 32x32 block
    printf("--- Correctness Verification ---\n");
    std::vector<float> v_C_cpu(32 * 32);
    std::vector<float> v_C_gpu(32 * 32);
    cpu_gemm_fp16_ref(h_A.data(), h_B.data(), v_C_cpu.data(), 32, 32, 32);

    half *d_vA, *d_vB;
    float *d_vC;
    CUDA_CHECK(cudaMalloc(&d_vA, 32 * 32 * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_vB, 32 * 32 * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_vC, 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_vA, h_A.data(), 32 * 32 * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vB, h_B.data(), 32 * 32 * sizeof(half), cudaMemcpyHostToDevice));

    gemm_wmma_fp16<<<dim3(1, 1), dim3(BLOCK_THREADS)>>>(d_vA, d_vB, d_vC, 32, 32, 32);
    CUDA_CHECK(cudaMemcpy(v_C_gpu.data(), d_vC, 32 * 32 * sizeof(float), cudaMemcpyDeviceToHost));
    verify_close(v_C_cpu.data(), v_C_gpu.data(), 32 * 32, 1e-2f, 1e-2f, "WMMA Tensor Core");

    CUDA_CHECK(cudaFree(d_vA));
    CUDA_CHECK(cudaFree(d_vB));
    CUDA_CHECK(cudaFree(d_vC));

    // Full 2048 x 2048 benchmark
    half *d_A, *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_THREADS);
    dim3 grid(ceil_div(N, WARPS_N * WMMA_N), ceil_div(M, WARPS_M * WMMA_M));

    printf("\n--- Performance Benchmarks [2048 x 2048 x 2048] ---\n");
    double total_flops = 2.0 * static_cast<double>(M) * N * K;

    float ms = time_kernel([&] {
        gemm_wmma_fp16<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }, 5, 20);

    double tflops = (total_flops / 1e12) / (ms / 1e3);
    printf("WMMA Tensor Cores: %7.3f ms | %6.2f TFLOPS\n", ms, tflops);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
