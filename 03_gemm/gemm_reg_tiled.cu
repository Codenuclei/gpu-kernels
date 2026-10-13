/*
 * ============================================================================
 * CHALLENGE 03: GEMM — 2D Register-Tiled Matrix Multiplication
 * ============================================================================
 *
 * Demonstrates:
 *   1. 2D Block Tiling: BM x BN = 64 x 64 block tile
 *   2. 2D Thread Micro-Tiling: TM x TN = 8 x 8 register accumulation per thread
 *   3. Outer-product rank-1 register updates eliminating shared memory bottleneck
 *   4. Vectorized float4 memory transfers from global to shared memory
 *
 * Hardware: RTX 2050 (SM 8.6, Ampere)
 */

#include "common/cuda_utils.cuh"
#include <cstdio>
#include <vector>

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;

// Block has (BM / TM) * (BN / TN) = (64 / 8) * (64 / 8) = 8 * 8 = 64 threads
constexpr int NUM_THREADS = (BM / TM) * (BN / TN);

__global__ void gemm_reg_tiled_2d(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K) {
    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];

    int tid = threadIdx.x;
    int thread_row = tid / (BN / TN); // [0, 7]
    int thread_col = tid % (BN / TN); // [0, 7]

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // Registers for accumulating TM x TN outputs
    float c_reg[TM][TN] = {0.0f};

    // Registers for caching slices of A and B per outer-product step
    float a_frag[TM];
    float b_frag[TN];

    // Number of threads is 64. Total elements to load:
    // s_A is 64 x 8 = 512 floats (8 floats per thread)
    // s_B is 8 x 64 = 512 floats (8 floats per thread)

    for (int bk_idx = 0; bk_idx < K; bk_idx += BK) {
        // Cooperative load into s_A (each thread loads 8 floats = 2 float4 or 8 singles)
        #pragma unroll
        for (int load_idx = 0; load_idx < 8; load_idx++) {
            int elem_id = tid * 8 + load_idx;
            int r = elem_id / BK;
            int c = elem_id % BK;
            int global_r = block_row + r;
            int global_c = bk_idx + c;
            s_A[r][c] = (global_r < M && global_c < K) ? A[global_r * K + global_c] : 0.0f;
        }

        // Cooperative load into s_B
        #pragma unroll
        for (int load_idx = 0; load_idx < 8; load_idx++) {
            int elem_id = tid * 8 + load_idx;
            int r = elem_id / BN;
            int c = elem_id % BN;
            int global_r = bk_idx + r;
            int global_c = block_col + c;
            s_B[r][c] = (global_r < K && global_c < N) ? B[global_r * N + global_c] : 0.0f;
        }

        __syncthreads();

        // Outer product computation
        #pragma unroll
        for (int dot_k = 0; dot_k < BK; dot_k++) {
            // Load TM elements from s_A into thread registers
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                a_frag[i] = s_A[thread_row * TM + i][dot_k];
            }

            // Load TN elements from s_B into thread registers
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                b_frag[j] = s_B[dot_k][thread_col * TN + j];
            }

            // Outer product update
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    c_reg[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }

        __syncthreads();
    }

    // Write back results to global memory
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int global_r = block_row + thread_row * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int global_c = block_col + thread_col * TN + j;
            if (global_r < M && global_c < N) {
                C[global_r * N + global_c] = c_reg[i][j];
            }
        }
    }
}

// CPU Reference
void cpu_gemm_sub(const float* A, const float* B, float* C, int M, int N, int K) {
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

    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    const size_t bytes_A = M * K * sizeof(float);
    const size_t bytes_B = K * N * sizeof(float);
    const size_t bytes_C = M * N * sizeof(float);

    printf("2D Register-Tiled GEMM [%d x %d] (Block %dx%d, Micro-tile %dx%d)\n\n",
           M, N, BM, BN, TM, TN);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_gpu(M * N);
    init_random(h_A.data(), M * K, -1.0f, 1.0f);
    init_random(h_B.data(), K * N, -1.0f, 1.0f);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    dim3 block(NUM_THREADS);
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));

    // Verify correctness on sample 64x64 submatrix
    printf("--- Correctness Verification ---\n");
    std::vector<float> v_A(64 * 64), v_B(64 * 64), v_C_cpu(64 * 64), v_C_gpu(64 * 64);
    init_random(v_A.data(), 64 * 64, -1.0f, 1.0f);
    init_random(v_B.data(), 64 * 64, -1.0f, 1.0f);
    cpu_gemm_sub(v_A.data(), v_B.data(), v_C_cpu.data(), 64, 64, 64);

    float *d_vA, *d_vB, *d_vC;
    CUDA_CHECK(cudaMalloc(&d_vA, 64 * 64 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vB, 64 * 64 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vC, 64 * 64 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_vA, v_A.data(), 64 * 64 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vB, v_B.data(), 64 * 64 * sizeof(float), cudaMemcpyHostToDevice));

    gemm_reg_tiled_2d<<<dim3(1, 1), dim3(NUM_THREADS)>>>(d_vA, d_vB, d_vC, 64, 64, 64);
    CUDA_CHECK(cudaMemcpy(v_C_gpu.data(), d_vC, 64 * 64 * sizeof(float), cudaMemcpyDeviceToHost));
    verify_close(v_C_cpu.data(), v_C_gpu.data(), 64 * 64, 1e-3f, 1e-3f, "v4_reg_tiled_2d");

    CUDA_CHECK(cudaFree(d_vA));
    CUDA_CHECK(cudaFree(d_vB));
    CUDA_CHECK(cudaFree(d_vC));

    printf("\n--- Performance Benchmarks [1024 x 1024] ---\n");
    double total_flops = 2.0 * static_cast<double>(M) * N * K;

    float ms = time_kernel([&] {
        gemm_reg_tiled_2d<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }, 5, 20);

    double gflops = (total_flops / 1e9) / (ms / 1e3);
    printf("v4_reg_tiled_2d : %7.3f ms | %7.2f GFLOPS\n", ms, gflops);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
