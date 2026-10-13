/*
 * ============================================================================
 * CHALLENGE 04: ATTENTION — Multi-Head Attention (MHA) Baseline
 * ============================================================================
 *
 * Demonstrates:
 *   1. Standard 3-stage Attention pipeline:
 *      - Step 1: S = Q * K^T / sqrt(d)
 *      - Step 2: P = softmax(S)
 *      - Step 3: O = P * V
 *   2. Materialization of the O(N^2) attention score matrix in global HBM
 *   3. Why FlashAttention is necessary (memory explosion and bandwidth bottleneck)
 *
 * Hardware: RTX 2050 (SM 8.6, Ampere)
 */

#include "common/cuda_utils.cuh"
#include <cstdio>
#include <vector>
#include <cmath>

// ----------------------------------------------------------------------------
// Step 1: Q * K^T / sqrt(d) -> S [N x N] per head
// ----------------------------------------------------------------------------
__global__ void qk_gemm_kernel(const float* __restrict__ Q,
                               const float* __restrict__ K,
                               float* __restrict__ S,
                               int N, int d, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // query token idx
    int col = blockIdx.x * blockDim.x + threadIdx.x; // key token idx

    if (row < N && col < N) {
        float sum = 0.0f;
        #pragma unroll 4
        for (int k = 0; k < d; k++) {
            sum += Q[row * d + k] * K[col * d + k];
        }
        S[row * N + col] = sum * scale;
    }
}

// ----------------------------------------------------------------------------
// Step 2: Row-wise Softmax on S [N x N] -> P [N x N]
// ----------------------------------------------------------------------------
__global__ void attention_softmax_kernel(const float* __restrict__ S,
                                         float* __restrict__ P,
                                         int N) {
    int row = blockIdx.x; // 1 block per row
    int tid = threadIdx.x;

    const float* row_s = S + row * N;
    float* row_p = P + row * N;

    // 1. Find max across row
    float local_max = -1e30f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, row_s[i]);
    }
    local_max = warp_reduce_max(local_max);

    __shared__ float s_max[32];
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    if (lane == 0) s_max[wid] = local_max;
    __syncthreads();

    int num_warps = blockDim.x / WARP_SIZE;
    local_max = (tid < num_warps) ? s_max[lane] : -1e30f;
    if (wid == 0) local_max = warp_reduce_max(local_max);

    __shared__ float row_max_val;
    if (tid == 0) row_max_val = local_max;
    __syncthreads();
    float max_val = row_max_val;

    // 2. Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += __expf(row_s[i] - max_val);
    }
    local_sum = warp_reduce_sum(local_sum);

    __shared__ float s_sum[32];
    if (lane == 0) s_sum[wid] = local_sum;
    __syncthreads();

    local_sum = (tid < num_warps) ? s_sum[lane] : 0.0f;
    if (wid == 0) local_sum = warp_reduce_sum(local_sum);

    __shared__ float row_sum_val;
    if (tid == 0) row_sum_val = local_sum;
    __syncthreads();
    float inv_sum = 1.0f / (row_sum_val + 1e-6f);

    // 3. Write normalized probabilities
    for (int i = tid; i < N; i += blockDim.x) {
        row_p[i] = __expf(row_s[i] - max_val) * inv_sum;
    }
}

// ----------------------------------------------------------------------------
// Step 3: P * V -> O [N x d]
// ----------------------------------------------------------------------------
__global__ void pv_gemm_kernel(const float* __restrict__ P,
                               const float* __restrict__ V,
                               float* __restrict__ O,
                               int N, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // query idx
    int col = blockIdx.x * blockDim.x + threadIdx.x; // head dim idx

    if (row < N && col < d) {
        float sum = 0.0f;
        #pragma unroll 4
        for (int k = 0; k < N; k++) {
            sum += P[row * N + k] * V[k * d + col];
        }
        O[row * d + col] = sum;
    }
}

// ----------------------------------------------------------------------------
// CPU Ground Truth Reference
// ----------------------------------------------------------------------------
void cpu_attention(const float* Q, const float* K, const float* V,
                   float* O, int N, int d) {
    float scale = 1.0f / std::sqrt(static_cast<float>(d));
    std::vector<float> S(N * N);
    std::vector<float> P(N * N);

    // Step 1: Q * K^T * scale
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d; k++) {
                sum += Q[i * d + k] * K[j * d + k];
            }
            S[i * N + j] = sum * scale;
        }
    }

    // Step 2: Softmax(S) row-wise
    for (int i = 0; i < N; i++) {
        float max_val = -1e30f;
        for (int j = 0; j < N; j++) {
            if (S[i * N + j] > max_val) max_val = S[i * N + j];
        }
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            P[i * N + j] = std::exp(S[i * N + j] - max_val);
            sum += P[i * N + j];
        }
        for (int j = 0; j < N; j++) {
            P[i * N + j] /= sum;
        }
    }

    // Step 3: P * V
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += P[i * N + k] * V[k * d + j];
            }
            O[i * d + j] = sum;
        }
    }
}

int main() {
    print_gpu_info();

    const int N = 1024; // Sequence length
    const int d = 64;   // Head dimension
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    printf("Naive Multi-Head Attention Baseline [N=%d, d=%d]\n", N, d);
    printf("Intermediate Score Matrix P [N x N]: %.2f MB VRAM\n\n",
           (N * N * sizeof(float)) / (1024.0 * 1024.0));

    std::vector<float> h_Q(N * d), h_K(N * d), h_V(N * d);
    std::vector<float> h_O_cpu(N * d), h_O_gpu(N * d);

    init_random(h_Q.data(), N * d, -0.5f, 0.5f);
    init_random(h_K.data(), N * d, -0.5f, 0.5f);
    init_random(h_V.data(), N * d, -0.5f, 0.5f);

    cpu_attention(h_Q.data(), h_K.data(), h_V.data(), h_O_cpu.data(), N, d);

    float *d_Q, *d_K, *d_V, *d_S, *d_P, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, N * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, N * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, N * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_S, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_P, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, N * d * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), N * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), N * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), N * d * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block_qk(16, 16);
    dim3 grid_qk(ceil_div(N, 16), ceil_div(N, 16));

    dim3 block_pv(16, 16);
    dim3 grid_pv(ceil_div(d, 16), ceil_div(N, 16));

    auto run_mha = [&] {
        qk_gemm_kernel<<<grid_qk, block_qk>>>(d_Q, d_K, d_S, N, d, scale);
        attention_softmax_kernel<<<N, 256>>>(d_S, d_P, N);
        pv_gemm_kernel<<<grid_pv, block_pv>>>(d_P, d_V, d_O, N, d);
    };

    printf("--- Correctness Verification ---\n");
    run_mha();
    CUDA_CHECK(cudaMemcpy(h_O_gpu.data(), d_O, N * d * sizeof(float), cudaMemcpyDeviceToHost));
    verify_close(h_O_cpu.data(), h_O_gpu.data(), N * d, 1e-3f, 1e-3f, "Naive MHA Output");

    printf("\n--- Performance Benchmarks ---\n");
    float ms_qk = time_kernel([&] {
        qk_gemm_kernel<<<grid_qk, block_qk>>>(d_Q, d_K, d_S, N, d, scale);
    }, 5, 20);

    float ms_softmax = time_kernel([&] {
        attention_softmax_kernel<<<N, 256>>>(d_S, d_P, N);
    }, 5, 20);

    float ms_pv = time_kernel([&] {
        pv_gemm_kernel<<<grid_pv, block_pv>>>(d_P, d_V, d_O, N, d);
    }, 5, 20);

    float ms_total = time_kernel(run_mha, 5, 20);

    printf("1. Q*K^T GEMM    : %6.3f ms\n", ms_qk);
    printf("2. Global Softmax: %6.3f ms\n", ms_softmax);
    printf("3. P*V GEMM      : %6.3f ms\n", ms_pv);
    printf("------------------------------------\n");
    printf("Total Pipeline   : %6.3f ms\n", ms_total);

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_P));
    CUDA_CHECK(cudaFree(d_O));
    return 0;
}
