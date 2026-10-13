/*
 * ============================================================================
 * CHALLENGE 04: ATTENTION — Fused FlashAttention Forward Kernel
 * ============================================================================
 *
 * Demonstrates:
 *   1. Tri Dao's FlashAttention algorithm from first principles
 *   2. Online Softmax normalizer calculation in shared memory (SRAM)
 *   3. Zero global memory materialization of N x N attention matrix (O(N) memory)
 *   4. Causal masking support for autoregressive transformers
 *
 * Hardware: RTX 2050 (SM 8.6, Ampere)
 */

#include "common/cuda_utils.cuh"
#include <cstdio>
#include <vector>
#include <cmath>

constexpr int Bc = 32; // Key / Value block size
constexpr int Br = 32; // Query block size
constexpr int HEAD_DIM = 64;

// ----------------------------------------------------------------------------
// Fused FlashAttention Forward Kernel
// ----------------------------------------------------------------------------
template <bool IS_CAUSAL>
__global__ void flash_attention_forward_kernel(const float* __restrict__ Q,
                                              const float* __restrict__ K,
                                              const float* __restrict__ V,
                                              float* __restrict__ O,
                                              int N, int d, float scale) {
    // Each block processes one Br query tile
    int q_block_idx = blockIdx.x;
    int tid = threadIdx.x; // [0, Br - 1] = [0, 31]

    int q_row = q_block_idx * Br + tid;

    __shared__ float s_K[Bc][HEAD_DIM];
    __shared__ float s_V[Bc][HEAD_DIM];

    // Local registers per query token for online softmax and output accumulation
    float q_reg[HEAD_DIM];
    float o_reg[HEAD_DIM] = {0.0f};
    float m_i = -1e30f; // running max
    float l_i = 0.0f;   // running sum

    // Load Q row into registers
    if (q_row < N) {
        #pragma unroll
        for (int k = 0; k < HEAD_DIM; k++) {
            q_reg[k] = Q[q_row * d + k];
        }
    } else {
        #pragma unroll
        for (int k = 0; k < HEAD_DIM; k++) {
            q_reg[k] = 0.0f;
        }
    }

    int num_kv_blocks = ceil_div(N, Bc);

    // Loop over Key/Value blocks
    for (int kv_idx = 0; kv_idx < num_kv_blocks; kv_idx++) {
        // Cooperative load of K and V into shared memory
        // Thread `tid` loads row `tid` of K and V
        int kv_row = kv_idx * Bc + tid;
        #pragma unroll
        for (int k = 0; k < HEAD_DIM; k++) {
            s_K[tid][k] = (kv_row < N) ? K[kv_row * d + k] : 0.0f;
            s_V[tid][k] = (kv_row < N) ? V[kv_row * d + k] : 0.0f;
        }
        __syncthreads();

        if (q_row < N) {
            // Compute S_ij = Q_i * K_j^T * scale for all j in Bc
            float s_ij[Bc];
            float local_max = -1e30f;

            #pragma unroll
            for (int j = 0; j < Bc; j++) {
                int global_kv_row = kv_idx * Bc + j;
                if (global_kv_row < N) {
                    if (IS_CAUSAL && global_kv_row > q_row) {
                        s_ij[j] = -1e30f;
                    } else {
                        float dot = 0.0f;
                        #pragma unroll
                        for (int k = 0; k < HEAD_DIM; k++) {
                            dot += q_reg[k] * s_K[j][k];
                        }
                        s_ij[j] = dot * scale;
                        if (s_ij[j] > local_max) local_max = s_ij[j];
                    }
                } else {
                    s_ij[j] = -1e30f;
                }
            }

            // Online Softmax update
            float m_new = fmaxf(m_i, local_max);
            float alpha = __expf(m_i - m_new);

            // Rescale running sum
            float sum_new = l_i * alpha;

            // Compute exp(S_ij - m_new) and accumulate
            float p_ij[Bc];
            #pragma unroll
            for (int j = 0; j < Bc; j++) {
                if (s_ij[j] > -1e20f) {
                    p_ij[j] = __expf(s_ij[j] - m_new);
                    sum_new += p_ij[j];
                } else {
                    p_ij[j] = 0.0f;
                }
            }

            // Rescale previous accumulator: O_new = O_old * alpha + P_ij * V_j
            #pragma unroll
            for (int k = 0; k < HEAD_DIM; k++) {
                float pv_term = 0.0f;
                #pragma unroll
                for (int j = 0; j < Bc; j++) {
                    pv_term += p_ij[j] * s_V[j][k];
                }
                o_reg[k] = o_reg[k] * alpha + pv_term;
            }

            // Update running statistics
            m_i = m_new;
            l_i = sum_new;
        }

        __syncthreads();
    }

    // Final normalization: O = O / l_i
    if (q_row < N && l_i > 0.0f) {
        float inv_l = 1.0f / l_i;
        #pragma unroll
        for (int k = 0; k < HEAD_DIM; k++) {
            O[q_row * d + k] = o_reg[k] * inv_l;
        }
    }
}

// ----------------------------------------------------------------------------
// CPU Ground Truth Reference
// ----------------------------------------------------------------------------
void cpu_attention_ref(const float* Q, const float* K, const float* V,
                       float* O, int N, int d, bool causal) {
    float scale = 1.0f / std::sqrt(static_cast<float>(d));
    std::vector<float> S(N * N);
    std::vector<float> P(N * N);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (causal && j > i) {
                S[i * N + j] = -1e30f;
            } else {
                float sum = 0.0f;
                for (int k = 0; k < d; k++) {
                    sum += Q[i * d + k] * K[j * d + k];
                }
                S[i * N + j] = sum * scale;
            }
        }
    }

    for (int i = 0; i < N; i++) {
        float max_val = -1e30f;
        for (int j = 0; j < N; j++) {
            if (S[i * N + j] > max_val) max_val = S[i * N + j];
        }
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            if (S[i * N + j] > -1e20f) {
                P[i * N + j] = std::exp(S[i * N + j] - max_val);
                sum += P[i * N + j];
            } else {
                P[i * N + j] = 0.0f;
            }
        }
        float inv_sum = 1.0f / (sum > 0.0f ? sum : 1.0f);
        for (int j = 0; j < N; j++) {
            P[i * N + j] *= inv_sum;
        }
    }

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
    const int d = HEAD_DIM;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    printf("Fused FlashAttention Forward Benchmark [N=%d, d=%d]\n", N, d);
    printf("Memory Complexity: O(N) = %.2f KB vs Naive O(N^2) = %.2f MB\n\n",
           (N * d * sizeof(float)) / 1024.0,
           (N * N * sizeof(float)) / (1024.0 * 1024.0));

    std::vector<float> h_Q(N * d), h_K(N * d), h_V(N * d);
    std::vector<float> h_O_cpu(N * d), h_O_gpu(N * d);

    init_random(h_Q.data(), N * d, -0.5f, 0.5f);
    init_random(h_K.data(), N * d, -0.5f, 0.5f);
    init_random(h_V.data(), N * d, -0.5f, 0.5f);

    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, N * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, N * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, N * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, N * d * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), N * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), N * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), N * d * sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid(ceil_div(N, Br));
    dim3 block(Br);

    printf("--- Verification (Standard & Causal) ---\n");

    // Standard Non-Causal
    cpu_attention_ref(h_Q.data(), h_K.data(), h_V.data(), h_O_cpu.data(), N, d, false);
    flash_attention_forward_kernel<false><<<grid, block>>>(d_Q, d_K, d_V, d_O, N, d, scale);
    CUDA_CHECK(cudaMemcpy(h_O_gpu.data(), d_O, N * d * sizeof(float), cudaMemcpyDeviceToHost));
    verify_close(h_O_cpu.data(), h_O_gpu.data(), N * d, 1e-3f, 1e-3f, "FlashAttention (Non-Causal)");

    // Causal Masking
    cpu_attention_ref(h_Q.data(), h_K.data(), h_V.data(), h_O_cpu.data(), N, d, true);
    flash_attention_forward_kernel<true><<<grid, block>>>(d_Q, d_K, d_V, d_O, N, d, scale);
    CUDA_CHECK(cudaMemcpy(h_O_gpu.data(), d_O, N * d * sizeof(float), cudaMemcpyDeviceToHost));
    verify_close(h_O_cpu.data(), h_O_gpu.data(), N * d, 1e-3f, 1e-3f, "FlashAttention (Causal)");

    printf("\n--- Performance Benchmarks ---\n");
    float ms_non_causal = time_kernel([&] {
        flash_attention_forward_kernel<false><<<grid, block>>>(d_Q, d_K, d_V, d_O, N, d, scale);
    }, 5, 20);

    float ms_causal = time_kernel([&] {
        flash_attention_forward_kernel<true><<<grid, block>>>(d_Q, d_K, d_V, d_O, N, d, scale);
    }, 5, 20);

    printf("FlashAttention (Non-Causal) : %6.3f ms\n", ms_non_causal);
    printf("FlashAttention (Causal)     : %6.3f ms\n", ms_causal);

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    return 0;
}
