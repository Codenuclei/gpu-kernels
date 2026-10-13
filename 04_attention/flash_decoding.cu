/*
 * ============================================================================
 * CHALLENGE 04: ATTENTION — Flash-Decoding (Split-KV Inference Kernel)
 * ============================================================================
 *
 * Demonstrates:
 *   1. Architectural Split-KV parallelization for autoregressive decoding (q=1, long KV)
 *   2. Stage 1: Partition KV sequence across all 16 SMs, computing partial outputs, max, and sums
 *   3. Stage 2: Log-Sum-Exp reduction combining partial split states into final output
 *   4. Eliminates the batch=1 / q=1 GPU underutilization bottleneck
 *
 * Hardware: RTX 2050 (SM 8.6, 16 SMs, Ampere)
 */

#include "common/cuda_utils.cuh"
#include <cstdio>
#include <vector>
#include <cmath>

constexpr int HEAD_DIM = 64;
constexpr int NUM_SPLITS = 16; // One split per SM on RTX 2050

// ----------------------------------------------------------------------------
// Stage 1: Partial Split-KV Attention Kernel (q = 1)
// ----------------------------------------------------------------------------
__global__ void flash_decoding_stage1_kernel(const float* __restrict__ q,
                                             const float* __restrict__ K,
                                             const float* __restrict__ V,
                                             float* __restrict__ partial_O,
                                             float* __restrict__ partial_M,
                                             float* __restrict__ partial_L,
                                             int N, int d, float scale) {
    int split_idx = blockIdx.x; // [0, NUM_SPLITS - 1]
    int tid = threadIdx.x;

    int split_size = ceil_div(N, NUM_SPLITS);
    int kv_start = split_idx * split_size;
    int kv_end = min(kv_start + split_size, N);

    // Thread 0 in each block loads Query into shared memory
    __shared__ float s_q[HEAD_DIM];
    if (tid < HEAD_DIM) {
        s_q[tid] = q[tid];
    }
    __syncthreads();

    // Register accumulator per thread for partial output
    float local_o[HEAD_DIM] = {0.0f};
    float m_i = -1e30f;
    float l_i = 0.0f;

    // Process elements in this split
    for (int k_idx = kv_start + tid; k_idx < kv_end; k_idx += blockDim.x) {
        // Dot product q . K[k_idx]
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < HEAD_DIM; i++) {
            dot += s_q[i] * K[k_idx * d + i];
        }
        float score = dot * scale;

        // Online Softmax update
        float m_new = fmaxf(m_i, score);
        float alpha = __expf(m_i - m_new);
        float p = __expf(score - m_new);

        l_i = l_i * alpha + p;

        #pragma unroll
        for (int i = 0; i < HEAD_DIM; i++) {
            local_o[i] = local_o[i] * alpha + p * V[k_idx * d + i];
        }

        m_i = m_new;
    }

    // Inter-thread reduction across the block
    __shared__ float s_max[32];
    __shared__ float s_sum[32];
    __shared__ float s_out[32][HEAD_DIM];

    // Intra-warp reduction
    float warp_max = warp_reduce_max(m_i);
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    if (lane == 0) s_max[wid] = warp_max;
    __syncthreads();

    int num_warps = blockDim.x / WARP_SIZE;
    warp_max = (tid < num_warps) ? s_max[lane] : -1e30f;
    if (wid == 0) warp_max = warp_reduce_max(warp_max);

    __shared__ float block_max;
    if (tid == 0) block_max = warp_max;
    __syncthreads();

    float b_max = block_max;
    float rescale = __expf(m_i - b_max);
    l_i *= rescale;
    #pragma unroll
    for (int i = 0; i < HEAD_DIM; i++) {
        local_o[i] *= rescale;
    }

    // Reduce sum across threads
    float warp_sum = warp_reduce_sum(l_i);
    if (lane == 0) s_sum[wid] = warp_sum;
    __syncthreads();

    warp_sum = (tid < num_warps) ? s_sum[lane] : 0.0f;
    if (wid == 0) warp_sum = warp_reduce_sum(warp_sum);

    __shared__ float block_sum;
    if (tid == 0) block_sum = warp_sum;
    __syncthreads();

    // Reduce output vector across threads using atomicAdd to shared or direct write
    // For simplicity, thread 0 of block accumulates
    #pragma unroll
    for (int i = 0; i < HEAD_DIM; i++) {
        float sum_dim = warp_reduce_sum(local_o[i]);
        if (lane == 0) s_out[wid][i] = sum_dim;
    }
    __syncthreads();

    if (tid < HEAD_DIM) {
        float total_dim = 0.0f;
        for (int w = 0; w < num_warps; w++) {
            total_dim += s_out[w][tid];
        }
        partial_O[split_idx * d + tid] = total_dim;
    }

    if (tid == 0) {
        partial_M[split_idx] = b_max;
        partial_L[split_idx] = block_sum;
    }
}

// ----------------------------------------------------------------------------
// Stage 2: Log-Sum-Exp Reduction Kernel Combining Splits
// ----------------------------------------------------------------------------
__global__ void flash_decoding_stage2_kernel(const float* __restrict__ partial_O,
                                             const float* __restrict__ partial_M,
                                             const float* __restrict__ partial_L,
                                             float* __restrict__ out,
                                             int num_splits, int d) {
    int tid = threadIdx.x; // [0, 63]

    // 1. Find global max across splits
    float global_max = -1e30f;
    for (int s = 0; s < num_splits; s++) {
        if (partial_M[s] > global_max) {
            global_max = partial_M[s];
        }
    }

    // 2. Compute global normalizer sum
    float global_sum = 0.0f;
    for (int s = 0; s < num_splits; s++) {
        float weight = __expf(partial_M[s] - global_max);
        global_sum += partial_L[s] * weight;
    }

    // 3. Accumulate final output element
    if (tid < d) {
        float final_val = 0.0f;
        for (int s = 0; s < num_splits; s++) {
            float weight = __expf(partial_M[s] - global_max);
            final_val += partial_O[s * d + tid] * weight;
        }
        out[tid] = final_val / (global_sum > 0.0f ? global_sum : 1.0f);
    }
}

// ----------------------------------------------------------------------------
// CPU Ground Truth Reference
// ----------------------------------------------------------------------------
void cpu_decode_attention(const float* q, const float* K, const float* V,
                          float* out, int N, int d) {
    float scale = 1.0f / std::sqrt(static_cast<float>(d));
    std::vector<float> scores(N);

    float max_score = -1e30f;
    for (int i = 0; i < N; i++) {
        float dot = 0.0f;
        for (int k = 0; k < d; k++) {
            dot += q[k] * K[i * d + k];
        }
        scores[i] = dot * scale;
        if (scores[i] > max_score) max_score = scores[i];
    }

    float sum_exp = 0.0f;
    std::vector<float> probs(N);
    for (int i = 0; i < N; i++) {
        probs[i] = std::exp(scores[i] - max_score);
        sum_exp += probs[i];
    }

    for (int k = 0; k < d; k++) {
        float acc = 0.0f;
        for (int i = 0; i < N; i++) {
            acc += (probs[i] / sum_exp) * V[i * d + k];
        }
        out[k] = acc;
    }
}

int main() {
    print_gpu_info();

    const int N = 4096; // 4K Context Length
    const int d = HEAD_DIM;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    printf("Flash-Decoding (Split-KV) Benchmark [q=1, N=%d (4K), d=%d, Splits=%d]\n\n",
           N, d, NUM_SPLITS);

    std::vector<float> h_q(d);
    std::vector<float> h_K(N * d);
    std::vector<float> h_V(N * d);
    std::vector<float> h_out_cpu(d);
    std::vector<float> h_out_gpu(d);

    init_random(h_q.data(), d, -0.5f, 0.5f);
    init_random(h_K.data(), N * d, -0.5f, 0.5f);
    init_random(h_V.data(), N * d, -0.5f, 0.5f);

    cpu_decode_attention(h_q.data(), h_K.data(), h_V.data(), h_out_cpu.data(), N, d);

    float *d_q, *d_K, *d_V, *d_out;
    float *d_partial_O, *d_partial_M, *d_partial_L;

    CUDA_CHECK(cudaMalloc(&d_q, d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, N * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, N * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, d * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_partial_O, NUM_SPLITS * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial_M, NUM_SPLITS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial_L, NUM_SPLITS * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), N * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), N * d * sizeof(float), cudaMemcpyHostToDevice));

    auto run_flash_decoding = [&] {
        flash_decoding_stage1_kernel<<<NUM_SPLITS, 128>>>(
            d_q, d_K, d_V, d_partial_O, d_partial_M, d_partial_L, N, d, scale);
        flash_decoding_stage2_kernel<<<1, HEAD_DIM>>>(
            d_partial_O, d_partial_M, d_partial_L, d_out, NUM_SPLITS, d);
    };

    printf("--- Correctness Verification ---\n");
    run_flash_decoding();
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, d * sizeof(float), cudaMemcpyDeviceToHost));
    verify_close(h_out_cpu.data(), h_out_gpu.data(), d, 1e-3f, 1e-3f, "Flash-Decoding Output");

    printf("\n--- Performance Benchmarks ---\n");
    float ms_stage1 = time_kernel([&] {
        flash_decoding_stage1_kernel<<<NUM_SPLITS, 128>>>(
            d_q, d_K, d_V, d_partial_O, d_partial_M, d_partial_L, N, d, scale);
    }, 10, 50);

    float ms_stage2 = time_kernel([&] {
        flash_decoding_stage2_kernel<<<1, HEAD_DIM>>>(
            d_partial_O, d_partial_M, d_partial_L, d_out, NUM_SPLITS, d);
    }, 10, 50);

    float ms_total = time_kernel(run_flash_decoding, 10, 50);

    printf("Stage 1 (16-SM Split-KV) : %6.4f ms\n", ms_stage1);
    printf("Stage 2 (Log-Sum-Exp Red): %6.4f ms\n", ms_stage2);
    printf("Total Flash-Decoding Lat : %6.4f ms (%.1f microseconds)\n",
           ms_total, ms_total * 1000.0f);

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_partial_O));
    CUDA_CHECK(cudaFree(d_partial_M));
    CUDA_CHECK(cudaFree(d_partial_L));
    return 0;
}
