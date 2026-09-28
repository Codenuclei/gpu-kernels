/*
 * ============================================================================
 * CHALLENGE 01: ONLINE SOFTMAX — The Gateway Kernel
 * ============================================================================
 *
 * WHY THIS KERNEL:
 *   Softmax is inside every single attention layer of every transformer.
 *   It's the first non-trivial kernel every inference engineer writes.
 *   It touches: reduction, numerical stability, memory access patterns,
 *   and the "online algorithm" trick that FlashAttention is built on.
 *
 * THE MATH:
 *   softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 *
 *   Naive: 3 passes over data (find max, compute exp+sum, normalize)
 *   Online: 1 pass (track running max + running sum simultaneously)
 *   See: "Online Normalizer Calculation for Softmax" (Milakov & Gimelshein, 2018)
 *         https://arxiv.org/abs/1805.02867
 *
 * YOUR MISSION:
 *   Implement 3 versions of row-wise softmax on a matrix [N x D]:
 *   - Each row is independently softmaxed
 *   - N = number of rows (batch dimension)
 *   - D = row length (vocabulary / sequence dimension)
 *
 * VERSION 1 (v1_naive):
 *   - 3-pass: one kernel to find row max, one to compute exp(x-max),
 *     one to normalize by sum
 *   - OR: single kernel with 3 sequential loops
 *   - Just make it correct. Don't optimize yet.
 *
 * VERSION 2 (v2_shared_mem):
 *   - 1 block per row
 *   - Use shared memory for the reduction (max + sum)
 *   - 2 passes: pass 1 computes max via shared mem reduction,
 *     pass 2 computes exp+sum+normalize
 *
 * VERSION 3 (v3_online):
 *   - 1 block per row
 *   - TRUE single-pass online softmax (the Milakov-Gimelshein algorithm)
 *   - Track running (max, sum) simultaneously using the correction factor
 *   - The key insight: when you find a new max, you can rescale the
 *     running sum: sum_new = sum_old * exp(old_max - new_max) + exp(x - new_max)
 *
 * BENCHMARK TARGETS (RTX 2050, FP32, D=4096, N=1024):
 *   - v1 should work (correctness is the goal)
 *   - v2 should be ~2-3x faster than v1
 *   - v3 should be ~1.5-2x faster than v2 (fewer global memory passes)
 *   - Compare all against PyTorch: torch.softmax(x, dim=-1)
 *
 * WHAT YOU'LL LEARN:
 *   ✓ Parallel reduction patterns (warp shuffle + shared memory)
 *   ✓ Numerical stability (the max-subtraction trick)
 *   ✓ Online algorithms (the foundation of FlashAttention)
 *   ✓ Memory bandwidth analysis (are you compute or memory bound?)
 *   ✓ Profiling with ncu
 *
 * VALIDATION:
 *   Compare output against CPU reference. Max absolute error < 1e-5.
 *
 * COMPILE:
 *   nvcc -O3 -arch=sm_86 -o softmax softmax.cu
 *
 * PROFILE:
 *   ncu --set full ./softmax
 *
 * ============================================================================
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include "../common/cuda_utils.cuh"

// ─── CPU REFERENCE (your correctness oracle) ────────────────────────────────

void softmax_cpu(const float* input, float* output, int N, int D) {
    for (int row = 0; row < N; row++) {
        const float* x = input + row * D;
        float* y = output + row * D;

        // Find max
        float max_val = -FLT_MAX;
        for (int j = 0; j < D; j++) {
            if (x[j] > max_val) max_val = x[j];
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int j = 0; j < D; j++) {
            y[j] = expf(x[j] - max_val);
            sum += y[j];
        }

        // Normalize
        for (int j = 0; j < D; j++) {
            y[j] /= sum;
        }
    }
}

// ─── YOUR KERNELS GO HERE ───────────────────────────────────────────────────

// VERSION 1: Naive 3-pass softmax
// One thread per element, use atomics or multiple kernel launches
// HINT: Start simple. Even 1 thread per row works. Just get it correct.
__global__ void softmax_v1_naive(const float* input, float* output, int N, int D) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    const float* x = input + row * D;
    float* y = output + row * D;

    float max_val = -FLT_MAX;
    for (int col = 0; col < D; col++) max_val = fmaxf(max_val, x[col]);
    float sum = 0.0f;
    for (int col = 0; col < D; col++) sum += expf(x[col] - max_val);
    for (int col = 0; col < D; col++) y[col] = expf(x[col] - max_val) / sum;
}

// VERSION 2: Shared memory reduction
// One block per row. Use shared memory to cooperatively find max and sum.
// HINT: Look at the parallel reduction slides from NVIDIA:
//   https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
__global__ void softmax_v2_shared(const float* input, float* output, int N, int D) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= N) return;
    const float* x = input + row * D;
    float* y = output + row * D;
    extern __shared__ float smem[];

    float thread_max = -FLT_MAX;
    for (int col = tid; col < D; col += blockDim.x)
        thread_max = fmaxf(thread_max, x[col]);
    smem[tid] = thread_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        __syncthreads();
    }
    float max_val = smem[0];

    float thread_sum = 0.0f;
    for (int col = tid; col < D; col += blockDim.x)
        thread_sum += expf(x[col] - max_val);
    smem[tid] = thread_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    float sum = smem[0];
    for (int col = tid; col < D; col += blockDim.x)
        y[col] = expf(x[col] - max_val) / sum;
}

// VERSION 3: Online softmax (single pass)
// One block per row. Track (running_max, running_sum) simultaneously.
// HINT: The key recurrence from the paper:
//   When processing element x:
//     if x > running_max:
//         running_sum = running_sum * exp(running_max - x) + 1.0
//         running_max = x
//     else:
//         running_sum += exp(x - running_max)
//   Then reduce (max, sum) pairs across threads using the same correction.
__global__ void softmax_v3_online(const float* input, float* output, int N, int D) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= N) return;
    const float* x = input + row * D;
    float* y = output + row * D;
    __shared__ float block_max[256];
    __shared__ float block_sum[256];

    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    for (int col = tid; col < D; col += blockDim.x) {
        float value = x[col];
        if (value > local_max) {
            local_sum = local_sum * expf(local_max - value) + 1.0f;
            local_max = value;
        } else {
            local_sum += expf(value - local_max);
        }
    }
    block_max[tid] = local_max;
    block_sum[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_max = block_max[tid + stride];
            float other_sum = block_sum[tid + stride];
            float merged_max = fmaxf(block_max[tid], other_max);
            block_sum[tid] =
                block_sum[tid] * expf(block_max[tid] - merged_max) +
                other_sum * expf(other_max - merged_max);
            block_max[tid] = merged_max;
        }
        __syncthreads();
    }
    float max_val = block_max[0];
    float sum = block_sum[0];
    for (int col = tid; col < D; col += blockDim.x)
        y[col] = expf(x[col] - max_val) / sum;
}

// ─── VALIDATION ─────────────────────────────────────────────────────────────

bool validate(const float* gpu_out, const float* cpu_out, int N, int D,
              const char* name) {
    float max_err = 0.0f;
    for (int i = 0; i < N * D; i++) {
        float err = fabsf(gpu_out[i] - cpu_out[i]);
        if (err > max_err) max_err = err;
    }
    bool pass = max_err < 1e-5f;
    printf("  %-20s  max_err=%.2e  %s\n", name, max_err,
           pass ? "✅ PASS" : "❌ FAIL");
    return pass;
}

// ─── MAIN ───────────────────────────────────────────────────────────────────

int main() {
    print_gpu_info();

    // Problem size — these are realistic for inference
    const int N = 1024;   // batch of rows
    const int D = 4096;   // row length (e.g., vocab projection, attention scores)
    const size_t bytes = N * D * sizeof(float);

    printf("Softmax Challenge: N=%d rows, D=%d cols (%.1f MB)\n\n", N, D,
           bytes / (1024.0f * 1024.0f));

    // Allocate host
    float* h_input = (float*)malloc(bytes);
    float* h_cpu_out = (float*)malloc(bytes);
    float* h_gpu_out = (float*)malloc(bytes);

    // Random input (use values that test numerical stability)
    srand(42);
    for (int i = 0; i < N * D; i++) {
        h_input[i] = ((float)rand() / RAND_MAX - 0.5f) * 20.0f;  // range [-10, 10]
    }

    // CPU reference
    softmax_cpu(h_input, h_cpu_out, N, D);

    // Allocate device
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    printf("─── Correctness ───\n");

    // ── V1: Naive ──
    softmax_v1_naive<<<(N+255)/256, 256>>>(d_input, d_output, N, D);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_gpu_out, d_output, bytes, cudaMemcpyDeviceToHost));
    validate(h_gpu_out, h_cpu_out, N, D, "v1_naive");

    // ── V2: Shared Memory ──
    int block_size = 256;
    int smem_bytes = block_size * sizeof(float);
    softmax_v2_shared<<<N, block_size, smem_bytes>>>(d_input, d_output, N, D);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_gpu_out, d_output, bytes, cudaMemcpyDeviceToHost));
    validate(h_gpu_out, h_cpu_out, N, D, "v2_shared");

    // ── V3: Online ──
    softmax_v3_online<<<N, 256>>>(d_input, d_output, N, D);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_gpu_out, d_output, bytes, cudaMemcpyDeviceToHost));
    validate(h_gpu_out, h_cpu_out, N, D, "v3_online");

    printf("\n─── Benchmarks ───\n");

    float ms_v1 = time_kernel([&]{
        softmax_v1_naive<<<(N+255)/256, 256>>>(d_input, d_output, N, D);
    });
    printf("  v1_naive:   %8.3f ms  (%6.1f GB/s)\n", ms_v1,
           2.0*bytes/ms_v1/1e6);

    float ms_v2 = time_kernel([&]{
        softmax_v2_shared<<<N, 256, 256*sizeof(float)>>>(d_input, d_output, N, D);
    });
    printf("  v2_shared:  %8.3f ms  (%6.1f GB/s)\n", ms_v2,
           2.0*bytes/ms_v2/1e6);

    float ms_v3 = time_kernel([&]{
        softmax_v3_online<<<N, 256>>>(d_input, d_output, N, D);
    });
    printf("  v3_online:  %8.3f ms  (%6.1f GB/s)\n", ms_v3,
           2.0*bytes/ms_v3/1e6);

    printf("\n  Speedup v2/v1: %.2fx\n", ms_v1/ms_v2);
    printf("  Speedup v3/v1: %.2fx\n", ms_v1/ms_v3);

    printf("\n  RTX 2050 theoretical peak: ~224 GB/s\n");
    printf("  Your v3 achieves: %.1f%% of peak\n",
           (2.0*bytes/ms_v3/1e6)/224.0*100.0);

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_cpu_out);
    free(h_gpu_out);

    return 0;
}
