/*
 * ============================================================================
 * CHALLENGE 02: REDUCE — Parallel Reduction Hierarchy (7 Stages)
 * ============================================================================
 *
 * Demonstrates:
 *   Stage 1: Interleaved addressing with divergent branching
 *   Stage 2: Contiguous threads (resolves warp divergence)
 *   Stage 3: Sequential addressing (resolves shared memory bank conflicts)
 *   Stage 4: First add during global memory load (halves block count)
 *   Stage 5: Warp synchronous reduction (unrolling last warp)
 *   Stage 6: Template-based complete loop unrolling
 *   Stage 7: Hardware Warp Shuffle (__shfl_down_sync) bypassing shared memory
 *
 * Hardware: RTX 2050 (SM 8.6, Ampere)
 */

#include "common/cuda_utils.cuh"
#include <cstdio>
#include <vector>

// ----------------------------------------------------------------------------
// Stage 1: Interleaved Addressing with Divergent Branching
// ----------------------------------------------------------------------------
__global__ void reduce_v1_interleaved_divergent(const float* __restrict__ g_idata,
                                                float* __restrict__ g_odata,
                                                int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();

    // Highly divergent branching within warps: threadIdx.x % (2*stride) == 0
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) atomicAdd(g_odata, sdata[0]);
}

// ----------------------------------------------------------------------------
// Stage 2: Contiguous Threads (Resolves Branch Divergence)
// ----------------------------------------------------------------------------
__global__ void reduce_v2_contiguous(const float* __restrict__ g_idata,
                                     float* __restrict__ g_odata,
                                     int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();

    // Contiguous threads active, but suffers shared memory bank conflicts
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    if (tid == 0) atomicAdd(g_odata, sdata[0]);
}

// ----------------------------------------------------------------------------
// Stage 3: Sequential Addressing (Resolves Bank Conflicts)
// ----------------------------------------------------------------------------
__global__ void reduce_v3_sequential(const float* __restrict__ g_idata,
                                     float* __restrict__ g_odata,
                                     int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();

    // Sequential addressing avoids bank conflicts
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) atomicAdd(g_odata, sdata[0]);
}

// ----------------------------------------------------------------------------
// Stage 4: First Add During Global Memory Load
// ----------------------------------------------------------------------------
__global__ void reduce_v4_first_add_during_load(const float* __restrict__ g_idata,
                                                float* __restrict__ g_odata,
                                                int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Load two elements and perform first add
    float my_sum = (i < n) ? g_idata[i] : 0.0f;
    if (i + blockDim.x < n) {
        my_sum += g_idata[i + blockDim.x];
    }
    sdata[tid] = my_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) atomicAdd(g_odata, sdata[0]);
}

// ----------------------------------------------------------------------------
// Stage 5: Warp Synchronous (Unrolling Last Warp)
// ----------------------------------------------------------------------------
__device__ void warp_reduce_volatile(volatile float* sdata, int tid) {
    sdata[tid] = sdata[tid] + sdata[tid + 32];
    sdata[tid] = sdata[tid] + sdata[tid + 16];
    sdata[tid] = sdata[tid] + sdata[tid + 8];
    sdata[tid] = sdata[tid] + sdata[tid + 4];
    sdata[tid] = sdata[tid] + sdata[tid + 2];
    sdata[tid] = sdata[tid] + sdata[tid + 1];
}

__global__ void reduce_v5_warp_unroll(const float* __restrict__ g_idata,
                                      float* __restrict__ g_odata,
                                      int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float my_sum = (i < n) ? g_idata[i] : 0.0f;
    if (i + blockDim.x < n) {
        my_sum += g_idata[i + blockDim.x];
    }
    sdata[tid] = my_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warp_reduce_volatile(sdata, tid);
    }

    if (tid == 0) atomicAdd(g_odata, sdata[0]);
}

// ----------------------------------------------------------------------------
// Stage 6: Template-based Complete Unrolling
// ----------------------------------------------------------------------------
template <unsigned int blockSize>
__global__ void reduce_v6_template_unroll(const float* __restrict__ g_idata,
                                          float* __restrict__ g_odata,
                                          int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;

    float my_sum = (i < n) ? g_idata[i] : 0.0f;
    if (i + blockSize < n) {
        my_sum += g_idata[i + blockSize];
    }
    sdata[tid] = my_sum;
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32) {
        warp_reduce_volatile(sdata, tid);
    }

    if (tid == 0) atomicAdd(g_odata, sdata[0]);
}

// ----------------------------------------------------------------------------
// Stage 7: Hardware Warp Shuffle (__shfl_down_sync)
// ----------------------------------------------------------------------------
template <unsigned int blockSize>
__global__ void reduce_v7_warp_shuffle(const float* __restrict__ g_idata,
                                       float* __restrict__ g_odata,
                                       int n) {
    // Warp-level registers only — shared memory only for inter-warp reduction (32 floats)
    __shared__ float s_warp_sums[32];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;

    float my_sum = (i < n) ? g_idata[i] : 0.0f;
    if (i + blockSize < n) {
        my_sum += g_idata[i + blockSize];
    }

    // Intra-warp shuffle reduction
    my_sum = warp_reduce_sum(my_sum);

    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    if (lane == 0) {
        s_warp_sums[wid] = my_sum;
    }
    __syncthreads();

    // First warp aggregates inter-warp sums
    int num_warps = blockSize / WARP_SIZE;
    my_sum = (tid < num_warps) ? s_warp_sums[lane] : 0.0f;
    if (wid == 0) {
        my_sum = warp_reduce_sum(my_sum);
    }

    if (tid == 0) {
        atomicAdd(g_odata, my_sum);
    }
}

// ----------------------------------------------------------------------------
// CPU Ground Truth
// ----------------------------------------------------------------------------
double cpu_reduce(const float* data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += static_cast<double>(data[i]);
    }
    return sum;
}

int main() {
    print_gpu_info();

    const int N = 1 << 24; // 16M floats = 64 MB
    const size_t bytes = N * sizeof(float);
    printf("Parallel Reduction Benchmark on N = %d elements (%.2f MB)\n\n",
           N, bytes / (1024.0 * 1024.0));

    std::vector<float> h_data(N);
    // Use small values to prevent catastrophic floating point cancellation
    init_random(h_data.data(), N, 0.0f, 1.0f);
    double ref_sum = cpu_reduce(h_data.data(), N);
    printf("CPU Ground-Truth Sum: %.6f\n\n", ref_sum);

    float *d_idata, *d_odata;
    CUDA_CHECK(cudaMalloc(&d_idata, bytes));
    CUDA_CHECK(cudaMalloc(&d_odata, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_idata, h_data.data(), bytes, cudaMemcpyHostToDevice));

    const int block_size = 256;
    const int grid_single = ceil_div(N, block_size);
    const int grid_double = ceil_div(N, block_size * 2);

    auto verify_result = [&](const char* name) {
        float h_res = 0.0f;
        CUDA_CHECK(cudaMemcpy(&h_res, d_odata, sizeof(float), cudaMemcpyDeviceToHost));
        float diff = std::fabs(static_cast<float>(ref_sum) - h_res);
        float rel_err = diff / static_cast<float>(ref_sum);
        if (rel_err < 1e-4f) {
            printf("  [PASS] %-25s: sum = %.6f (rel_err = %.2e)\n", name, h_res, rel_err);
            return true;
        } else {
            printf("  [FAIL] %-25s: sum = %.6f vs ref = %.6f (rel_err = %.2e)\n", name, h_res, ref_sum, rel_err);
            return false;
        }
    };

    printf("--- Correctness Verification ---\n");

    CUDA_CHECK(cudaMemset(d_odata, 0, sizeof(float)));
    reduce_v1_interleaved_divergent<<<grid_single, block_size, block_size * sizeof(float)>>>(d_idata, d_odata, N);
    verify_result("Stage 1 (Interleaved)");

    CUDA_CHECK(cudaMemset(d_odata, 0, sizeof(float)));
    reduce_v2_contiguous<<<grid_single, block_size, block_size * sizeof(float)>>>(d_idata, d_odata, N);
    verify_result("Stage 2 (Contiguous)");

    CUDA_CHECK(cudaMemset(d_odata, 0, sizeof(float)));
    reduce_v3_sequential<<<grid_single, block_size, block_size * sizeof(float)>>>(d_idata, d_odata, N);
    verify_result("Stage 3 (Sequential)");

    CUDA_CHECK(cudaMemset(d_odata, 0, sizeof(float)));
    reduce_v4_first_add_during_load<<<grid_double, block_size, block_size * sizeof(float)>>>(d_idata, d_odata, N);
    verify_result("Stage 4 (Add on Load)");

    CUDA_CHECK(cudaMemset(d_odata, 0, sizeof(float)));
    reduce_v5_warp_unroll<<<grid_double, block_size, block_size * sizeof(float)>>>(d_idata, d_odata, N);
    verify_result("Stage 5 (Warp Unroll)");

    CUDA_CHECK(cudaMemset(d_odata, 0, sizeof(float)));
    reduce_v6_template_unroll<block_size><<<grid_double, block_size, block_size * sizeof(float)>>>(d_idata, d_odata, N);
    verify_result("Stage 6 (Template Unroll)");

    CUDA_CHECK(cudaMemset(d_odata, 0, sizeof(float)));
    reduce_v7_warp_shuffle<block_size><<<grid_double, block_size>>>(d_idata, d_odata, N);
    verify_result("Stage 7 (Warp Shuffle)");

    printf("\n--- Performance Benchmarks ---\n");
    auto calc_bw = [&](float ms) {
        return bytes / (ms * 1e6); // Read GB/s
    };

    float ms1 = time_kernel([&] {
        CUDA_CHECK(cudaMemset(d_odata, 0, sizeof(float)));
        reduce_v1_interleaved_divergent<<<grid_single, block_size, block_size * sizeof(float)>>>(d_idata, d_odata, N);
    });

    float ms2 = time_kernel([&] {
        CUDA_CHECK(cudaMemset(d_odata, 0, sizeof(float)));
        reduce_v2_contiguous<<<grid_single, block_size, block_size * sizeof(float)>>>(d_idata, d_odata, N);
    });

    float ms3 = time_kernel([&] {
        CUDA_CHECK(cudaMemset(d_odata, 0, sizeof(float)));
        reduce_v3_sequential<<<grid_single, block_size, block_size * sizeof(float)>>>(d_idata, d_odata, N);
    });

    float ms4 = time_kernel([&] {
        CUDA_CHECK(cudaMemset(d_odata, 0, sizeof(float)));
        reduce_v4_first_add_during_load<<<grid_double, block_size, block_size * sizeof(float)>>>(d_idata, d_odata, N);
    });

    float ms5 = time_kernel([&] {
        CUDA_CHECK(cudaMemset(d_odata, 0, sizeof(float)));
        reduce_v5_warp_unroll<<<grid_double, block_size, block_size * sizeof(float)>>>(d_idata, d_odata, N);
    });

    float ms6 = time_kernel([&] {
        CUDA_CHECK(cudaMemset(d_odata, 0, sizeof(float)));
        reduce_v6_template_unroll<block_size><<<grid_double, block_size, block_size * sizeof(float)>>>(d_idata, d_odata, N);
    });

    float ms7 = time_kernel([&] {
        CUDA_CHECK(cudaMemset(d_odata, 0, sizeof(float)));
        reduce_v7_warp_shuffle<block_size><<<grid_double, block_size>>>(d_idata, d_odata, N);
    });

    printf("========================================================================\n");
    printf("Kernel                          | Latency (ms) | Bandwidth | Speedup\n");
    printf("------------------------------------------------------------------------\n");
    printf("Stage 1 (Interleaved Divergent) |   %7.3f ms | %5.1f GB/s |  1.00x\n", ms1, calc_bw(ms1));
    printf("Stage 2 (Contiguous Threads)    |   %7.3f ms | %5.1f GB/s |  %5.2fx\n", ms2, calc_bw(ms2), ms1 / ms2);
    printf("Stage 3 (Sequential Addressing) |   %7.3f ms | %5.1f GB/s |  %5.2fx\n", ms3, calc_bw(ms3), ms1 / ms3);
    printf("Stage 4 (Add on Global Load)    |   %7.3f ms | %5.1f GB/s |  %5.2fx\n", ms4, calc_bw(ms4), ms1 / ms4);
    printf("Stage 5 (Warp Unrolling)        |   %7.3f ms | %5.1f GB/s |  %5.2fx\n", ms5, calc_bw(ms5), ms1 / ms5);
    printf("Stage 6 (Template Full Unroll)  |   %7.3f ms | %5.1f GB/s |  %5.2fx\n", ms6, calc_bw(ms6), ms1 / ms6);
    printf("Stage 7 (Warp Shuffle Register) |   %7.3f ms | %5.1f GB/s |  %5.2fx\n", ms7, calc_bw(ms7), ms1 / ms7);
    printf("========================================================================\n");

    CUDA_CHECK(cudaFree(d_idata));
    CUDA_CHECK(cudaFree(d_odata));
    return 0;
}
