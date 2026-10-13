/*
 * ============================================================================
 * CHALLENGE 02: REDUCE — Parallel Prefix Scan (Cumulative Sum)
 * ============================================================================
 *
 * Demonstrates:
 *   1. Hillis-Steele scan: step-efficient O(N log N) work
 *   2. Blelloch scan: work-efficient O(N) work (Up-Sweep + Down-Sweep)
 *   3. Warp-Shuffle scan: __shfl_up_sync intra-warp register scan
 *
 * Hardware: RTX 2050 (SM 8.6, Ampere)
 */

#include "common/cuda_utils.cuh"
#include <cstdio>
#include <vector>

constexpr int BLOCK_SIZE = 512;
constexpr int LOG_NUM_BANKS = 5;
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

// ----------------------------------------------------------------------------
// Version 1: Hillis-Steele (Inclusive Scan, O(N log N) work)
// ----------------------------------------------------------------------------
__global__ void scan_hillis_steele(const float* __restrict__ in,
                                   float* __restrict__ out,
                                   int n) {
    __shared__ float s_data[2][BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int cur = 0;
    s_data[cur][tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        cur = 1 - cur;
        if (tid >= offset) {
            s_data[cur][tid] = s_data[1 - cur][tid] + s_data[1 - cur][tid - offset];
        } else {
            s_data[cur][tid] = s_data[1 - cur][tid];
        }
        __syncthreads();
    }

    if (i < n) {
        out[i] = s_data[cur][tid];
    }
}

// ----------------------------------------------------------------------------
// Version 2: Blelloch Work-Efficient Scan (Exclusive Scan, O(N) work)
// ----------------------------------------------------------------------------
__global__ void scan_blelloch(const float* __restrict__ in,
                              float* __restrict__ out,
                              int n) {
    // Shared memory with padding to prevent bank conflicts
    __shared__ float s_data[2 * BLOCK_SIZE + (2 * BLOCK_SIZE >> LOG_NUM_BANKS)];

    int tid = threadIdx.x;
    int base = blockIdx.x * (2 * blockDim.x);

    int ai = tid;
    int bi = tid + blockDim.x;
    int bank_offset_a = CONFLICT_FREE_OFFSET(ai);
    int bank_offset_b = CONFLICT_FREE_OFFSET(bi);

    s_data[ai + bank_offset_a] = (base + ai < n) ? in[base + ai] : 0.0f;
    s_data[bi + bank_offset_b] = (base + bi < n) ? in[base + bi] : 0.0f;

    // Up-Sweep (Reduction Phase)
    int offset = 1;
    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai_idx = offset * (2 * tid + 1) - 1;
            int bi_idx = offset * (2 * tid + 2) - 1;
            ai_idx += CONFLICT_FREE_OFFSET(ai_idx);
            bi_idx += CONFLICT_FREE_OFFSET(bi_idx);
            s_data[bi_idx] += s_data[ai_idx];
        }
        offset <<= 1;
    }

    // Set root to zero
    if (tid == 0) {
        int root = 2 * blockDim.x - 1;
        root += CONFLICT_FREE_OFFSET(root);
        s_data[root] = 0.0f;
    }

    // Down-Sweep (Distribution Phase)
    for (int d = 1; d <= blockDim.x; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai_idx = offset * (2 * tid + 1) - 1;
            int bi_idx = offset * (2 * tid + 2) - 1;
            ai_idx += CONFLICT_FREE_OFFSET(ai_idx);
            bi_idx += CONFLICT_FREE_OFFSET(bi_idx);

            float t = s_data[ai_idx];
            s_data[ai_idx] = s_data[bi_idx];
            s_data[bi_idx] += t;
        }
    }
    __syncthreads();

    if (base + ai < n) out[base + ai] = s_data[ai + bank_offset_a];
    if (base + bi < n) out[base + bi] = s_data[bi + bank_offset_b];
}

// ----------------------------------------------------------------------------
// Version 3: Warp Shuffle Scan (__shfl_up_sync)
// ----------------------------------------------------------------------------
__device__ __forceinline__ float warp_scan_inclusive(float val) {
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float temp = __shfl_up_sync(FULL_MASK, val, offset);
        if ((threadIdx.x % WARP_SIZE) >= offset) {
            val += temp;
        }
    }
    return val;
}

__global__ void scan_warp_shuffle(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  int n) {
    __shared__ float s_warp_sums[WARP_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (i < n) ? in[i] : 0.0f;
    val = warp_scan_inclusive(val);

    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    if (lane == WARP_SIZE - 1) {
        s_warp_sums[wid] = val;
    }
    __syncthreads();

    // First warp scans the warp totals
    if (wid == 0) {
        float warp_total = (lane < blockDim.x / WARP_SIZE) ? s_warp_sums[lane] : 0.0f;
        warp_total = warp_scan_inclusive(warp_total);
        if (lane < blockDim.x / WARP_SIZE) {
            s_warp_sums[lane] = warp_total;
        }
    }
    __syncthreads();

    // Add prefix from previous warps
    if (wid > 0) {
        val += s_warp_sums[wid - 1];
    }

    if (i < n) {
        out[i] = val;
    }
}

// ----------------------------------------------------------------------------
// CPU Ground Truth References
// ----------------------------------------------------------------------------
void cpu_inclusive_scan(const float* in, float* out, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += in[i];
        out[i] = static_cast<float>(sum);
    }
}

void cpu_exclusive_scan(const float* in, float* out, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        out[i] = static_cast<float>(sum);
        sum += in[i];
    }
}

int main() {
    print_gpu_info();

    // Test across a full 512-element block
    const int N = BLOCK_SIZE;
    const size_t bytes = N * sizeof(float);
    printf("Parallel Prefix Scan Benchmark on Block (N = %d elements)\n\n", N);

    std::vector<float> h_in(N);
    std::vector<float> h_out_inc_ref(N);
    std::vector<float> h_out_exc_ref(N);
    std::vector<float> h_out_gpu(N);

    init_random(h_in.data(), N, 0.1f, 1.0f);
    cpu_inclusive_scan(h_in.data(), h_out_inc_ref.data(), N);
    cpu_exclusive_scan(h_in.data(), h_out_exc_ref.data(), N);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    printf("--- Correctness Verification ---\n");

    // Hillis-Steele
    scan_hillis_steele<<<1, BLOCK_SIZE>>>(d_in, d_out, N);
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    verify_close(h_out_inc_ref.data(), h_out_gpu.data(), N, 1e-4f, 1e-4f, "Hillis-Steele (Inclusive)");

    // Blelloch
    scan_blelloch<<<1, BLOCK_SIZE / 2>>>(d_in, d_out, N);
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    verify_close(h_out_exc_ref.data(), h_out_gpu.data(), N, 1e-4f, 1e-4f, "Blelloch (Exclusive)");

    // Warp Shuffle
    scan_warp_shuffle<<<1, BLOCK_SIZE>>>(d_in, d_out, N);
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    verify_close(h_out_inc_ref.data(), h_out_gpu.data(), N, 1e-4f, 1e-4f, "Warp Shuffle (Inclusive)");

    printf("\n--- Performance Benchmarks (10,000 repeats) ---\n");

    float ms_hs = time_kernel([&] {
        scan_hillis_steele<<<1, BLOCK_SIZE>>>(d_in, d_out, N);
    }, 50, 10000);

    float ms_bl = time_kernel([&] {
        scan_blelloch<<<1, BLOCK_SIZE / 2>>>(d_in, d_out, N);
    }, 50, 10000);

    float ms_ws = time_kernel([&] {
        scan_warp_shuffle<<<1, BLOCK_SIZE>>>(d_in, d_out, N);
    }, 50, 10000);

    printf("Hillis-Steele (O(N log N)) : %6.4f ms\n", ms_hs);
    printf("Blelloch Work-Efficient    : %6.4f ms\n", ms_bl);
    printf("Warp Shuffle Register Scan : %6.4f ms | Speedup: %.2fx vs Hillis-Steele\n",
           ms_ws, ms_hs / ms_ws);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
