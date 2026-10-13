#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Hard crash on CUDA error — never silently fail
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// Integer ceiling division
inline __host__ __device__ int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// ============================================================================
// Warp-level reduction primitives
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val;
}

// Block-level reduction sum (requires sdata of size blockDim.x / 32)
__device__ __forceinline__ float block_reduce_sum(float val, float* sdata) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        sdata[wid] = val;
    }
    __syncthreads();

    // First warp reduces the partial sums from each warp
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? sdata[lane] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

// Block-level reduction max
__device__ __forceinline__ float block_reduce_max(float val, float* sdata) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_max(val);

    if (lane == 0) {
        sdata[wid] = val;
    }
    __syncthreads();

    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? sdata[lane] : -1e30f;
    if (wid == 0) {
        val = warp_reduce_max(val);
    }
    return val;
}

// ============================================================================
// Vectorized load & store helpers
// ============================================================================

__device__ __forceinline__ float4 load_float4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

__device__ __forceinline__ void store_float4(float* ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
}

__device__ __forceinline__ half2 load_half2(const half* ptr) {
    return *reinterpret_cast<const half2*>(ptr);
}

__device__ __forceinline__ void store_half2(half* ptr, half2 val) {
    *reinterpret_cast<half2*>(ptr) = val;
}

// ============================================================================
// Benchmarking & Timing
// ============================================================================

// Time a kernel launch in milliseconds
template <typename F>
float time_kernel(F kernel_launch, int warmup = 5, int repeats = 100) {
    // Warmup
    for (int i = 0; i < warmup; i++) {
        kernel_launch();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; i++) {
        kernel_launch();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / repeats;
}

// Print GPU info
inline void print_gpu_info() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("====================================================\n");
    printf("GPU: %s (SM %d.%d, Ampere)\n", prop.name, prop.major, prop.minor);
    printf("  SMs: %d | Max threads/block: %d\n",
           prop.multiProcessorCount, prop.maxThreadsPerBlock);
    printf("  VRAM: %zu MB | Shared mem/block: %zu KB\n",
           prop.totalGlobalMem / (1024 * 1024),
           prop.sharedMemPerBlock / 1024);
    printf("  L2 cache: %d KB | Mem bus: %d-bit\n",
           prop.l2CacheSize / 1024, prop.memoryBusWidth);
    printf("====================================================\n");
}

// ============================================================================
// CPU Ground Truth Verification Helpers
// ============================================================================

inline void init_random(float* data, size_t n, float low = -1.0f, float high = 1.0f) {
    for (size_t i = 0; i < n; i++) {
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data[i] = low + r * (high - low);
    }
}

inline bool verify_close(const float* ref, const float* test, size_t n,
                         float rtol = 1e-3f, float atol = 1e-3f,
                         const char* name = "Output") {
    float max_diff = 0.0f;
    size_t err_idx = 0;
    bool pass = true;

    for (size_t i = 0; i < n; i++) {
        float r = ref[i];
        float t = test[i];
        float diff = std::fabs(r - t);
        float tol = atol + rtol * std::fabs(r);

        if (diff > max_diff) {
            max_diff = diff;
        }

        if (diff > tol && pass) {
            pass = false;
            err_idx = i;
        }
    }

    if (pass) {
        printf("  [PASS] %s matches CPU reference (max_diff = %.2e)\n", name, max_diff);
    } else {
        printf("  [FAIL] %s mismatch at idx %zu: ref = %.6f, test = %.6f, diff = %.6f (max_diff = %.2e)\n",
               name, err_idx, ref[err_idx], test[err_idx], std::fabs(ref[err_idx] - test[err_idx]), max_diff);
    }
    return pass;
}
