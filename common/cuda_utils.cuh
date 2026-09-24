#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

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

// Time a kernel launch in milliseconds
// Usage: float ms = time_kernel([&]{ my_kernel<<<grid,block>>>(args); });
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
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  SMs: %d | Max threads/block: %d\n",
           prop.multiProcessorCount, prop.maxThreadsPerBlock);
    printf("  VRAM: %zu MB | Shared mem/block: %zu KB\n",
           prop.totalGlobalMem / (1024 * 1024),
           prop.sharedMemPerBlock / 1024);
    printf("  L2 cache: %d KB | Mem bus: %d-bit\n",
           prop.l2CacheSize / 1024, prop.memoryBusWidth);
    printf("---\n");
}
