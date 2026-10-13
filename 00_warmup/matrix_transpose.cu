/*
 * ============================================================================
 * CHALLENGE 00: WARMUP — Matrix Transpose & Shared Memory Bank Conflicts
 * ============================================================================
 *
 * Demonstrates:
 *   1. Naive transpose: coalesced read, strided/uncoalesced global write
 *   2. Shared memory tiled transpose: coalesced read & write, but 32-way bank conflict
 *   3. Conflict-free padded shared memory transpose: tile[TILE_DIM][TILE_DIM + 1]
 *
 * Hardware: RTX 2050 (SM 8.6, 32 shared memory banks of 4 bytes each)
 */

#include "common/cuda_utils.cuh"
#include <cstdio>
#include <vector>

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8; // Each thread processes 4 elements per tile

// ----------------------------------------------------------------------------
// Version 1: Naive (Coalesced Read, Uncoalesced Write)
// ----------------------------------------------------------------------------
__global__ void transpose_naive(const float* __restrict__ in,
                                float* __restrict__ out,
                                int width, int height) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            out[x * height + (y + j)] = in[(y + j) * width + x];
        }
    }
}

// ----------------------------------------------------------------------------
// Version 2: Shared Memory Tiled (Has 32-way Bank Conflicts)
// ----------------------------------------------------------------------------
__global__ void transpose_shared_bank_conflicts(const float* __restrict__ in,
                                                float* __restrict__ out,
                                                int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Coalesced load into shared memory
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
        }
    }
    __syncthreads();

    // Transposed block coordinates
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Coalesced write to global memory, but reading shared mem has 32-way bank conflict
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// ----------------------------------------------------------------------------
// Version 3: Conflict-Free Shared Memory Tiled (Padding + 1)
// ----------------------------------------------------------------------------
__global__ void transpose_conflict_free(const float* __restrict__ in,
                                        float* __restrict__ out,
                                        int width, int height) {
    // Adding +1 column padding skews rows so consecutive threads access different banks
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
        }
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// CPU Reference
void cpu_transpose(const float* in, float* out, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            out[x * height + y] = in[y * width + x];
        }
    }
}

int main() {
    print_gpu_info();

    const int width = 4096;
    const int height = 4096;
    const int N = width * height; // 16M floats = 64 MB
    const size_t bytes = N * sizeof(float);

    printf("Matrix Transpose [%d x %d] (%.2f MB)\n\n",
           width, height, bytes / (1024.0 * 1024.0));

    std::vector<float> h_in(N);
    std::vector<float> h_out_cpu(N);
    std::vector<float> h_out_gpu(N);

    init_random(h_in.data(), N, -10.0f, 10.0f);
    cpu_transpose(h_in.data(), h_out_cpu.data(), width, height);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid(ceil_div(width, TILE_DIM), ceil_div(height, TILE_DIM));

    printf("--- Verification ---\n");

    // Naive
    transpose_naive<<<grid, block>>>(d_in, d_out, width, height);
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    verify_close(h_out_cpu.data(), h_out_gpu.data(), N, 1e-4f, 1e-4f, "v1_naive");

    // Shared Bank Conflicts
    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    transpose_shared_bank_conflicts<<<grid, block>>>(d_in, d_out, width, height);
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    verify_close(h_out_cpu.data(), h_out_gpu.data(), N, 1e-4f, 1e-4f, "v2_shared_bank_conflicts");

    // Conflict Free
    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    transpose_conflict_free<<<grid, block>>>(d_in, d_out, width, height);
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    verify_close(h_out_cpu.data(), h_out_gpu.data(), N, 1e-4f, 1e-4f, "v3_conflict_free");

    printf("\n--- Performance Benchmarks ---\n");
    auto calc_bandwidth = [&](float ms) {
        return (2.0 * bytes) / (ms * 1e6); // Read + Write GB/s
    };

    float ms_naive = time_kernel([&] {
        transpose_naive<<<grid, block>>>(d_in, d_out, width, height);
    });
    printf("v1_naive (uncoalesced write) : %6.3f ms | Bandwidth: %6.2f GB/s\n",
           ms_naive, calc_bandwidth(ms_naive));

    float ms_shared = time_kernel([&] {
        transpose_shared_bank_conflicts<<<grid, block>>>(d_in, d_out, width, height);
    });
    printf("v2_shared (32-way bank conflict): %6.3f ms | Bandwidth: %6.2f GB/s | Speedup: %.2fx\n",
           ms_shared, calc_bandwidth(ms_shared), ms_naive / ms_shared);

    float ms_pad = time_kernel([&] {
        transpose_conflict_free<<<grid, block>>>(d_in, d_out, width, height);
    });
    printf("v3_conflict_free (padding +1)   : %6.3f ms | Bandwidth: %6.2f GB/s | Speedup: %.2fx\n",
           ms_pad, calc_bandwidth(ms_pad), ms_naive / ms_pad);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
