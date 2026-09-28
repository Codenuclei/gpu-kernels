#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define FULL 0xffffffffu

struct MS { float m, s; };

__device__ __forceinline__ MS merge(MS a, MS b) {
    float m = fmaxf(a.m, b.m);
    return { m, a.s * __expf(a.m - m) + b.s * __expf(b.m - m) };
}

// -------------------------------------------------------------
// 1. Warp-per-Row FP16 (D <= 1024)
// 8 warps per block (256 threads) -> 8 rows per block
// ZERO shared memory, ZERO __syncthreads()
// -------------------------------------------------------------
template <int V>
__global__ void softmax_fwd_warp_fp16(
    const __half* __restrict__ x,
    __half* __restrict__ y,
    float* __restrict__ L,
    int rows, int D
) {
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int rows_per_block = blockDim.x >> 5; // 8
    const int D8 = D >> 3;
    const int stride = gridDim.x * rows_per_block;

    for (int row = blockIdx.x * rows_per_block + warp_id; row < rows; row += stride) {
        const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)row * D);
        uint4*       yr = reinterpret_cast<uint4*>(y + (size_t)row * D);

        uint4 r[V];
        MS p = { -FLT_MAX, 0.f };

        #pragma unroll
        for (int v = 0; v < V; ++v) {
            int idx = lane_id + v * 32;
            if (idx < D8) {
                r[v] = xr[idx];
                const __half2* h2 = reinterpret_cast<const __half2*>(&r[v]);
                float2 f0 = __half22float2(h2[0]);
                float2 f1 = __half22float2(h2[1]);
                float2 f2 = __half22float2(h2[2]);
                float2 f3 = __half22float2(h2[3]);

                float lm = fmaxf(fmaxf(fmaxf(f0.x, f0.y), fmaxf(f1.x, f1.y)),
                                 fmaxf(fmaxf(f2.x, f2.y), fmaxf(f3.x, f3.y)));
                MS q = { lm, __expf(f0.x - lm) + __expf(f0.y - lm) +
                             __expf(f1.x - lm) + __expf(f1.y - lm) +
                             __expf(f2.x - lm) + __expf(f2.y - lm) +
                             __expf(f3.x - lm) + __expf(f3.y - lm) };
                p = merge(p, q);
            }
        }

        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            MS q = { __shfl_xor_sync(FULL, p.m, o), __shfl_xor_sync(FULL, p.s, o) };
            p = merge(p, q);
        }

        const float inv = 1.f / p.s;
        #pragma unroll
        for (int v = 0; v < V; ++v) {
            int idx = lane_id + v * 32;
            if (idx < D8) {
                const __half2* h2 = reinterpret_cast<const __half2*>(&r[v]);
                float2 f0 = __half22float2(h2[0]);
                float2 f1 = __half22float2(h2[1]);
                float2 f2 = __half22float2(h2[2]);
                float2 f3 = __half22float2(h2[3]);

                float2 o0 = { __expf(f0.x - p.m) * inv, __expf(f0.y - p.m) * inv };
                float2 o1 = { __expf(f1.x - p.m) * inv, __expf(f1.y - p.m) * inv };
                float2 o2 = { __expf(f2.x - p.m) * inv, __expf(f2.y - p.m) * inv };
                float2 o3 = { __expf(f3.x - p.m) * inv, __expf(f3.y - p.m) * inv };

                uint4 out_u4;
                __half2* out_h2 = reinterpret_cast<__half2*>(&out_u4);
                out_h2[0] = __float22half2_rn(o0);
                out_h2[1] = __float22half2_rn(o1);
                out_h2[2] = __float22half2_rn(o2);
                out_h2[3] = __float22half2_rn(o3);

                yr[idx] = out_u4;
            }
        }

        if (lane_id == 0 && L != nullptr) {
            L[row] = p.m + logf(p.s);
        }
    }
}

// -------------------------------------------------------------
// 2. Block-per-Row FP16 (D > 1024)
// 256 threads per row, multi-warp reduction
// -------------------------------------------------------------
template <int BLOCK, int V>
__global__ void softmax_fwd_block_fp16(
    const __half* __restrict__ x,
    __half* __restrict__ y,
    float* __restrict__ L,
    int rows, int D
) {
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int D8 = D >> 3;
    const int stride = gridDim.x;

    for (int row = blockIdx.x; row < rows; row += stride) {
        const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)row * D);
        uint4*       yr = reinterpret_cast<uint4*>(y + (size_t)row * D);

        uint4 r[V];
        MS p = { -FLT_MAX, 0.f };

        #pragma unroll
        for (int v = 0; v < V; ++v) {
            int idx = tid + v * BLOCK;
            if (idx < D8) {
                r[v] = xr[idx];
                const __half2* h2 = reinterpret_cast<const __half2*>(&r[v]);
                float2 f0 = __half22float2(h2[0]);
                float2 f1 = __half22float2(h2[1]);
                float2 f2 = __half22float2(h2[2]);
                float2 f3 = __half22float2(h2[3]);

                float lm = fmaxf(fmaxf(fmaxf(f0.x, f0.y), fmaxf(f1.x, f1.y)),
                                 fmaxf(fmaxf(f2.x, f2.y), fmaxf(f3.x, f3.y)));
                MS q = { lm, __expf(f0.x - lm) + __expf(f0.y - lm) +
                             __expf(f1.x - lm) + __expf(f1.y - lm) +
                             __expf(f2.x - lm) + __expf(f2.y - lm) +
                             __expf(f3.x - lm) + __expf(f3.y - lm) };
                p = merge(p, q);
            }
        }

        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            MS q = { __shfl_xor_sync(FULL, p.m, o), __shfl_xor_sync(FULL, p.s, o) };
            p = merge(p, q);
        }

        __shared__ MS sh[BLOCK / 32];
        if (lane == 0) sh[warp] = p;
        __syncthreads();
        MS t = sh[0];
        #pragma unroll
        for (int w = 1; w < BLOCK / 32; ++w) t = merge(t, sh[w]);

        const float inv = 1.f / t.s;
        #pragma unroll
        for (int v = 0; v < V; ++v) {
            int idx = tid + v * BLOCK;
            if (idx < D8) {
                const __half2* h2 = reinterpret_cast<const __half2*>(&r[v]);
                float2 f0 = __half22float2(h2[0]);
                float2 f1 = __half22float2(h2[1]);
                float2 f2 = __half22float2(h2[2]);
                float2 f3 = __half22float2(h2[3]);

                float2 o0 = { __expf(f0.x - t.m) * inv, __expf(f0.y - t.m) * inv };
                float2 o1 = { __expf(f1.x - t.m) * inv, __expf(f1.y - t.m) * inv };
                float2 o2 = { __expf(f2.x - t.m) * inv, __expf(f2.y - t.m) * inv };
                float2 o3 = { __expf(f3.x - t.m) * inv, __expf(f3.y - t.m) * inv };

                uint4 out_u4;
                __half2* out_h2 = reinterpret_cast<__half2*>(&out_u4);
                out_h2[0] = __float22half2_rn(o0);
                out_h2[1] = __float22half2_rn(o1);
                out_h2[2] = __float22half2_rn(o2);
                out_h2[3] = __float22half2_rn(o3);

                yr[idx] = out_u4;
            }
        }

        if (tid == 0 && L != nullptr) {
            L[row] = t.m + logf(t.s);
        }
    }
}

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)

void test_block_per_row() {
    constexpr int ROWS = 8192, D = 4096;
    size_t n = (size_t)ROWS * D;

    std::vector<__half> hx(n), hy(n);
    for (size_t i = 0; i < n; ++i) {
        float val = 20.f * (rand() / (float)RAND_MAX) - 10.f;
        hx[i] = __float2half(val);
    }

    __half *dx, *dy;
    float *dL;
    CK(cudaMalloc(&dx, n * sizeof(__half)));
    CK(cudaMalloc(&dy, n * sizeof(__half)));
    CK(cudaMalloc(&dL, ROWS * sizeof(float)));
    CK(cudaMemcpy(dx, hx.data(), n * sizeof(__half), cudaMemcpyHostToDevice));

    constexpr int BLOCK = 256;
    constexpr int V = (D / 8 + BLOCK - 1) / BLOCK; // = 2
    for (int i = 0; i < 10; ++i) {
        softmax_fwd_block_fp16<BLOCK, V><<<1024, BLOCK>>>(dx, dy, dL, ROWS, D);
    }
    CK(cudaDeviceSynchronize());

    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    const int IT = 100;
    for (int i = 0; i < IT; ++i) {
        softmax_fwd_block_fp16<BLOCK, V><<<1024, BLOCK>>>(dx, dy, dL, ROWS, D);
    }
    cudaEventRecord(b);
    CK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b); ms /= IT;
    CK(cudaMemcpy(hy.data(), dy, n * sizeof(__half), cudaMemcpyDeviceToHost));

    double bytes_transferred = 2.0 * n * sizeof(__half);
    double bw = bytes_transferred / (ms * 1e-3) / 1e9;
    double equiv_fp32_bw = (2.0 * n * 4) / (ms * 1e-3) / 1e9;

    printf("=== 1. Block-per-Row FP16 [8192 x 4096] ===\n");
    printf("  Latency: %.3f ms\n", ms);
    printf("  Physical Memory Bandwidth: %.1f GB/s\n", bw);
    printf("  Equivalent FP32 Throughput: %.1f GB/s (%.2fx speedup vs FP32)\n",
           equiv_fp32_bw, 2.56 / ms);

    CUDA_CHECK_CLEANUP:
    cudaFree(dx); cudaFree(dy); cudaFree(dL);
}

void test_warp_per_row() {
    constexpr int ROWS = 32768, D = 128;
    size_t n = (size_t)ROWS * D;

    std::vector<__half> hx(n), hy(n);
    for (size_t i = 0; i < n; ++i) {
        float val = 20.f * (rand() / (float)RAND_MAX) - 10.f;
        hx[i] = __float2half(val);
    }

    __half *dx, *dy;
    float *dL;
    CK(cudaMalloc(&dx, n * sizeof(__half)));
    CK(cudaMalloc(&dy, n * sizeof(__half)));
    CK(cudaMalloc(&dL, ROWS * sizeof(float)));
    CK(cudaMemcpy(dx, hx.data(), n * sizeof(__half), cudaMemcpyHostToDevice));

    constexpr int V = 1;
    for (int i = 0; i < 10; ++i) {
        softmax_fwd_warp_fp16<V><<<512, 256>>>(dx, dy, dL, ROWS, D);
    }
    CK(cudaDeviceSynchronize());

    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    const int IT = 100;
    for (int i = 0; i < IT; ++i) {
        softmax_fwd_warp_fp16<V><<<512, 256>>>(dx, dy, dL, ROWS, D);
    }
    cudaEventRecord(b);
    CK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b); ms /= IT;
    CK(cudaMemcpy(hy.data(), dy, n * sizeof(__half), cudaMemcpyDeviceToHost));

    double bytes = 2.0 * n * sizeof(__half);
    double bw = bytes / (ms * 1e-3) / 1e9;
    printf("\n=== 2. Warp-per-Row FP16 [32768 x 128] (Attention Head Dimension) ===\n");
    printf("  Latency: %.3f ms\n", ms);
    printf("  Physical Memory Bandwidth: %.1f GB/s\n", bw);

    cudaFree(dx); cudaFree(dy); cudaFree(dL);
}

int main() {
    test_block_per_row();
    test_warp_per_row();
    return 0;
}
