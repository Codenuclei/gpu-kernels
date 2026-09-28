#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>

#define FULL 0xffffffffu

struct MS { float m, s; };

__device__ __forceinline__ MS merge(MS a, MS b) {
    float m = fmaxf(a.m, b.m);
    return { m, a.s * __expf(a.m - m) + b.s * __expf(b.m - m) };
}

// -------------------------------------------------------------
// 1. Warp-per-Row FP16 (D <= 1024)
// -------------------------------------------------------------
template <int V>
__global__ void softmax_fwd_warp_fp16_k(
    const __half* __restrict__ x,
    __half* __restrict__ y,
    float* __restrict__ L,
    int rows, int D
) {
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int rows_per_block = blockDim.x >> 5;
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
// -------------------------------------------------------------
template <int BLOCK, int V>
__global__ void softmax_fwd_block_fp16_k(
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

extern "C" {
void softmax_fwd_fp16_launch(
    const __half* x,
    __half* y,
    float* L,
    int rows, int D,
    cudaStream_t stream
) {
    if (D <= 1024) {
        int D8 = D / 8;
        int v = (D8 + 31) / 32;
        int blocks = (rows + 7) / 8;
        if (blocks > 1024) blocks = 1024;
        if      (v <= 1) softmax_fwd_warp_fp16_k<1><<<blocks, 256, 0, stream>>>(x, y, L, rows, D);
        else if (v <= 2) softmax_fwd_warp_fp16_k<2><<<blocks, 256, 0, stream>>>(x, y, L, rows, D);
        else             softmax_fwd_warp_fp16_k<4><<<blocks, 256, 0, stream>>>(x, y, L, rows, D);
    } else {
        int D8 = D / 8;
        int v = (D8 + 255) / 256;
        int blocks = rows > 1024 ? 1024 : rows;
        if      (v <= 1) softmax_fwd_block_fp16_k<256, 1><<<blocks, 256, 0, stream>>>(x, y, L, rows, D);
        else if (v <= 2) softmax_fwd_block_fp16_k<256, 2><<<blocks, 256, 0, stream>>>(x, y, L, rows, D);
        else             softmax_fwd_block_fp16_k<256, 4><<<blocks, 256, 0, stream>>>(x, y, L, rows, D);
    }
}
}
