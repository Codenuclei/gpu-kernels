#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>

#define FULL 0xffffffffu

// ---------------- Inline PTX Primitives ----------------
__device__ __forceinline__ float4 ptx_ld_cs_f4(const float4* ptr) {
    float4 r;
    asm volatile(
        "ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(*reinterpret_cast<unsigned*>(&r.x)),
          "=r"(*reinterpret_cast<unsigned*>(&r.y)),
          "=r"(*reinterpret_cast<unsigned*>(&r.z)),
          "=r"(*reinterpret_cast<unsigned*>(&r.w))
        : "l"(ptr)
    );
    return r;
}

__device__ __forceinline__ void ptx_st_cs_f4(float4* ptr, float4 v) {
    asm volatile(
        "st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
        :: "l"(ptr),
           "r"(*reinterpret_cast<unsigned*>(&v.x)),
           "r"(*reinterpret_cast<unsigned*>(&v.y)),
           "r"(*reinterpret_cast<unsigned*>(&v.z)),
           "r"(*reinterpret_cast<unsigned*>(&v.w))
        : "memory"
    );
}

__device__ __forceinline__ float ptx_fast_exp(float x) {
    constexpr float LOG2E = 1.4426950408889634f;
    float t = x * LOG2E;
    float res;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(res) : "f"(t));
    return res;
}

__device__ __forceinline__ float ptx_shfl_xor(float val, int mask) {
    float res;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(res) : "f"(val), "r"(mask));
    return res;
}

struct MS { float m, s; };
__device__ __forceinline__ MS ptx_merge(MS a, MS b) {
    float m = fmaxf(a.m, b.m);
    return { m, a.s * ptx_fast_exp(a.m - m) + b.s * ptx_fast_exp(b.m - m) };
}

// ---------------- 1. FP32 Inline PTX Softmax ----------------
template <int BLOCK, int V>
__global__ void softmax_fwd_ptx_k(const float* __restrict__ x, float* __restrict__ y, int rows, int D) {
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int D4 = D >> 2;

    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const float4* xr = reinterpret_cast<const float4*>(x + (size_t)row * D);
        float4*       yr = reinterpret_cast<float4*>(y + (size_t)row * D);

        float4 r[V];
        #pragma unroll
        for (int v = 0; v < V; ++v) {
            int idx = tid + v * BLOCK;
            if (idx < D4) r[v] = ptx_ld_cs_f4(xr + idx);
        }

        MS p = { -FLT_MAX, 0.f };
        #pragma unroll
        for (int v = 0; v < V; ++v) {
            int idx = tid + v * BLOCK;
            if (idx < D4) {
                float lm = fmaxf(fmaxf(r[v].x, r[v].y), fmaxf(r[v].z, r[v].w));
                MS q = { lm, ptx_fast_exp(r[v].x - lm) + ptx_fast_exp(r[v].y - lm) +
                             ptx_fast_exp(r[v].z - lm) + ptx_fast_exp(r[v].w - lm) };
                p = ptx_merge(p, q);
            }
        }

        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            MS q = { ptx_shfl_xor(p.m, o), ptx_shfl_xor(p.s, o) };
            p = ptx_merge(p, q);
        }

        __shared__ MS sh[BLOCK / 32];
        if (lane == 0) sh[warp] = p;
        __syncthreads();
        MS t = sh[0];
        #pragma unroll
        for (int w = 1; w < BLOCK / 32; ++w) t = ptx_merge(t, sh[w]);

        float inv;
        asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(t.s));

        #pragma unroll
        for (int v = 0; v < V; ++v) {
            int idx = tid + v * BLOCK;
            if (idx < D4) {
                float4 o4;
                o4.x = ptx_fast_exp(r[v].x - t.m) * inv;
                o4.y = ptx_fast_exp(r[v].y - t.m) * inv;
                o4.z = ptx_fast_exp(r[v].z - t.m) * inv;
                o4.w = ptx_fast_exp(r[v].w - t.m) * inv;
                ptx_st_cs_f4(yr + idx, o4);
            }
        }
    }
}

// ---------------- 2. FP16 Warp-per-Row Softmax (D <= 1024) ----------------
template <int V>
__global__ void softmax_fwd_warp_fp16_k(const __half* __restrict__ x, __half* __restrict__ y, int rows, int D) {
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
                MS q = { lm, ptx_fast_exp(f0.x - lm) + ptx_fast_exp(f0.y - lm) +
                             ptx_fast_exp(f1.x - lm) + ptx_fast_exp(f1.y - lm) +
                             ptx_fast_exp(f2.x - lm) + ptx_fast_exp(f2.y - lm) +
                             ptx_fast_exp(f3.x - lm) + ptx_fast_exp(f3.y - lm) };
                p = ptx_merge(p, q);
            }
        }

        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            MS q = { ptx_shfl_xor(p.m, o), ptx_shfl_xor(p.s, o) };
            p = ptx_merge(p, q);
        }

        float inv;
        asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(p.s));

        #pragma unroll
        for (int v = 0; v < V; ++v) {
            int idx = lane_id + v * 32;
            if (idx < D8) {
                const __half2* h2 = reinterpret_cast<const __half2*>(&r[v]);
                float2 f0 = __half22float2(h2[0]);
                float2 f1 = __half22float2(h2[1]);
                float2 f2 = __half22float2(h2[2]);
                float2 f3 = __half22float2(h2[3]);

                float2 o0 = { ptx_fast_exp(f0.x - p.m) * inv, ptx_fast_exp(f0.y - p.m) * inv };
                float2 o1 = { ptx_fast_exp(f1.x - p.m) * inv, ptx_fast_exp(f1.y - p.m) * inv };
                float2 o2 = { ptx_fast_exp(f2.x - p.m) * inv, ptx_fast_exp(f2.y - p.m) * inv };
                float2 o3 = { ptx_fast_exp(f3.x - p.m) * inv, ptx_fast_exp(f3.y - p.m) * inv };

                uint4 out_u4;
                __half2* out_h2 = reinterpret_cast<__half2*>(&out_u4);
                out_h2[0] = __float22half2_rn(o0);
                out_h2[1] = __float22half2_rn(o1);
                out_h2[2] = __float22half2_rn(o2);
                out_h2[3] = __float22half2_rn(o3);
                yr[idx] = out_u4;
            }
        }
    }
}

// ---------------- 3. FP16 Block-per-Row Softmax (D > 1024) ----------------
template <int BLOCK, int V>
__global__ void softmax_fwd_block_fp16_k(const __half* __restrict__ x, __half* __restrict__ y, int rows, int D) {
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
                MS q = { lm, ptx_fast_exp(f0.x - lm) + ptx_fast_exp(f0.y - lm) +
                             ptx_fast_exp(f1.x - lm) + ptx_fast_exp(f1.y - lm) +
                             ptx_fast_exp(f2.x - lm) + ptx_fast_exp(f2.y - lm) +
                             ptx_fast_exp(f3.x - lm) + ptx_fast_exp(f3.y - lm) };
                p = ptx_merge(p, q);
            }
        }

        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            MS q = { ptx_shfl_xor(p.m, o), ptx_shfl_xor(p.s, o) };
            p = ptx_merge(p, q);
        }

        __shared__ MS sh[BLOCK / 32];
        if (lane == 0) sh[warp] = p;
        __syncthreads();
        MS t = sh[0];
        #pragma unroll
        for (int w = 1; w < BLOCK / 32; ++w) t = ptx_merge(t, sh[w]);

        float inv;
        asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(t.s));

        #pragma unroll
        for (int v = 0; v < V; ++v) {
            int idx = tid + v * BLOCK;
            if (idx < D8) {
                const __half2* h2 = reinterpret_cast<const __half2*>(&r[v]);
                float2 f0 = __half22float2(h2[0]);
                float2 f1 = __half22float2(h2[1]);
                float2 f2 = __half22float2(h2[2]);
                float2 f3 = __half22float2(h2[3]);

                float2 o0 = { ptx_fast_exp(f0.x - t.m) * inv, ptx_fast_exp(f0.y - t.m) * inv };
                float2 o1 = { ptx_fast_exp(f1.x - t.m) * inv, ptx_fast_exp(f1.y - t.m) * inv };
                float2 o2 = { ptx_fast_exp(f2.x - t.m) * inv, ptx_fast_exp(f2.y - t.m) * inv };
                float2 o3 = { ptx_fast_exp(f3.x - t.m) * inv, ptx_fast_exp(f3.y - t.m) * inv };

                uint4 out_u4;
                __half2* out_h2 = reinterpret_cast<__half2*>(&out_u4);
                out_h2[0] = __float22half2_rn(o0);
                out_h2[1] = __float22half2_rn(o1);
                out_h2[2] = __float22half2_rn(o2);
                out_h2[3] = __float22half2_rn(o3);
                yr[idx] = out_u4;
            }
        }
    }
}

// ---------------- 4. Fused SRAM Flash-Softmax (Attention) ----------------
template <int HEAD_DIM>
__global__ void flash_softmax_sram_gemm_k(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N
) {
    const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    const int lane = threadIdx.x % 32;
    if (row >= N) return;

    float q_val = (lane < HEAD_DIM) ? Q[row * HEAD_DIM + lane] : 0.0f;
    float o_acc = 0.0f;
    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;

    for (int k_idx = 0; k_idx < N; ++k_idx) {
        float k_val = (lane < HEAD_DIM) ? K[k_idx * HEAD_DIM + lane] : 0.0f;
        float prod = q_val * k_val;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) prod += ptx_shfl_xor(prod, o);
        float score = __shfl_sync(FULL, prod, 0);

        float m_curr = fmaxf(m_prev, score);
        float alpha = ptx_fast_exp(m_prev - m_curr);
        float p = ptx_fast_exp(score - m_curr);
        float l_curr = l_prev * alpha + p;

        float v_val = (lane < HEAD_DIM) ? V[k_idx * HEAD_DIM + lane] : 0.0f;
        o_acc = o_acc * alpha + p * v_val;

        m_prev = m_curr;
        l_prev = l_curr;
    }

    float inv_l;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv_l) : "f"(l_prev));
    if (lane < HEAD_DIM) {
        O[row * HEAD_DIM + lane] = o_acc * inv_l;
    }
}

extern "C" {
void launch_ptx_fp32(const float* x, float* y, int rows, int D, cudaStream_t stream) {
    int D4 = D / 4;
    int v = (D4 + 255) / 256;
    int blocks = rows > 1024 ? 1024 : rows;
    if      (v <= 1) softmax_fwd_ptx_k<256, 1><<<blocks, 256, 0, stream>>>(x, y, rows, D);
    else if (v <= 2) softmax_fwd_ptx_k<256, 2><<<blocks, 256, 0, stream>>>(x, y, rows, D);
    else if (v <= 4) softmax_fwd_ptx_k<256, 4><<<blocks, 256, 0, stream>>>(x, y, rows, D);
    else             softmax_fwd_ptx_k<256, 8><<<blocks, 256, 0, stream>>>(x, y, rows, D);
}

void launch_latest_fp16(const __half* x, __half* y, int rows, int D, cudaStream_t stream) {
    if (D <= 1024) {
        int D8 = D / 8;
        int v = (D8 + 31) / 32;
        int blocks = (rows + 7) / 8;
        if (blocks > 1024) blocks = 1024;
        if      (v <= 1) softmax_fwd_warp_fp16_k<1><<<blocks, 256, 0, stream>>>(x, y, rows, D);
        else if (v <= 2) softmax_fwd_warp_fp16_k<2><<<blocks, 256, 0, stream>>>(x, y, rows, D);
        else             softmax_fwd_warp_fp16_k<4><<<blocks, 256, 0, stream>>>(x, y, rows, D);
    } else {
        int D8 = D / 8;
        int v = (D8 + 255) / 256;
        int blocks = rows > 1024 ? 1024 : rows;
        if      (v <= 1) softmax_fwd_block_fp16_k<256, 1><<<blocks, 256, 0, stream>>>(x, y, rows, D);
        else if (v <= 2) softmax_fwd_block_fp16_k<256, 2><<<blocks, 256, 0, stream>>>(x, y, rows, D);
        else             softmax_fwd_block_fp16_k<256, 4><<<blocks, 256, 0, stream>>>(x, y, rows, D);
    }
}

void launch_flash_softmax(const float* Q, const float* K, const float* V, float* O, int N, cudaStream_t stream) {
    int warps_per_block = 8;
    int blocks = (N + warps_per_block - 1) / warps_per_block;
    flash_softmax_sram_gemm_k<64><<<blocks, 256, 0, stream>>>(Q, K, V, O, N);
}
}
