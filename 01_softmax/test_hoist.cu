#include <cstdio>
#include <cuda_runtime.h>

#define FULL 0xffffffffu

struct MS { float m, s; };

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
    asm volatile(
        "shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
        : "=f"(res)
        : "f"(val), "r"(mask)
    );
    return res;
}

__device__ __forceinline__ MS ptx_merge(MS a, MS b) {
    float m = fmaxf(a.m, b.m);
    return { m, a.s * ptx_fast_exp(a.m - m) + b.s * ptx_fast_exp(b.m - m) };
}

template <int BLOCK, int V>
__global__ void softmax_hoisted(const float* __restrict__ x, float* __restrict__ y, int rows, int D) {
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int D4 = D >> 2;

    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const float4* xr = reinterpret_cast<const float4*>(x + (size_t)row * D);
        float4*       yr = reinterpret_cast<float4*>(y + (size_t)row * D);

        float4 r[V];
        // 1. Hoist all memory loads together to saturate memory pipeline
        #pragma unroll
        for (int v = 0; v < V; ++v) {
            int idx = tid + v * BLOCK;
            if (idx < D4) {
                r[v] = ptx_ld_cs_f4(xr + idx);
            }
        }

        // 2. Compute online max + sum
        MS p = { -3.402823466e+38f, 0.f };
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

        // Warp butterfly
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

int main() {
    constexpr int ROWS = 8192, D = 4096;
    size_t n = (size_t)ROWS * D;
    float *dx, *dy;
    cudaMalloc(&dx, n * 4);
    cudaMalloc(&dy, n * 4);
    constexpr int BLOCK = 256;
    constexpr int V = (D / 4 + BLOCK - 1) / BLOCK; // = 4

    // Warmup
    for (int i = 0; i < 15; ++i) softmax_hoisted<BLOCK, V><<<1024, BLOCK>>>(dx, dy, ROWS, D);
    cudaDeviceSynchronize();

    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    const int IT = 200;
    for (int i = 0; i < IT; ++i) softmax_hoisted<BLOCK, V><<<1024, BLOCK>>>(dx, dy, ROWS, D);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); ms /= IT;

    double bw = (2.0 * n * 4) / (ms * 1e-3) / 1e9;
    printf("Hoisted Memory Pipeline PTX [8192 x 4096]:\n");
    printf("  Latency: %.3f ms\n", ms);
    printf("  Bandwidth: %.1f GB/s\n", bw);
    return 0;
}
