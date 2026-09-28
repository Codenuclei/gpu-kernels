#include <cuda_runtime.h>
#include <cfloat>

#define FULL 0xffffffffu

struct MS { float m, s; };

__device__ __forceinline__ MS merge(MS a, MS b) {
    float m = fmaxf(a.m, b.m);
    return { m, a.s * __expf(a.m - m) + b.s * __expf(b.m - m) };
}

// ---------------- forward: saves L = m + log(s) ----------------
template <int BLOCK, int V>
__global__ void fwd_k(const float* __restrict__ x, float* __restrict__ y,
                      float* __restrict__ L, int D) {
    const int row = blockIdx.x, tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5, D4 = D >> 2;
    const float4* xr = reinterpret_cast<const float4*>(x + (size_t)row * D);
    float4*       yr = reinterpret_cast<float4*>(y + (size_t)row * D);

    float4 r[V];
    MS p = { -FLT_MAX, 0.f };
    #pragma unroll
    for (int v = 0; v < V; ++v) {
        int idx = tid + v * BLOCK;
        if (idx < D4) {
            r[v] = xr[idx];
            float lm = fmaxf(fmaxf(r[v].x, r[v].y), fmaxf(r[v].z, r[v].w));
            MS q = { lm, __expf(r[v].x - lm) + __expf(r[v].y - lm) +
                         __expf(r[v].z - lm) + __expf(r[v].w - lm) };
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
        if (idx < D4) {
            float4 o4;
            o4.x = __expf(r[v].x - t.m) * inv;
            o4.y = __expf(r[v].y - t.m) * inv;
            o4.z = __expf(r[v].z - t.m) * inv;
            o4.w = __expf(r[v].w - t.m) * inv;
            yr[idx] = o4;
        }
    }
    if (tid == 0) L[row] = t.m + logf(t.s);
}

// ---------------- backward: dx = y * (dy - sum(dy*y)), y = exp(x - L) ----------------
template <int BLOCK, int V>
__global__ void bwd_k(const float* __restrict__ x, const float* __restrict__ dy,
                      const float* __restrict__ L, float* __restrict__ dx, int D) {
    const int row = blockIdx.x, tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5, D4 = D >> 2;
    const float4* xr = reinterpret_cast<const float4*>(x  + (size_t)row * D);
    const float4* gr = reinterpret_cast<const float4*>(dy + (size_t)row * D);
    float4*       dr = reinterpret_cast<float4*>(dx + (size_t)row * D);
    const float Lr = L[row];

    float4 y[V], g[V];
    float dot = 0.f;
    #pragma unroll
    for (int v = 0; v < V; ++v) {
        int idx = tid + v * BLOCK;
        if (idx < D4) {
            float4 xv = xr[idx];
            g[v] = gr[idx];
            y[v].x = __expf(xv.x - Lr); y[v].y = __expf(xv.y - Lr);
            y[v].z = __expf(xv.z - Lr); y[v].w = __expf(xv.w - Lr);
            dot += y[v].x * g[v].x + y[v].y * g[v].y + y[v].z * g[v].z + y[v].w * g[v].w;
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(FULL, dot, o);
    __shared__ float sh[BLOCK / 32];
    if (lane == 0) sh[warp] = dot;
    __syncthreads();
    float tot = 0.f;
    #pragma unroll
    for (int w = 0; w < BLOCK / 32; ++w) tot += sh[w];

    #pragma unroll
    for (int v = 0; v < V; ++v) {
        int idx = tid + v * BLOCK;
        if (idx < D4) {
            float4 o4;
            o4.x = y[v].x * (g[v].x - tot);
            o4.y = y[v].y * (g[v].y - tot);
            o4.z = y[v].z * (g[v].z - tot);
            o4.w = y[v].w * (g[v].w - tot);
            dr[idx] = o4;
        }
    }
}

#define BLOCK 256
#define DISPATCH(KERNEL, D4, ROWS, STREAM, ...)                                   \
    do {                                                                          \
        int nv = ((D4) + BLOCK - 1) / BLOCK;                                      \
        if      (nv <= 1) KERNEL<BLOCK, 1><<<ROWS, BLOCK, 0, STREAM>>>(__VA_ARGS__); \
        else if (nv <= 2) KERNEL<BLOCK, 2><<<ROWS, BLOCK, 0, STREAM>>>(__VA_ARGS__); \
        else if (nv <= 4) KERNEL<BLOCK, 4><<<ROWS, BLOCK, 0, STREAM>>>(__VA_ARGS__); \
        else if (nv <= 8) KERNEL<BLOCK, 8><<<ROWS, BLOCK, 0, STREAM>>>(__VA_ARGS__); \
    } while (0)

extern "C" {
void softmax_fwd_cuda(const float* x, float* y, float* L, int rows, int D, cudaStream_t stream) {
    DISPATCH(fwd_k, D / 4, rows, stream, x, y, L, D);
}

void softmax_bwd_cuda(const float* x, const float* dy, const float* L, float* dx, int rows, int D, cudaStream_t stream) {
    DISPATCH(bwd_k, D / 4, rows, stream, x, dy, L, dx, D);
}
}
