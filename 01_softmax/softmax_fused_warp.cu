// Fused softmax forward: float4 loads, row kept in registers, online (m, s)
// with warp-shuffle XOR butterfly. Saves L = m + log(s) for backward.
// Build (Colab T4): nvcc -O3 -arch=sm_75 softmax.cu -o softmax && ./softmax

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include <cuda_runtime.h>

#define FULL 0xffffffffu

struct MS { float m, s; };

__device__ __forceinline__ MS merge(MS a, MS b) {
    float m = fmaxf(a.m, b.m);
    // -FLT_MAX (not -inf) as identity avoids inf - inf = NaN
    return { m, a.s * __expf(a.m - m) + b.s * __expf(b.m - m) };
}

// One block per row. D must be a multiple of 4, rows 16B aligned.
// V = float4 loads per thread = ceil(D / (4 * BLOCK))
template <int BLOCK, int V>
__global__ void softmax_fwd(const float* __restrict__ x,
                            float* __restrict__ y,
                            float* __restrict__ L, int D) {
    const int row = blockIdx.x, tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5, D4 = D >> 2;
    const float4* xr = reinterpret_cast<const float4*>(x + (size_t)row * D);
    float4*       yr = reinterpret_cast<float4*>(y + (size_t)row * D);

    // 1) single global read -> registers, per-thread online (m, s)
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

    // 2) warp butterfly: both partners merge, every lane ends with the result
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        MS q = { __shfl_xor_sync(FULL, p.m, o), __shfl_xor_sync(FULL, p.s, o) };
        p = merge(p, q);
    }

    // 3) cross-warp: each warp merges all warp results itself (no broadcast)
    __shared__ MS sh[BLOCK / 32];
    if (lane == 0) sh[warp] = p;
    __syncthreads();
    MS t = sh[0];
    #pragma unroll
    for (int w = 1; w < BLOCK / 32; ++w) t = merge(t, sh[w]);

    // 4) normalize from registers, single global write
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

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)

int main() {
    constexpr int ROWS = 8192, D = 4096, BLOCK = 256;
    constexpr int V = (D / 4 + BLOCK - 1) / BLOCK;   // = 4
    size_t n = (size_t)ROWS * D;

    std::vector<float> hx(n), hy(n);
    for (size_t i = 0; i < n; ++i) hx[i] = 20.f * (rand() / (float)RAND_MAX) - 10.f;

    float *dx, *dy, *dL;
    CK(cudaMalloc(&dx, n * 4)); CK(cudaMalloc(&dy, n * 4)); CK(cudaMalloc(&dL, ROWS * 4));
    CK(cudaMemcpy(dx, hx.data(), n * 4, cudaMemcpyHostToDevice));

    for (int i = 0; i < 10; ++i) softmax_fwd<BLOCK, V><<<ROWS, BLOCK>>>(dx, dy, dL, D); // warmup
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    const int IT = 100;
    for (int i = 0; i < IT; ++i) softmax_fwd<BLOCK, V><<<ROWS, BLOCK>>>(dx, dy, dL, D);
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b); ms /= IT;
    CK(cudaGetLastError());
    CK(cudaMemcpy(hy.data(), dy, n * 4, cudaMemcpyDeviceToHost));

    // CPU reference check (first 64 rows)
    double maxerr = 0;
    for (int r = 0; r < 64; ++r) {
        const float* xr = &hx[(size_t)r * D];
        double m = -1e30, s = 0;
        for (int i = 0; i < D; ++i) m = fmax(m, xr[i]);
        for (int i = 0; i < D; ++i) s += exp(xr[i] - m);
        for (int i = 0; i < D; ++i)
            maxerr = fmax(maxerr, fabs(exp(xr[i] - m) / s - hy[(size_t)r * D + i]));
    }
    printf("max abs err: %.3e\n", maxerr);
    printf("time: %.3f ms  |  effective BW: %.1f GB/s (1 read + 1 write)\n",
           ms, 2.0 * n * 4 / (ms * 1e-3) / 1e9);
    return 0;
}
