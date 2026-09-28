#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include <cuda_runtime.h>
#include <cudnn.h>

#define FULL 0xffffffffu

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)

#define CUDNN_CHECK(x) do { cudnnStatus_t s = (x); if (s != CUDNN_STATUS_SUCCESS) { \
    printf("cuDNN error %s at %s:%d\n", cudnnGetErrorString(s), __FILE__, __LINE__); exit(1);} } while (0)

struct MS { float m, s; };

// -------------------------------------------------------------
// 1. Inline PTX Primitives tailored to Ampere GA107 (RTX 2050)
// -------------------------------------------------------------

// Load float4 with Cache Streaming policy (.cs: evict-first from 1MB L2 cache)
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

// Store float4 with Cache Streaming policy (.cs: bypass polluting L1/L2)
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

// SFU hardware exp2 with Flush-To-Zero (.ftz): exp(x) = 2^(x * log2(e))
__device__ __forceinline__ float ptx_fast_exp(float x) {
    constexpr float LOG2E = 1.4426950408889634f;
    float t = x * LOG2E;
    float res;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(res) : "f"(t));
    return res;
}

// Hardware butterfly warp shuffle
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

// -------------------------------------------------------------
// 2. Full Inline PTX Fused Softmax Kernel
// -------------------------------------------------------------
template <int BLOCK, int V>
__global__ void softmax_fwd_ptx(
    const float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ L,
    int rows, int D
) {
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int D4 = D >> 2;
    const int stride = gridDim.x;

    for (int row = blockIdx.x; row < rows; row += stride) {
        const float4* xr = reinterpret_cast<const float4*>(x + (size_t)row * D);
        float4*       yr = reinterpret_cast<float4*>(y + (size_t)row * D);

        float4 r[V];
        MS p = { -FLT_MAX, 0.f };

        #pragma unroll
        for (int v = 0; v < V; ++v) {
            int idx = tid + v * BLOCK;
            if (idx < D4) {
                // Inline PTX streaming load (bypasses L2 pollution)
                r[v] = ptx_ld_cs_f4(xr + idx);

                float lm = fmaxf(fmaxf(r[v].x, r[v].y), fmaxf(r[v].z, r[v].w));
                MS q = { lm, ptx_fast_exp(r[v].x - lm) + ptx_fast_exp(r[v].y - lm) +
                             ptx_fast_exp(r[v].z - lm) + ptx_fast_exp(r[v].w - lm) };
                p = ptx_merge(p, q);
            }
        }

        // Intra-warp butterfly reduction via PTX
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

        const float inv = 1.f / t.s;
        #pragma unroll
        for (int v = 0; v < V; ++v) {
            int idx = tid + v * BLOCK;
            if (idx < D4) {
                float4 o4;
                o4.x = ptx_fast_exp(r[v].x - t.m) * inv;
                o4.y = ptx_fast_exp(r[v].y - t.m) * inv;
                o4.z = ptx_fast_exp(r[v].z - t.m) * inv;
                o4.w = ptx_fast_exp(r[v].w - t.m) * inv;

                // Inline PTX streaming write
                ptx_st_cs_f4(yr + idx, o4);
            }
        }

        if (tid == 0 && L != nullptr) {
            L[row] = t.m + logf(t.s);
        }
    }
}

// -------------------------------------------------------------
// 3. cuDNN Softmax Runner
// -------------------------------------------------------------
struct CudnnSoftmaxRunner {
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t desc;
    float alpha = 1.0f, beta = 0.0f;

    CudnnSoftmaxRunner(int rows, int D) {
        CUDNN_CHECK(cudnnCreate(&handle));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
        // Shape: [rows, D, 1, 1]
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, rows, D, 1, 1
        ));
    }

    ~CudnnSoftmaxRunner() {
        cudnnDestroyTensorDescriptor(desc);
        cudnnDestroy(handle);
    }

    void run(const float* x, float* y) {
        CUDNN_CHECK(cudnnSoftmaxForward(
            handle,
            CUDNN_SOFTMAX_ACCURATE,
            CUDNN_SOFTMAX_MODE_INSTANCE, // Softmax per row [N x D]
            &alpha,
            desc, x,
            &beta,
            desc, y
        ));
    }
};

// -------------------------------------------------------------
// 4. Benchmark Harness
// -------------------------------------------------------------
void run_comparison(int ROWS, int D) {
    size_t n = (size_t)ROWS * D;
    size_t bytes = n * sizeof(float);

    std::vector<float> hx(n), hy_ptx(n), hy_cudnn(n);
    for (size_t i = 0; i < n; ++i) {
        hx[i] = 20.f * (rand() / (float)RAND_MAX) - 10.f;
    }

    float *dx, *dy_ptx, *dy_cudnn, *dL;
    CUDA_CHECK(cudaMalloc(&dx, bytes));
    CUDA_CHECK(cudaMalloc(&dy_ptx, bytes));
    CUDA_CHECK(cudaMalloc(&dy_cudnn, bytes));
    CUDA_CHECK(cudaMalloc(&dL, ROWS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));

    CudnnSoftmaxRunner cudnn_runner(ROWS, D);

    // Warmup
    constexpr int BLOCK = 256;
    int V = (D / 4 + BLOCK - 1) / BLOCK;
    int grid = ROWS > 1024 ? 1024 : ROWS;

    for (int i = 0; i < 15; ++i) {
        if      (V <= 1) softmax_fwd_ptx<BLOCK, 1><<<grid, BLOCK>>>(dx, dy_ptx, dL, ROWS, D);
        else if (V <= 2) softmax_fwd_ptx<BLOCK, 2><<<grid, BLOCK>>>(dx, dy_ptx, dL, ROWS, D);
        else if (V <= 4) softmax_fwd_ptx<BLOCK, 4><<<grid, BLOCK>>>(dx, dy_ptx, dL, ROWS, D);
        else             softmax_fwd_ptx<BLOCK, 8><<<grid, BLOCK>>>(dx, dy_ptx, dL, ROWS, D);
        cudnn_runner.run(dx, dy_cudnn);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure PTX
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int ITERS = 150;

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        if      (V <= 1) softmax_fwd_ptx<BLOCK, 1><<<grid, BLOCK>>>(dx, dy_ptx, dL, ROWS, D);
        else if (V <= 2) softmax_fwd_ptx<BLOCK, 2><<<grid, BLOCK>>>(dx, dy_ptx, dL, ROWS, D);
        else if (V <= 4) softmax_fwd_ptx<BLOCK, 4><<<grid, BLOCK>>>(dx, dy_ptx, dL, ROWS, D);
        else             softmax_fwd_ptx<BLOCK, 8><<<grid, BLOCK>>>(dx, dy_ptx, dL, ROWS, D);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_ptx;
    CUDA_CHECK(cudaEventElapsedTime(&ms_ptx, start, stop));
    ms_ptx /= ITERS;

    // Measure cuDNN
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        cudnn_runner.run(dx, dy_cudnn);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_cudnn;
    CUDA_CHECK(cudaEventElapsedTime(&ms_cudnn, start, stop));
    ms_cudnn /= ITERS;

    // Accuracy check: PTX vs cuDNN
    CUDA_CHECK(cudaMemcpy(hy_ptx.data(), dy_ptx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hy_cudnn.data(), dy_cudnn, bytes, cudaMemcpyDeviceToHost));

    double max_diff = 0;
    for (size_t i = 0; i < 4096 && i < n; ++i) {
        max_diff = fmax(max_diff, fabs(hy_ptx[i] - hy_cudnn[i]));
    }

    double gb_transferred = 2.0 * bytes / 1e9;
    double bw_ptx = gb_transferred / (ms_ptx * 1e-3);
    double bw_cudnn = gb_transferred / (ms_cudnn * 1e-3);
    double win_vs_cudnn = ((ms_cudnn / ms_ptx) - 1.0) * 100.0;

    printf("%-16s | %7.3f ms (%5.1f GB/s) | %7.3f ms (%5.1f GB/s) | %5.2fx (%+5.1f%%) | %.2e\n",
           (std::to_string(ROWS) + "x" + std::to_string(D)).c_str(),
           ms_ptx, bw_ptx,
           ms_cudnn, bw_cudnn,
           ms_cudnn / ms_ptx, win_vs_cudnn,
           max_diff);

    cudaFree(dx); cudaFree(dy_ptx); cudaFree(dy_cudnn); cudaFree(dL);
}

int main() {
    printf("===================================================================================================\n");
    printf("              INLINE PTX (Cache-Streaming & SFU Fast Math) vs NVIDIA cuDNN 9.19                   \n");
    printf("                                  Target: RTX 2050 (sm_86)                                         \n");
    printf("===================================================================================================\n");
    printf("%-16s | %-24s | %-24s | %-16s | %-8s\n",
           "Shape", "Inline PTX (Ours)", "NVIDIA cuDNN 9.19", "Win vs cuDNN", "Max Diff");
    printf("---------------------------------------------------------------------------------------------------\n");

    run_comparison(8192, 1024);
    run_comparison(16384, 1024);
    run_comparison(8192, 2048);
    run_comparison(4096, 4096);
    run_comparison(8192, 4096);
    run_comparison(2048, 8192);
    run_comparison(4096, 8192);

    printf("===================================================================================================\n");
    return 0;
}
