#include <cstdio>
#include <cuda_runtime.h>

__global__ void l2_softmax(const float* __restrict__ x, float* __restrict__ y, int rows, int D) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;
    const float* xr = x + row * D;
    float* yr = y + row * D;

    float val = xr[tid];
    // Intra-warp max & sum
    float m = val;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, m, o);
        m = fmaxf(m, other);
    }
    float s = __expf(val - m);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        s += __shfl_xor_sync(0xffffffff, s, o);
    }
    yr[tid] = __expf(val - m) / s;
}

int main() {
    // 32 threads, 1024 rows -> 1024 * 32 * 4 = 128 KB (easily fits inside 1024 KB L2 cache!)
    constexpr int ROWS = 8192, D = 32;
    size_t bytes = ROWS * D * sizeof(float);
    float *dx, *dy;
    cudaMalloc(&dx, bytes);
    cudaMalloc(&dy, bytes);

    // Warm L2 cache
    for (int i = 0; i < 50; ++i) l2_softmax<<<ROWS, 32>>>(dx, dy, ROWS, D);
    cudaDeviceSynchronize();

    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    const int IT = 1000;
    for (int i = 0; i < IT; ++i) {
        l2_softmax<<<ROWS, 32>>>(dx, dy, ROWS, D);
    }
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); ms /= IT;

    double bw = (2.0 * bytes) / (ms * 1e-3) / 1e9;
    printf("L2-Resident Softmax (Fits in 1MB L2 cache):\n");
    printf("  Latency: %.4f ms\n", ms);
    printf("  Effective Bandwidth: %.1f GB/s\n", bw);
    return 0;
}
