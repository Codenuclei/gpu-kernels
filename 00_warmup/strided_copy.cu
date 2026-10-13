// nvcc -O3 -arch=sm_86 strided_copy.cu -o sc && ./sc
#include <cstdio>
#include <cuda_runtime.h>

// Every element is copied exactly once, but neighbouring threads
// in a warp touch addresses `stride` floats apart.
__global__ void strided_copy(const float *in, float *out, int n, int stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int rows = n / stride;
  int idx = (i % rows) * stride + (i / rows);
  out[idx] = in[idx];
}

int main() {
  const int n = 1 << 24; // 16M floats = 64 MB per array
  float *in, *out;
  cudaMalloc(&in, n * sizeof(float));
  cudaMalloc(&out, n * sizeof(float));
  cudaMemset(in, 0, n * sizeof(float));

  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);

  int strides[] = {1, 2, 4, 8, 16, 32, 64};
  for (int s : strides) {
    strided_copy<<<n / 256, 256>>>(in, out, n, s); // warm-up
    cudaEventRecord(a);
    for (int r = 0; r < 10; r++)
      strided_copy<<<n / 256, 256>>>(in, out, n, s);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    ms /= 10;
    double gbs = 2.0 * n * sizeof(float) / (ms * 1e6); // read + write
    printf("stride %2d: %7.3f ms  %6.1f GB/s useful\n", s, ms, gbs);
  }
  cudaFree(in);
  cudaFree(out);
  return 0;
}
