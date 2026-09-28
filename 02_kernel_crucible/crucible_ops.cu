#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float fast_exp(float x) {
    float res;
    asm("ex2.approx.f32 %0, %1;" : "=f"(res) : "f"(x * 1.4426950408889634f));
    return res;
}

/*
 ============================================================================
  1. VECTORIZED IN-REGISTER FP16 SOFTMAX (D = 1024)
 ============================================================================
*/
extern "C" __global__ void softmax_fp16_1024_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int num_rows
) {
    int row = blockIdx.x;
    if (row >= num_rows) return;

    int tid = threadIdx.x; // 0..127
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE; // 0..3

    const float4* in_ptr = (const float4*)(input + row * 1024);
    float4* out_ptr = (float4*)(output + row * 1024);

    float4 raw = in_ptr[tid];
    __half* h_in = (__half*)&raw;

    float r[8];
    float thread_max = -1e30f;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        r[i] = __half2float(h_in[i]);
        thread_max = fmaxf(thread_max, r[i]);
    }

    thread_max = warp_reduce_max(thread_max);

    __shared__ float s_max[4];
    if (lane == 0) s_max[warp_id] = thread_max;
    __syncthreads();

    float row_max = (tid < 4) ? s_max[tid] : -1e30f;
    if (warp_id == 0) {
        row_max = warp_reduce_max(row_max);
        if (lane == 0) s_max[0] = row_max;
    }
    __syncthreads();
    row_max = s_max[0];

    float thread_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        r[i] = fast_exp(r[i] - row_max);
        thread_sum += r[i];
    }

    thread_sum = warp_reduce_sum(thread_sum);

    __shared__ float s_sum[4];
    if (lane == 0) s_sum[warp_id] = thread_sum;
    __syncthreads();

    float row_sum = (tid < 4) ? s_sum[tid] : 0.0f;
    if (warp_id == 0) {
        row_sum = warp_reduce_sum(row_sum);
        if (lane == 0) s_sum[0] = row_sum;
    }
    __syncthreads();
    row_sum = s_sum[0];

    float inv_sum = 1.0f / (row_sum + 1e-12f);

    float4 out_raw;
    __half* h_out = (__half*)&out_raw;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        h_out[i] = __float2half(r[i] * inv_sum);
    }
    out_ptr[tid] = out_raw;
}

/*
 ============================================================================
  2. GENERAL FP16 SOFTMAX (Arbitrary D)
 ============================================================================
*/
extern "C" __global__ void softmax_fp16_general_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int num_rows,
    int dim
) {
    int row = blockIdx.x;
    if (row >= num_rows) return;

    const __half* in_row = input + row * dim;
    __half* out_row = output + row * dim;

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    bool is_half2 = (dim % 2 == 0);
    float thread_max = -1e30f;
    if (is_half2) {
        const __half2* in2 = (const __half2*)in_row;
        int dim2 = dim / 2;
        for (int i = tid; i < dim2; i += bdim) {
            float2 f2 = __half22float2(in2[i]);
            thread_max = fmaxf(thread_max, fmaxf(f2.x, f2.y));
        }
    } else {
        for (int i = tid; i < dim; i += bdim) {
            thread_max = fmaxf(thread_max, __half2float(in_row[i]));
        }
    }
    thread_max = warp_reduce_max(thread_max);

    extern __shared__ float s_buf[];
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int num_warps = bdim / WARP_SIZE;

    if (lane == 0) s_buf[warp_id] = thread_max;
    __syncthreads();

    float row_max = (tid < num_warps) ? s_buf[tid] : -1e30f;
    if (warp_id == 0) {
        row_max = warp_reduce_max(row_max);
        if (lane == 0) s_buf[0] = row_max;
    }
    __syncthreads();
    row_max = s_buf[0];

    float thread_sum = 0.0f;
    if (is_half2) {
        const __half2* in2 = (const __half2*)in_row;
        int dim2 = dim / 2;
        for (int i = tid; i < dim2; i += bdim) {
            float2 f2 = __half22float2(in2[i]);
            thread_sum += fast_exp(f2.x - row_max) + fast_exp(f2.y - row_max);
        }
    } else {
        for (int i = tid; i < dim; i += bdim) {
            thread_sum += fast_exp(__half2float(in_row[i]) - row_max);
        }
    }
    thread_sum = warp_reduce_sum(thread_sum);

    if (lane == 0) s_buf[warp_id] = thread_sum;
    __syncthreads();

    float row_sum = (tid < num_warps) ? s_buf[tid] : 0.0f;
    if (warp_id == 0) {
        row_sum = warp_reduce_sum(row_sum);
        if (lane == 0) s_buf[0] = row_sum;
    }
    __syncthreads();
    row_sum = s_buf[0];

    float inv_sum = 1.0f / (row_sum + 1e-12f);

    if (is_half2) {
        const __half2* in2 = (const __half2*)in_row;
        __half2* out2 = (__half2*)out_row;
        int dim2 = dim / 2;
        for (int i = tid; i < dim2; i += bdim) {
            float2 f2 = __half22float2(in2[i]);
            float p0 = fast_exp(f2.x - row_max) * inv_sum;
            float p1 = fast_exp(f2.y - row_max) * inv_sum;
            out2[i] = __floats2half2_rn(p0, p1);
        }
    } else {
        for (int i = tid; i < dim; i += bdim) {
            float p = fast_exp(__half2float(in_row[i]) - row_max) * inv_sum;
            out_row[i] = __float2half(p);
        }
    }
}

/*
 ============================================================================
  3. GENERAL FP32 SOFTMAX
 ============================================================================
*/
extern "C" __global__ void softmax_fp32_general_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_rows,
    int dim
) {
    int row = blockIdx.x;
    if (row >= num_rows) return;

    const float* in_row = input + row * dim;
    float* out_row = output + row * dim;

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    float thread_max = -1e30f;
    for (int i = tid; i < dim; i += bdim) {
        thread_max = fmaxf(thread_max, in_row[i]);
    }
    thread_max = warp_reduce_max(thread_max);

    extern __shared__ float s_mem[];
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int num_warps = bdim / WARP_SIZE;

    if (lane == 0) s_mem[warp_id] = thread_max;
    __syncthreads();

    float row_max = (tid < num_warps) ? s_mem[tid] : -1e30f;
    if (warp_id == 0) {
        row_max = warp_reduce_max(row_max);
        if (lane == 0) s_mem[0] = row_max;
    }
    __syncthreads();
    row_max = s_mem[0];

    float thread_sum = 0.0f;
    for (int i = tid; i < dim; i += bdim) {
        thread_sum += fast_exp(in_row[i] - row_max);
    }
    thread_sum = warp_reduce_sum(thread_sum);

    if (lane == 0) s_mem[warp_id] = thread_sum;
    __syncthreads();

    float row_sum = (tid < num_warps) ? s_mem[tid] : 0.0f;
    if (warp_id == 0) {
        row_sum = warp_reduce_sum(row_sum);
        if (lane == 0) s_mem[0] = row_sum;
    }
    __syncthreads();
    row_sum = s_mem[0];

    float inv_sum = 1.0f / (row_sum + 1e-12f);
    for (int i = tid; i < dim; i += bdim) {
        out_row[i] = fast_exp(in_row[i] - row_max) * inv_sum;
    }
}

/*
 ============================================================================
  4. SPLIT-KV FLASH-DECODING ATTENTION (Champion Bare-Metal Kernel)
  Partitions tokens across chunks (16 SM blocks) + fast 1-warp final reduction
 ============================================================================
*/
#define MAX_CHUNKS 32

extern "C" __global__ void split_kv_stage1_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    float* __restrict__ partial_out,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    int total_tokens,
    int head_dim,
    int chunk_size,
    float scale
) {
    int chunk_id = blockIdx.x;
    int lane = threadIdx.x;
    int token_start = chunk_id * chunk_size;
    int token_end = token_start + chunk_size;
    if (token_end > total_tokens) token_end = total_tokens;

    int elems = head_dim / WARP_SIZE;
    float q_reg[8];
    #pragma unroll
    for (int e = 0; e < elems; e++) {
        q_reg[e] = __half2float(Q[lane + e * WARP_SIZE]);
    }

    float o_acc[8];
    #pragma unroll
    for (int e = 0; e < elems; e++) {
        o_acc[e] = 0.0f;
    }

    float m_prev = -1e30f;
    float l_prev = 0.0f;

    for (int t = token_start; t < token_end; t++) {
        float dot = 0.0f;
        #pragma unroll
        for (int e = 0; e < elems; e++) {
            float k_val = __half2float(K[t * head_dim + lane + e * WARP_SIZE]);
            dot += q_reg[e] * k_val;
        }

        dot = warp_reduce_sum(dot);
        float score = __shfl_sync(0xffffffff, dot, 0) * scale;

        float m_curr = fmaxf(m_prev, score);
        float alpha = fast_exp(m_prev - m_curr);
        float p = fast_exp(score - m_curr);
        float l_curr = l_prev * alpha + p;

        #pragma unroll
        for (int e = 0; e < elems; e++) {
            float v_val = __half2float(V[t * head_dim + lane + e * WARP_SIZE]);
            o_acc[e] = o_acc[e] * alpha + p * v_val;
        }

        m_prev = m_curr;
        l_prev = l_curr;
    }

    #pragma unroll
    for (int e = 0; e < elems; e++) {
        partial_out[chunk_id * head_dim + lane + e * WARP_SIZE] = o_acc[e];
    }
    if (lane == 0) {
        partial_max[chunk_id] = m_prev;
        partial_sum[chunk_id] = l_prev;
    }
}

// Stage 2: Combine Chunks in 1 Warp
extern "C" __global__ void split_kv_stage2_kernel(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    __half* __restrict__ final_out,
    int num_chunks,
    int head_dim
) {
    int lane = threadIdx.x;
    int elems = head_dim / WARP_SIZE;

    // Find global max across chunks
    float g_max = -1e30f;
    if (lane < num_chunks) {
        g_max = partial_max[lane];
    }
    g_max = warp_reduce_max(g_max);
    g_max = __shfl_sync(0xffffffff, g_max, 0);

    // Compute global denominator and rescale factors
    float g_sum = 0.0f;
    float rescale[MAX_CHUNKS];
    for (int c = 0; c < num_chunks; c++) {
        float factor = fast_exp(partial_max[c] - g_max);
        rescale[c] = factor;
        g_sum += partial_sum[c] * factor;
    }

    float inv_g_sum = 1.0f / (g_sum + 1e-12f);

    // Accumulate rescaled outputs
    #pragma unroll
    for (int e = 0; e < elems; e++) {
        float val = 0.0f;
        int idx = lane + e * WARP_SIZE;
        for (int c = 0; c < num_chunks; c++) {
            val += partial_out[c * head_dim + idx] * rescale[c];
        }
        final_out[idx] = __float2half(val * inv_g_sum);
    }
}

// C-Bridge Invocation Helpers
extern "C" {
    void launch_softmax_fp16(
        const __half* input,
        __half* output,
        int num_rows,
        int dim,
        cudaStream_t stream
    ) {
        if (dim == 1024) {
            softmax_fp16_1024_kernel<<<num_rows, 128, 0, stream>>>(
                input, output, num_rows
            );
        } else {
            int block_size = 256;
            if (dim <= 64) block_size = 32;
            else if (dim <= 128) block_size = 64;
            else if (dim <= 256) block_size = 128;
            else if (dim <= 512) block_size = 256;
            else block_size = 512;

            int shared_mem = (block_size / WARP_SIZE) * sizeof(float);
            softmax_fp16_general_kernel<<<num_rows, block_size, shared_mem, stream>>>(
                input, output, num_rows, dim
            );
        }
    }

    void launch_softmax_fp32(
        const float* input,
        float* output,
        int num_rows,
        int dim,
        cudaStream_t stream
    ) {
        int block_size = 256;
        if (dim <= 64) block_size = 32;
        else if (dim <= 128) block_size = 64;
        else if (dim <= 256) block_size = 128;
        else if (dim <= 512) block_size = 256;
        else block_size = 512;

        int shared_mem = (block_size / WARP_SIZE) * sizeof(float);
        softmax_fp32_general_kernel<<<num_rows, block_size, shared_mem, stream>>>(
            input, output, num_rows, dim
        );
    }

    void launch_split_kv_decode(
        const __half* Q,
        const __half* K,
        const __half* V,
        float* partial_out,
        float* partial_max,
        float* partial_sum,
        __half* final_out,
        int total_tokens,
        int head_dim,
        int num_chunks,
        float scale,
        cudaStream_t stream
    ) {
        int chunk_size = (total_tokens + num_chunks - 1) / num_chunks;
        split_kv_stage1_kernel<<<num_chunks, WARP_SIZE, 0, stream>>>(
            Q, K, V, partial_out, partial_max, partial_sum,
            total_tokens, head_dim, chunk_size, scale
        );
        split_kv_stage2_kernel<<<1, WARP_SIZE, 0, stream>>>(
            partial_out, partial_max, partial_sum,
            final_out, num_chunks, head_dim
        );
    }
}
