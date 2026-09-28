import os
import sys
import time
import subprocess
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.join(BASE_DIR, "workspace")

WORKLOAD_TEMPLATES = {
    "split_kv_d64": {
        "name": "LLM Single-Token Decoding (N=1024, D=64)",
        "description": "Ultra-low latency inference decoding. Partitions 1024 tokens across 16 SMs.",
        "tokens": 1024,
        "head_dim": 64,
        "chunks": 16,
        "test_type": "standard"
    },
    "prefill_long_n8k": {
        "name": "Long-Context LLM Prefill (N=8192, D=64)",
        "description": "Heavy memory bandwidth test. 8,192 tokens across 16 chunks (512 tokens/chunk).",
        "tokens": 8192,
        "head_dim": 64,
        "chunks": 16,
        "test_type": "bandwidth"
    },
    "vision_head_d256": {
        "name": "Vision / Multimodal High-Dim Head (N=1024, D=256)",
        "description": "High register pressure and multi-warp reduction across D=256 channels.",
        "tokens": 1024,
        "head_dim": 256,
        "chunks": 16,
        "test_type": "high_dim"
    },
    "numerical_stress": {
        "name": "Extreme Numerical Stress Test (Range: [-500.0, +500.0])",
        "description": "Tests numerical stability against exponent overflow, underflow, and NaN collapse.",
        "tokens": 1024,
        "head_dim": 64,
        "chunks": 16,
        "test_type": "stress"
    }
}

def generate_workload_cuda(workload_key, unroll=4, cache_stream=0, ptx_mode=1, maxreg=0):
    cfg = WORKLOAD_TEMPLATES.get(workload_key, WORKLOAD_TEMPLATES["split_kv_d64"])
    n_tokens = cfg["tokens"]
    d_dim = cfg["head_dim"]
    chunks = cfg["chunks"]
    chunk_size = n_tokens // chunks
    is_stress = (cfg["test_type"] == "stress")

    code = f"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WARP_SIZE 32
#define D {d_dim}
#define TOTAL_TOKENS {n_tokens}
#define CHUNKS {chunks}
#define CHUNK_SIZE {chunk_size}

__device__ __forceinline__ float warp_reduce_sum(float val) {{
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {{
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }}
    return val;
}}

__device__ __forceinline__ float fast_exp(float x) {{
    float res;
    asm("ex2.approx.f32 %0, %1;" : "=f"(res) : "f"(x * 1.4426950408889634f));
    return res;
}}

extern "C" __global__ void workload_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    float* __restrict__ partial_out,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum
) {{
    int chunk_id = blockIdx.x;
    int lane = threadIdx.x;
    int token_start = chunk_id * CHUNK_SIZE;

    // Head dim elements per thread
    constexpr int ELEMS_PER_THREAD = D / WARP_SIZE;
    float q_reg[ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {{
        q_reg[e] = __half2float(Q[lane + e * WARP_SIZE]);
    }}

    float o_acc[ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {{
        o_acc[e] = 0.0f;
    }}

    float m_prev = -1e20f;
    float l_prev = 0.0f;
    const float scale = 1.0f / sqrtf((float)D);

    #pragma unroll {unroll}
    for (int i = 0; i < CHUNK_SIZE; i++) {{
        int t = token_start + i;
        float dot = 0.0f;

        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {{
            float k_val = __half2float(K[t * D + lane + e * WARP_SIZE]);
#if {ptx_mode} == 1
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dot) : "f"(q_reg[e]), "f"(k_val), "f"(dot));
#else
            dot += q_reg[e] * k_val;
#endif
        }}

        dot = warp_reduce_sum(dot);
        float score = __shfl_sync(0xffffffff, dot, 0) * scale;

        float m_curr = fmaxf(m_prev, score);
        float alpha = fast_exp(m_prev - m_curr);
        float p = fast_exp(score - m_curr);
        float l_curr = l_prev * alpha + p;

        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {{
            float v_val = __half2float(V[t * D + lane + e * WARP_SIZE]);
#if {ptx_mode} == 1
            float o_s;
            asm("fma.rn.f32 %0, %1, %2, 0.0;" : "=f"(o_s) : "f"(o_acc[e]), "f"(alpha));
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(o_acc[e]) : "f"(p), "f"(v_val), "f"(o_s));
#else
            o_acc[e] = o_acc[e] * alpha + p * v_val;
#endif
        }}

        m_prev = m_curr;
        l_prev = l_curr;
    }}

    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {{
        partial_out[chunk_id * D + lane + e * WARP_SIZE] = o_acc[e];
    }}
    if (lane == 0) {{
        partial_max[chunk_id] = m_prev;
        partial_sum[chunk_id] = l_prev;
    }}
}}

int main() {{
    __half *h_Q = (__half*)malloc(D * sizeof(__half));
    __half *h_K = (__half*)malloc(TOTAL_TOKENS * D * sizeof(__half));
    __half *h_V = (__half*)malloc(TOTAL_TOKENS * D * sizeof(__half));

    float mult = {500.0 if is_stress else 1.0}f;
    for (int i = 0; i < D; i++) h_Q[i] = __float2half(sinf(i * 0.1f) * mult);
    for (int i = 0; i < TOTAL_TOKENS * D; i++) {{
        h_K[i] = __float2half(cosf(i * 0.05f) * mult);
        h_V[i] = __float2half(sinf(i * 0.03f) * mult);
    }}

    __half *d_Q, *d_K, *d_V;
    float *d_out, *d_max, *d_sum;
    cudaMalloc(&d_Q, D * sizeof(__half));
    cudaMalloc(&d_K, TOTAL_TOKENS * D * sizeof(__half));
    cudaMalloc(&d_V, TOTAL_TOKENS * D * sizeof(__half));
    cudaMalloc(&d_out, CHUNKS * D * sizeof(float));
    cudaMalloc(&d_max, CHUNKS * sizeof(float));
    cudaMalloc(&d_sum, CHUNKS * sizeof(float));

    cudaMemcpy(d_Q, h_Q, D * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, TOTAL_TOKENS * D * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, TOTAL_TOKENS * D * sizeof(__half), cudaMemcpyHostToDevice);

    for (int i = 0; i < 20; i++) {{
        workload_kernel<<<CHUNKS, WARP_SIZE>>>(d_Q, d_K, d_V, d_out, d_max, d_sum);
    }}
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iters = 200;
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {{
        workload_kernel<<<CHUNKS, WARP_SIZE>>>(d_Q, d_K, d_V, d_out, d_max, d_sum);
    }}
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_us = (total_ms / iters) * 1000.0f;

    float *h_max = (float*)malloc(CHUNKS * sizeof(float));
    cudaMemcpy(h_max, d_max, CHUNKS * sizeof(float), cudaMemcpyDeviceToHost);
    int valid = 1;
    for (int i = 0; i < CHUNKS; i++) {{
        if (isnan(h_max[i]) || isinf(h_max[i])) valid = 0;
    }}

    if (valid) {{
        printf("RESULT_SUCCESS:%f\\n", avg_us);
    }} else {{
        printf("RESULT_NAN:0\\n");
    }}

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_out); cudaFree(d_max); cudaFree(d_sum);
    free(h_Q); free(h_K); free(h_V); free(h_max);
    return 0;
}}
"""
    return code
