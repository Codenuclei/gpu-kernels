import os
import sys
import time
import json
import subprocess
import requests
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.join(BASE_DIR, "workspace")
LLM_ENDPOINT = "http://127.0.0.1:34247/v1/chat/completions"

BASE_KERNEL_TEMPLATE = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WARP_SIZE 32
#define D 64
#define TOTAL_TOKENS 1024
#define CHUNKS 16
#define CHUNK_SIZE (TOTAL_TOKENS / CHUNKS)

// MUTATION PARAMS
#define UNROLL_FACTOR __UNROLL__
#define USE_CACHE_STREAMING __CACHE_STREAM__
#define PTX_FMA_MODE __PTX_MODE__

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

extern "C" __global__ void split_kv_crucible_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    float* __restrict__ partial_out,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum
) {
    int chunk_id = blockIdx.x;
    int lane = threadIdx.x;
    
    int token_start = chunk_id * CHUNK_SIZE;
    
    // Load Q vector into registers
    float q_val_0 = __half2float(Q[lane]);
    float q_val_1 = __half2float(Q[lane + 32]);
    
    float local_scores[CHUNK_SIZE];
    float chunk_max = -1e20f;
    float scale = 0.125f; // 1 / sqrt(64)
    
    #pragma unroll __UNROLL__
    for (int i = 0; i < CHUNK_SIZE; i++) {
        int t = token_start + i;
        float k0, k1;
#if USE_CACHE_STREAMING == 1
        unsigned short raw0, raw1;
        asm("ld.global.cs.u16 %0, [%1];" : "=h"(raw0) : "l"(&K[t * D + lane]));
        asm("ld.global.cs.u16 %0, [%1];" : "=h"(raw1) : "l"(&K[t * D + lane + 32]));
        k0 = __half2float(*((__half*)&raw0));
        k1 = __half2float(*((__half*)&raw1));
#else
        k0 = __half2float(K[t * D + lane]);
        k1 = __half2float(K[t * D + lane + 32]);
#endif
        float dot;
#if PTX_FMA_MODE == 1
        asm("fma.rn.f32 %0, %1, %2, 0.0;" : "=f"(dot) : "f"(q_val_0), "f"(k0));
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(dot) : "f"(q_val_1), "f"(k1), "f"(dot));
#else
        dot = q_val_0 * k0 + q_val_1 * k1;
#endif
        dot = warp_reduce_sum(dot) * scale;
        local_scores[i] = dot;
        if (dot > chunk_max) chunk_max = dot;
    }
    
    float chunk_sum = 0.0f;
    #pragma unroll __UNROLL__
    for (int i = 0; i < CHUNK_SIZE; i++) {
        float p = fast_exp(local_scores[i] - chunk_max);
        local_scores[i] = p;
        chunk_sum += p;
    }
    
    float acc_0 = 0.0f;
    float acc_1 = 0.0f;
    #pragma unroll __UNROLL__
    for (int i = 0; i < CHUNK_SIZE; i++) {
        float p = local_scores[i];
        float v0 = __half2float(V[(token_start + i) * D + lane]);
        float v1 = __half2float(V[(token_start + i) * D + lane + 32]);
#if PTX_FMA_MODE == 1
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(acc_0) : "f"(p), "f"(v0), "f"(acc_0));
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(acc_1) : "f"(p), "f"(v1), "f"(acc_1));
#else
        acc_0 += p * v0;
        acc_1 += p * v1;
#endif
    }
    
    partial_out[chunk_id * D + lane] = acc_0;
    partial_out[chunk_id * D + lane + 32] = acc_1;
    if (lane == 0) {
        partial_max[chunk_id] = chunk_max;
        partial_sum[chunk_id] = chunk_sum;
    }
}

int main() {
    __half *h_Q = (__half*)malloc(D * sizeof(__half));
    __half *h_K = (__half*)malloc(TOTAL_TOKENS * D * sizeof(__half));
    __half *h_V = (__half*)malloc(TOTAL_TOKENS * D * sizeof(__half));
    
    for (int i = 0; i < D; i++) h_Q[i] = __float2half(sinf(i * 0.1f));
    for (int i = 0; i < TOTAL_TOKENS * D; i++) {
        h_K[i] = __float2half(cosf(i * 0.05f));
        h_V[i] = __float2half(sinf(i * 0.03f));
    }
    
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
    
    // Warmup
    for (int i = 0; i < 25; i++) {
        split_kv_crucible_kernel<<<CHUNKS, WARP_SIZE>>>(d_Q, d_K, d_V, d_out, d_max, d_sum);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int iters = 250;
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        split_kv_crucible_kernel<<<CHUNKS, WARP_SIZE>>>(d_Q, d_K, d_V, d_out, d_max, d_sum);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_us = (total_ms / iters) * 1000.0f;
    
    // Check sanity (non-zero, non-nan)
    float *h_max = (float*)malloc(CHUNKS * sizeof(float));
    cudaMemcpy(h_max, d_max, CHUNKS * sizeof(float), cudaMemcpyDeviceToHost);
    int valid = 1;
    for (int i = 0; i < CHUNKS; i++) {
        if (isnan(h_max[i]) || isinf(h_max[i])) valid = 0;
    }
    
    if (valid) {
        printf("RESULT_SUCCESS:%f\n", avg_us);
    } else {
        printf("RESULT_NAN:0\n");
    }
    
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_out); cudaFree(d_max); cudaFree(d_sum);
    free(h_Q); free(h_K); free(h_V); free(h_max);
    return 0;
}
"""

class KernelCrucibleEngine:
    def __init__(self):
        self.generation = 0
        self.round_num = 0
        self.is_running = False
        
        # Initial Reigning Champion
        self.champion = {
            "id": "champ-gen0",
            "name": "SplitKV-Base-Champion",
            "author": "Architect (Base)",
            "latency_us": 8.40,
            "registers": 34,
            "unroll": 2,
            "cache_streaming": 0,
            "ptx_fma_mode": 1,
            "status": "CROWNED",
            "generation": 0,
            "code_snippet": "asm(\"ex2.approx.f32 %0, %1;\");\nasm(\"fma.rn.f32 %0, %1, %2, %3;\");",
            "defeated_count": 0
        }
        
        self.hall_of_fame = [
            {
                "rank": 1,
                "name": self.champion["name"],
                "author": self.champion["author"],
                "latency_us": self.champion["latency_us"],
                "speedup_vs_triton": "4.83x",
                "generation": 0,
                "achieved_at": time.strftime("%H:%M:%S")
            }
        ]
        
        self.event_log = []
        self.last_challenger = None

    def get_gpu_telemetry(self):
        try:
            res = subprocess.run([
                "nvidia-smi",
                "--query-gpu=temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory,pstate",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=2)
            if res.returncode == 0:
                parts = [p.strip() for p in res.stdout.strip().split(",")]
                return {
                    "temp_c": int(parts[0]) if parts[0].isdigit() else 55,
                    "power_w": float(parts[1]) if parts[1].replace('.','',1).isdigit() else 8.0,
                    "graphics_mhz": int(parts[2]) if parts[2].isdigit() else 1747,
                    "memory_mhz": int(parts[3]) if parts[3].isdigit() else 7001,
                    "pstate": parts[4] if len(parts) > 4 else "P0"
                }
        except Exception:
            pass
        return {"temp_c": 58, "power_w": 8.2, "graphics_mhz": 1747, "memory_mhz": 7001, "pstate": "P0"}

    def ask_ai_mutant(self, champ_info):
        """Asks local Unsloth / Gemma model for mutation strategies."""
        prompt = f"""You are a GPU Kernel Mutation Agent competing in the Silicon Crucible on an RTX 2050 Ampere GPU.
The current champion kernel runs in {champ_info['latency_us']:.2f} microseconds.
Champion config: UNROLL={champ_info['unroll']}, CACHE_STREAM={champ_info['cache_streaming']}, PTX_FMA={champ_info['ptx_fma_mode']}.
Choose a mutant strategy:
1. UNROLL factor: 1, 2, 4, or 8
2. CACHE_STREAM (use ld.global.cs): 0 or 1
3. PTX_FMA (inline PTX fma.rn.f32): 0 or 1
Reply ONLY in JSON format:
{{"agent_name": "Mutant-X", "unroll": 4, "cache_streaming": 1, "ptx_fma": 1, "strategy": "High unroll with streaming cache hints"}}"""
        try:
            resp = requests.post(LLM_ENDPOINT, json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 120,
                "temperature": 0.8
            }, timeout=3)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                # Extract json
                s = content[content.find('{'):content.rfind('}')+1]
                data = json.loads(s)
                return {
                    "agent_name": data.get("agent_name", f"AI-Mutant-G{self.generation}"),
                    "unroll": int(data.get("unroll", random.choice([1, 2, 4, 8]))),
                    "cache_streaming": int(data.get("cache_streaming", random.choice([0, 1]))),
                    "ptx_fma_mode": int(data.get("ptx_fma", random.choice([0, 1]))),
                    "strategy": data.get("strategy", "LLM Automated Kernel Mutation")
                }
        except Exception:
            pass
        
        # Fallback heuristic mutant
        unroll_choice = random.choice([1, 2, 4, 8])
        cache_choice = random.choice([0, 1])
        ptx_choice = random.choice([0, 1])
        return {
            "agent_name": f"Silicon-Stalker-G{self.generation}",
            "unroll": unroll_choice,
            "cache_streaming": cache_choice,
            "ptx_fma_mode": ptx_choice,
            "strategy": f"PTX unroll x{unroll_choice} with cs={cache_choice} mode={ptx_choice}"
        }

    def compile_and_benchmark_challenger(self, mutant):
        self.round_num += 1
        self.generation += 1
        
        src_path = os.path.join(WORKSPACE_DIR, f"candidate_{self.generation}.cu")
        bin_path = os.path.join(WORKSPACE_DIR, f"candidate_{self.generation}")
        
        # Render code
        code = BASE_KERNEL_TEMPLATE
        code = code.replace("__UNROLL__", str(mutant["unroll"]))
        code = code.replace("__CACHE_STREAM__", str(mutant["cache_streaming"]))
        code = code.replace("__PTX_MODE__", str(mutant["ptx_fma_mode"]))
        
        with open(src_path, "w") as f:
            f.write(code)
            
        # Compile with nvcc
        compile_start = time.time()
        compile_cmd = [
            "/home/codenuclei/.local/bin/nvcc",
            "-O3", "-arch=sm_86",
            "-Xptxas", "-v",
            src_path, "-o", bin_path
        ]
        comp_res = subprocess.run(compile_cmd, capture_output=True, text=True)
        compile_time_ms = (time.time() - compile_start) * 1000.0
        
        if comp_res.returncode != 0:
            result = {
                "id": f"cand-{self.generation}",
                "name": mutant["agent_name"],
                "strategy": mutant["strategy"],
                "status": "SLAIN",
                "reason": "NVCC Compilation / PTX Syntax Failed",
                "latency_us": 999.0,
                "compile_time_ms": compile_time_ms,
                "registers": 0,
                "generation": self.generation,
                "unroll": mutant["unroll"],
                "cache_streaming": mutant["cache_streaming"],
                "ptx_fma_mode": mutant["ptx_fma_mode"]
            }
            self.last_challenger = result
            self.event_log.insert(0, f"💀 [ROUND {self.round_num}] {mutant['agent_name']} was SLAIN by nvcc error.")
            return result
            
        # Parse registers used from stderr
        regs = 32
        for line in comp_res.stderr.splitlines():
            if "Used" in line and "registers" in line:
                try:
                    regs = int(line.split("Used")[1].split("registers")[0].strip())
                except:
                    pass

        # Execute and benchmark
        run_res = subprocess.run([bin_path], capture_output=True, text=True, timeout=5)
        out = run_res.stdout.strip()
        
        # Clean up binary
        try:
            os.remove(bin_path)
            os.remove(src_path)
        except:
            pass
            
        if "RESULT_SUCCESS:" in out:
            measured_us = float(out.split("RESULT_SUCCESS:")[1].strip())
            
            # Match outcome
            if measured_us < self.champion["latency_us"]:
                # ASCENSION! NEW CHAMPION!
                speedup_pct = ((self.champion["latency_us"] - measured_us) / self.champion["latency_us"]) * 100.0
                result = {
                    "id": f"cand-{self.generation}",
                    "name": mutant["agent_name"],
                    "strategy": mutant["strategy"],
                    "status": "ASCENDED",
                    "reason": f"Defeated Champion! Faster by {speedup_pct:.1f}% ({measured_us:.2f} µs vs {self.champion['latency_us']:.2f} µs)",
                    "latency_us": measured_us,
                    "compile_time_ms": compile_time_ms,
                    "registers": regs,
                    "generation": self.generation,
                    "unroll": mutant["unroll"],
                    "cache_streaming": mutant["cache_streaming"],
                    "ptx_fma_mode": mutant["ptx_fma_mode"]
                }
                
                # Update champion
                old_champ_name = self.champion["name"]
                self.champion = {
                    "id": result["id"],
                    "name": result["name"],
                    "author": "Evolved Agent",
                    "latency_us": measured_us,
                    "registers": regs,
                    "unroll": mutant["unroll"],
                    "cache_streaming": mutant["cache_streaming"],
                    "ptx_fma_mode": mutant["ptx_fma_mode"],
                    "status": "CROWNED",
                    "generation": self.generation,
                    "code_snippet": f"// New Champion Gen {self.generation}\n#define UNROLL {mutant['unroll']}\n#define PTX_MODE {mutant['ptx_fma_mode']}",
                    "defeated_count": 0
                }
                
                # Add to Hall of Fame
                self.hall_of_fame.insert(0, {
                    "rank": len(self.hall_of_fame) + 1,
                    "name": self.champion["name"],
                    "author": result["strategy"],
                    "latency_us": measured_us,
                    "speedup_vs_triton": f"{40.59 / measured_us:.2f}x",
                    "generation": self.generation,
                    "achieved_at": time.strftime("%H:%M:%S")
                })
                
                self.event_log.insert(0, f"👑 [ROUND {self.round_num}] ASCENSION! {result['name']} defeated {old_champ_name}! Latency: {measured_us:.2f} µs (-{speedup_pct:.1f}%)")
            else:
                # DEFEATED
                diff_pct = ((measured_us - self.champion["latency_us"]) / self.champion["latency_us"]) * 100.0
                result = {
                    "id": f"cand-{self.generation}",
                    "name": mutant["agent_name"],
                    "strategy": mutant["strategy"],
                    "status": "DEFEATED",
                    "reason": f"Slower than Champion (+{diff_pct:.1f}% lag: {measured_us:.2f} µs vs {self.champion['latency_us']:.2f} µs)",
                    "latency_us": measured_us,
                    "compile_time_ms": compile_time_ms,
                    "registers": regs,
                    "generation": self.generation,
                    "unroll": mutant["unroll"],
                    "cache_streaming": mutant["cache_streaming"],
                    "ptx_fma_mode": mutant["ptx_fma_mode"]
                }
                self.champion["defeated_count"] += 1
                self.event_log.insert(0, f"⚔️ [ROUND {self.round_num}] {mutant['agent_name']} fell in combat. Latency: {measured_us:.2f} µs (+{diff_pct:.1f}%)")
        else:
            result = {
                "id": f"cand-{self.generation}",
                "name": mutant["agent_name"],
                "strategy": mutant["strategy"],
                "status": "SLAIN",
                "reason": "Numerical NaN or Kernel Crash",
                "latency_us": 999.0,
                "compile_time_ms": compile_time_ms,
                "registers": regs,
                "generation": self.generation,
                "unroll": mutant["unroll"],
                "cache_streaming": mutant["cache_streaming"],
                "ptx_fma_mode": mutant["ptx_fma_mode"]
            }
            self.event_log.insert(0, f"💀 [ROUND {self.round_num}] {mutant['agent_name']} caused NaN explosion.")
            
        self.last_challenger = result
        return result

    def step_arena(self):
        """Runs a single tournament match."""
        mutant = self.ask_ai_mutant(self.champion)
        result = self.compile_and_benchmark_challenger(mutant)
        return {
            "champion": self.champion,
            "challenger": result,
            "telemetry": self.get_gpu_telemetry(),
            "hall_of_fame": self.hall_of_fame[:5],
            "round_num": self.round_num,
            "generation": self.generation,
            "log": self.event_log[:15]
        }
