import os
import sys
import time
import json
import random
import requests
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from kernel_gym_env import KernelOptimizationEnv
from test_suites import WORKLOAD_TEMPLATES, generate_workload_cuda

LLM_ENDPOINT = "http://127.0.0.1:34247/v1/chat/completions"

class KernelCrucibleEngine:
    def __init__(self):
        self.gym_env = KernelOptimizationEnv()
        self.obs, _ = self.gym_env.reset()
        self.current_workload = "split_kv_d64"
        
        self.generation = 0
        self.round_num = 0
        self.total_reward = 0.0
        
        init_lat = float(self.gym_env.champion_latency)
        self.champion = {
            "id": "champ-gen0",
            "name": self.gym_env.champion_name,
            "author": "Hardware Initial Benchmark",
            "latency_us": init_lat,
            "registers": 32,
            "unroll": 1,
            "cache_streaming": 0,
            "ptx_fma_mode": 0,
            "status": "CROWNED",
            "generation": 0,
            "code_snippet": "// Initial hardware measured baseline",
            "defeated_count": 0
        }
        
        self.hall_of_fame = [
            {
                "rank": 1,
                "name": self.champion["name"],
                "author": self.champion["author"],
                "latency_us": init_lat,
                "speedup_vs_triton": "1.00x",
                "generation": 0,
                "achieved_at": time.strftime("%H:%M:%S")
            }
        ]
        
        self.event_log = []
        self.last_challenger = None

    def set_workload(self, workload_key):
        if workload_key not in WORKLOAD_TEMPLATES:
            return False
        self.current_workload = workload_key
        w_cfg = WORKLOAD_TEMPLATES[workload_key]
        
        # Benchmark initial unoptimized baseline on silicon
        code = generate_workload_cuda(workload_key, unroll=1, cache_stream=0, ptx_mode=0)
        src = os.path.join(WORKSPACE_DIR, f"bench_{workload_key}.cu")
        binary = os.path.join(WORKSPACE_DIR, f"bench_{workload_key}")
        with open(src, "w") as f:
            f.write(code)
        subprocess.run(["/home/codenuclei/.local/bin/nvcc", "-O3", "-arch=sm_86", src, "-o", binary], capture_output=True)
        try:
            r = subprocess.run([binary], capture_output=True, text=True, timeout=5)
            out = r.stdout.strip()
            measured_base = float(out.split("RESULT_SUCCESS:")[1].strip())
        except Exception:
            measured_base = 30.0
            
        try:
            if os.path.exists(src): os.remove(src)
            if os.path.exists(binary): os.remove(binary)
        except:
            pass
            
        self.gym_env.target_baseline_us = measured_base
        self.gym_env.champion_latency = measured_base
        self.gym_env.champion_name = f"{w_cfg['name']} (Base)"
        
        self.champion = {
            "id": f"champ-{workload_key}-base",
            "name": f"{w_cfg['name']} (Base)",
            "author": "Hardware Initial Benchmark",
            "latency_us": measured_base,
            "registers": 32,
            "unroll": 1,
            "cache_streaming": 0,
            "ptx_fma_mode": 0,
            "status": "CROWNED",
            "generation": 0,
            "code_snippet": f"// Workload: {w_cfg['name']}\n// Baseline: {measured_base:.2f} us",
            "defeated_count": 0
        }
        self.hall_of_fame = [{
            "rank": 1,
            "name": self.champion["name"],
            "author": "Hardware Baseline",
            "latency_us": measured_base,
            "speedup_vs_triton": "1.00x",
            "generation": 0,
            "achieved_at": time.strftime("%H:%M:%S")
        }]
        self.last_challenger = None
        self.event_log.insert(0, f"🌐 Switched Arena Workload to: {w_cfg['name']} (Baseline: {measured_base:.2f} µs)!")
        return True

    def get_gpu_telemetry(self):
        temp, power = self.gym_env._get_telemetry()
        return {
            "temp_c": int(temp),
            "power_w": float(power),
            "graphics_mhz": 1747,
            "memory_mhz": 7001,
            "pstate": "P0"
        }

    def ask_ai_policy_action(self):
        """Asks local LLM (Unsloth Gemma-2B) for RL policy action [unroll_idx, cs_idx, ptx_idx, reg_idx]."""
        prompt = f"""You are an RL Agent controlling a Gymnasium Environment on an RTX 2050 Mobile GPU.
Target: Optimize Split-KV Fused Attention kernel to beat the champion ({self.gym_env.champion_latency:.2f} us).
Choose action indices:
- unroll_idx (0: 1x, 1: 2x, 2: 4x, 3: 8x)
- cs_idx (0: L1 cached, 1: ld.global.cs streaming)
- ptx_idx (0: C++, 1: inline PTX fma.rn.f32)
- reg_idx (0: unconstrained, 1: maxreg 32, 2: maxreg 48)
Reply ONLY in JSON:
{{"agent_name": "RL-Policy-Agent", "action": [1, 0, 1, 0], "reasoning": "Keep L1 cache warm with inline PTX FMA"}}"""
        try:
            resp = requests.post(LLM_ENDPOINT, json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0.7
            }, timeout=2.5)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                s = content[content.find('{'):content.rfind('}')+1]
                data = json.loads(s)
                act = data.get("action", [1, 0, 1, 0])
                if len(act) == 4:
                    return {
                        "name": data.get("agent_name", f"RL-Agent-G{self.generation}"),
                        "action": [int(act[0])%4, int(act[1])%2, int(act[2])%2, int(act[3])%3],
                        "strategy": data.get("reasoning", "LLM RL Policy Step")
                    }
        except Exception:
            pass
            
        # Sample from Gymnasium action space
        sampled = self.gym_env.action_space.sample()
        return {
            "name": f"Gym-Explorer-G{self.generation}",
            "action": [int(x) for x in sampled],
            "strategy": f"Gym Action Space MultiDiscrete({sampled.tolist()})"
        }

    def step_arena(self):
        self.round_num += 1
        self.generation += 1
        
        agent_proposal = self.ask_ai_policy_action()
        action = agent_proposal["action"]
        
        # Execute official Gymnasium Env Step on bare GPU hardware
        t0 = time.time()
        obs, reward, terminated, truncated, info = self.gym_env.step(action)
        step_duration_ms = (time.time() - t0) * 1000.0
        self.total_reward += reward
        
        if terminated:
            self.gym_env.reset()
            
        unroll_val = self.gym_env.UNROLL_OPTIONS[action[0]]
        cs_val = self.gym_env.CACHE_STREAM_OPTIONS[action[1]]
        ptx_val = self.gym_env.PTX_MODE_OPTIONS[action[2]]
        reg_val = self.gym_env.MAXREGCOUNT_OPTIONS[action[3]]
        
        challenger = {
            "id": f"rl-cand-{self.generation}",
            "name": agent_proposal["name"],
            "strategy": agent_proposal["strategy"],
            "status": info["status"],
            "reason": info["reason"],
            "latency_us": info.get("latency_us", 999.0),
            "compile_time_ms": step_duration_ms,
            "registers": info.get("registers", 32),
            "generation": self.generation,
            "unroll": unroll_val,
            "cache_streaming": cs_val,
            "ptx_fma_mode": ptx_val,
            "reward": float(reward)
        }
        self.last_challenger = challenger
        
        if info["status"] == "ASCENDED":
            old_name = self.champion["name"]
            self.champion = {
                "id": challenger["id"],
                "name": challenger["name"],
                "author": "RL Evolved Policy",
                "latency_us": info["latency_us"],
                "registers": challenger["registers"],
                "unroll": unroll_val,
                "cache_streaming": cs_val,
                "ptx_fma_mode": ptx_val,
                "status": "CROWNED",
                "generation": self.generation,
                "code_snippet": f"// Champion Evolved via Gym Step {self.round_num}\n#define UNROLL {unroll_val}\n#define PTX_MODE {ptx_val}",
                "defeated_count": 0
            }
            self.hall_of_fame.insert(0, {
                "rank": len(self.hall_of_fame) + 1,
                "name": self.champion["name"],
                "author": challenger["strategy"],
                "latency_us": info["latency_us"],
                "speedup_vs_triton": f"{self.gym_env.target_baseline_us / info['latency_us']:.2f}x",
                "generation": self.generation,
                "achieved_at": time.strftime("%H:%M:%S")
            })
            self.event_log.insert(0, f"👑 [GYM STEP {self.round_num}] ASCENSION! Reward: +{reward:.1f} | {challenger['name']} defeated {old_name} ({info['latency_us']:.2f} µs)!")
        elif info["status"] == "DEFEATED":
            self.champion["defeated_count"] += 1
            self.event_log.insert(0, f"⚔️ [GYM STEP {self.round_num}] DEFEATED. Reward: {reward:.1f} | {challenger['name']} ({info['latency_us']:.2f} µs)")
        else:
            self.event_log.insert(0, f"💀 [GYM STEP {self.round_num}] SLAIN. Reward: {reward:.1f} | {challenger['name']} ({info['reason']})")
            
        return {
            "champion": self.champion,
            "challenger": challenger,
            "telemetry": self.get_gpu_telemetry(),
            "hall_of_fame": self.hall_of_fame[:5],
            "round_num": self.round_num,
            "generation": self.generation,
            "total_reward": round(self.total_reward, 1),
            "log": self.event_log[:15],
            "last_grpo_cohort": getattr(self, "last_grpo_cohort", None)
        }

    def step_grpo(self, rlvr_engine):
        self.round_num += 1
        self.generation += 1
        
        grpo_summary = rlvr_engine.run_grpo_round()
        cohort = grpo_summary["results"]
        best = grpo_summary["best_candidate"]
        self.last_grpo_cohort = cohort
        
        # Check if the best candidate beats the reigning champion
        if best["latency_us"] < self.champion["latency_us"]:
            status = "ASCENDED"
            old_name = self.champion["name"]
            speedup = ((self.champion["latency_us"] - best["latency_us"]) / self.champion["latency_us"]) * 100.0
            
            challenger = {
                "id": best["cand_id"],
                "name": best["name"],
                "strategy": f"GRPO Winner (Advantage: {best['grpo_advantage']:+.2f}, Speedup: {best['speedup_vs_baseline']:.2f}x)",
                "status": "ASCENDED",
                "reason": f"Beat Champion by {speedup:.1f}% ({best['latency_us']:.2f} µs vs {self.champion['latency_us']:.2f} µs)",
                "latency_us": best["latency_us"],
                "compile_time_ms": 1200.0,
                "registers": best["registers"],
                "generation": self.generation,
                "unroll": best["unroll"],
                "cache_streaming": best["cache_streaming"],
                "ptx_fma_mode": best["ptx_fma_mode"],
                "reward": best["reward"]
            }
            
            self.champion = {
                "id": challenger["id"],
                "name": challenger["name"],
                "author": "RLVR GRPO Cohort Victor",
                "latency_us": best["latency_us"],
                "registers": best["registers"],
                "unroll": best["unroll"],
                "cache_streaming": best["cache_streaming"],
                "ptx_fma_mode": best["ptx_fma_mode"],
                "status": "CROWNED",
                "generation": self.generation,
                "code_snippet": f"// RLVR GRPO Champ Gen {self.generation}\n#define UNROLL {best['unroll']}\n#define PTX_MODE {best['ptx_fma_mode']}",
                "defeated_count": 0
            }
            
            self.hall_of_fame.insert(0, {
                "rank": len(self.hall_of_fame) + 1,
                "name": self.champion["name"],
                "author": challenger["strategy"],
                "latency_us": best["latency_us"],
                "speedup_vs_triton": f"{self.gym_env.target_baseline_us / best['latency_us']:.2f}x",
                "generation": self.generation,
                "achieved_at": time.strftime("%H:%M:%S")
            })
            self.event_log.insert(0, f"👑 [RLVR GRPO] ASCENSION! {best['name']} crowned! Latency: {best['latency_us']:.2f} µs (Adv: {best['grpo_advantage']:+.2f})")
        else:
            status = "DEFEATED"
            lag = best["latency_us"] - self.champion["latency_us"]
            challenger = {
                "id": best["cand_id"],
                "name": best["name"],
                "strategy": f"GRPO Winner (Advantage: {best['grpo_advantage']:+.2f}, Speedup: {best['speedup_vs_baseline']:.2f}x)",
                "status": "DEFEATED",
                "reason": f"Cohort victor slower than Champion (+{lag:.2f} µs: {best['latency_us']:.2f} µs vs {self.champion['latency_us']:.2f} µs)",
                "latency_us": best["latency_us"],
                "compile_time_ms": 1200.0,
                "registers": best["registers"],
                "generation": self.generation,
                "unroll": best["unroll"],
                "cache_streaming": best["cache_streaming"],
                "ptx_fma_mode": best["ptx_fma_mode"],
                "reward": best["reward"]
            }
            self.champion["defeated_count"] += 1
            self.event_log.insert(0, f"⚔️ [RLVR GRPO] G=4 Cohort completed. Best: {best['name']} ({best['latency_us']:.2f} µs, Adv: {best['grpo_advantage']:+.2f})")
            
        self.last_challenger = challenger
        self.total_reward += best["reward"]
        
        return {
            "champion": self.champion,
            "challenger": challenger,
            "telemetry": self.get_gpu_telemetry(),
            "hall_of_fame": self.hall_of_fame[:5],
            "round_num": self.round_num,
            "generation": self.generation,
            "total_reward": round(self.total_reward, 1),
            "log": self.event_log[:15],
            "last_grpo_cohort": self.last_grpo_cohort
        }
