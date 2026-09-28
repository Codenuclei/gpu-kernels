import os
import sys
import time
import json
import numpy as np
from rlvr_verifier import VerifiableRewardOracle
from kernel_gym_env import BASE_KERNEL_TEMPLATE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class RLVREngine:
    """
    RLVR + GRPO (Group Relative Policy Optimization) Engine for GPU Kernel Search.
    Samples a cohort of G=4 candidate kernels, verifies them with the deterministic
    RLVR Oracle, and optimizes the policy using group relative advantages.
    """
    def __init__(self, group_size=4, baseline_latency_us=None):
        self.oracle = VerifiableRewardOracle(baseline_latency_us=baseline_latency_us)
        self.group_size = group_size
        self.generation = 0
        
        # Policy probability logits over [unroll (4), cache_stream (2), ptx (2)]
        self.unroll_logits = np.array([0.0, 1.0, 0.5, 0.2], dtype=np.float32) # unrolls: [1, 2, 4, 8]
        self.cs_logits = np.array([0.5, 0.5], dtype=np.float32) # cs: [0, 1]
        self.ptx_logits = np.array([0.2, 1.2], dtype=np.float32) # ptx: [0, 1]
        
        self.history = []

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def sample_cohort(self):
        """Samples a group of G candidates using current policy logits."""
        cohort = []
        p_unroll = self._softmax(self.unroll_logits)
        p_cs = self._softmax(self.cs_logits)
        p_ptx = self._softmax(self.ptx_logits)
        
        unroll_options = [1, 2, 4, 8]
        cs_options = [0, 1]
        ptx_options = [0, 1]
        
        for i in range(self.group_size):
            u_idx = np.random.choice(len(unroll_options), p=p_unroll)
            c_idx = np.random.choice(len(cs_options), p=p_cs)
            p_idx = np.random.choice(len(ptx_options), p=p_ptx)
            
            cohort.append({
                "cand_id": f"g{self.generation}_c{i+1}",
                "name": f"GRPO-Mutant-{i+1}",
                "unroll_idx": u_idx,
                "cs_idx": c_idx,
                "ptx_idx": p_idx,
                "unroll": unroll_options[u_idx],
                "cache_streaming": cs_options[c_idx],
                "ptx_fma_mode": ptx_options[p_idx],
            })
        return cohort

    def run_grpo_round(self, ai_cohort=None):
        self.generation += 1
        unroll_options = [1, 2, 4, 8]
        cs_options = [0, 1]
        ptx_options = [0, 1]

        if ai_cohort and len(ai_cohort) == self.group_size:
            cohort = []
            for i, p in enumerate(ai_cohort):
                act = p["action"]
                u_idx, c_idx, pt_idx = act[0] % 4, act[1] % 2, act[2] % 2
                cohort.append({
                    "cand_id": f"g{self.generation}_c{i+1}",
                    "name": p.get("name", f"GRPO-AI-{i+1}"),
                    "strategy": p.get("strategy", "Concurrent LLM Slot"),
                    "unroll_idx": u_idx,
                    "cs_idx": c_idx,
                    "ptx_idx": pt_idx,
                    "unroll": unroll_options[u_idx],
                    "cache_streaming": cs_options[c_idx],
                    "ptx_fma_mode": ptx_options[pt_idx],
                })
        else:
            cohort = self.sample_cohort()

        results = []
        
        # Parallel evaluation across CPU threads & GPU streams (Saturate 4 workers)
        from concurrent.futures import ThreadPoolExecutor

        def eval_candidate(cand):
            code = BASE_KERNEL_TEMPLATE
            code = code.replace("__UNROLL__", str(cand["unroll"]))
            code = code.replace("__CACHE_STREAM__", str(cand["cache_streaming"]))
            code = code.replace("__PTX_MODE__", str(cand["ptx_fma_mode"]))
            ver_res = self.oracle.verify_and_score(code, cand["cand_id"])
            return {**cand, **ver_res}

        with ThreadPoolExecutor(max_workers=self.group_size) as executor:
            results = list(executor.map(eval_candidate, cohort))

        # ---------------- COMPUTE GRPO ADVANTAGES ----------------
        rewards = np.array([r["reward"] for r in results], dtype=np.float32)
        mean_r = np.mean(rewards)
        std_r = np.std(rewards) + 1e-6
        advantages = (rewards - mean_r) / std_r
        
        for i in range(len(results)):
            results[i]["grpo_advantage"] = float(advantages[i])

        # ---------------- POLICY GRADIENT UPDATE ----------------
        # Update logits proportionally to advantages (REINFORCE / GRPO style)
        lr = 0.15
        for i, res in enumerate(results):
            adv = advantages[i]
            # Gradient step on sampled indices
            self.unroll_logits[res["unroll_idx"]] += lr * adv
            self.cs_logits[res["cs_idx"]] += lr * adv
            self.ptx_logits[res["ptx_idx"]] += lr * adv

        # Clip logits to prevent explosion
        self.unroll_logits = np.clip(self.unroll_logits, -3.0, 3.0)
        self.cs_logits = np.clip(self.cs_logits, -3.0, 3.0)
        self.ptx_logits = np.clip(self.ptx_logits, -3.0, 3.0)

        # Find best candidate of the group
        best_cand = max(results, key=lambda x: x["reward"])
        
        summary = {
            "generation": self.generation,
            "cohort_size": self.group_size,
            "results": results,
            "best_candidate": best_cand,
            "mean_reward": float(mean_r),
            "std_reward": float(std_r),
            "updated_logits": {
                "unroll_p": [float(p) for p in self._softmax(self.unroll_logits)],
                "cs_p": [float(p) for p in self._softmax(self.cs_logits)],
                "ptx_p": [float(p) for p in self._softmax(self.ptx_logits)],
            }
        }
        self.history.append(summary)
        return summary
