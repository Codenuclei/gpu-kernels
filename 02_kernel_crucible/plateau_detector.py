#!/usr/bin/env python3
"""
plateau_detector.py - Window-Based Plateau Detection with Time-Decaying Tolerance

Monitors candidate trial outputs over a sliding window of 50 trials.
Checks if any candidate is able to approach the target latency / 8.8x speedup goal
within a +/- 5% tolerance mark that slowly decays with time.
When 50 consecutive trials fail to come near the goal within the decaying tolerance,
exploration is declared saturated (plateaued), triggering automated Silicon Gold distillation.
"""

import time
import math
from collections import deque
from typing import Dict, Any, Optional

class PlateauDetector:
    def __init__(
        self,
        window_size: int = 50,
        target_speedup: float = 8.8,
        target_us_goal: float = 8.88,
        base_tolerance: float = 0.05,        # +/- 5% initial score tolerance
        min_tolerance: float = 0.015,        # Decay floor (1.5%)
        decay_half_life_seconds: float = 180.0  # Decays slowly over 3 minutes
    ):
        self.window_size = window_size
        self.target_speedup = target_speedup
        self.target_us_goal = target_us_goal
        self.base_tolerance = base_tolerance
        self.min_tolerance = min_tolerance
        self.decay_half_life = decay_half_life_seconds
        
        self.trials_window = deque(maxlen=window_size)
        self.all_trials_count = 0
        self.last_ascension_time = time.time()
        self.is_plateaued = False
        self.plateau_event_count = 0

    def reset_ascension(self, new_champ_latency_us: Optional[float] = None):
        """Reset plateau state when a champion ascends or workload changes."""
        self.last_ascension_time = time.time()
        self.is_plateaued = False
        if new_champ_latency_us is not None and new_champ_latency_us > 0:
            # Update target goal to match or exceed the latest champion
            self.target_us_goal = min(self.target_us_goal, float(new_champ_latency_us))

    def get_current_tolerance(self) -> float:
        """
        Decays the tolerance margin slowly over time:
        tolerance(t) = max(min_tol, base_tol * (0.5 ** (dt / half_life)))
        """
        dt = max(0.0, time.time() - self.last_ascension_time)
        decay_factor = 0.5 ** (dt / self.decay_half_life)
        return max(self.min_tolerance, self.base_tolerance * decay_factor)

    def is_trial_near_goal(self, latency_us: float, baseline_us: float, tolerance: float) -> bool:
        """
        Check if a trial latency or speedup is within +/- tolerance of the goal.
        Goal: <= target_us_goal * (1 + tol) OR speedup >= target_speedup * (1 - tol)
        """
        if latency_us <= 0 or math.isnan(latency_us) or math.isinf(latency_us):
            return False
            
        speedup = (baseline_us / latency_us) if baseline_us > 0 else 1.0
        
        # Near latency check: within (1 + tol) of target µs goal (e.g. 8.88 µs * 1.05 = 9.32 µs)
        near_latency = latency_us <= (self.target_us_goal * (1.0 + tolerance))
        # Near speedup check: within (1 - tol) of 8.8x target (e.g. 8.8 * 0.95 = 8.36x)
        near_speedup = speedup >= (self.target_speedup * (1.0 - tolerance))
        
        return near_latency or near_speedup

    def register_trial(self, trial_info: Dict[str, Any], baseline_us: float) -> bool:
        """
        Register a single candidate mutant evaluation.
        Returns True if plateau condition is newly triggered.
        """
        self.all_trials_count += 1
        lat = float(trial_info.get("latency_us", 9999.0))
        speedup = (baseline_us / lat) if lat > 0 else 0.0
        
        entry = {
            "trial_id": self.all_trials_count,
            "name": trial_info.get("name", f"mutant-{self.all_trials_count}"),
            "latency_us": lat,
            "speedup": speedup,
            "valid": trial_info.get("status") != "COMPILE_ERROR" and lat < 9000.0,
            "timestamp": time.time()
        }
        self.trials_window.append(entry)

        # Requirement: Must have accumulated at least window_size (50) outputs
        if len(self.trials_window) < self.window_size:
            return False

        curr_tol = self.get_current_tolerance()

        # Count how many of the 50 outputs in the window came near the goal
        near_count = 0
        for t in self.trials_window:
            if t["valid"] and self.is_trial_near_goal(t["latency_us"], baseline_us, curr_tol):
                near_count += 1

        # Plateau condition: Across the 50 trials, 0 outputs were near the goal
        if near_count == 0 and not self.is_plateaued:
            self.is_plateaued = True
            self.plateau_event_count += 1
            return True

        return False

    def get_telemetry(self, baseline_us: float) -> Dict[str, Any]:
        """Provides real-time telemetry for the Web UI and API."""
        curr_tol = self.get_current_tolerance()
        valid_lats = [t["latency_us"] for t in self.trials_window if t["valid"]]
        best_in_window = min(valid_lats) if valid_lats else None
        
        near_count = 0
        for t in self.trials_window:
            if t["valid"] and self.is_trial_near_goal(t["latency_us"], baseline_us, curr_tol):
                near_count += 1

        elapsed = time.time() - self.last_ascension_time

        return {
            "window_size": self.window_size,
            "trials_in_window": len(self.trials_window),
            "all_trials_count": self.all_trials_count,
            "target_us_goal": round(self.target_us_goal, 2),
            "target_speedup_mark": f"{self.target_speedup:.2f}x",
            "base_tolerance_pct": round(self.base_tolerance * 100.0, 1),
            "current_tolerance_pct": round(curr_tol * 100.0, 2),
            "near_goal_count": near_count,
            "best_latency_in_window_us": round(best_in_window, 2) if best_in_window is not None else None,
            "is_plateaued": self.is_plateaued,
            "decay_elapsed_seconds": round(elapsed, 1),
            "plateau_event_count": self.plateau_event_count
        }
