#!/usr/bin/env python3
"""
Unified benchmark runner for gpu-kernels.
Compiles and executes the full GPU kernel suite and outputs performance metrics.
"""

import subprocess
import os
import sys

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"[*] Building and running benchmark suite in {root}")

    # Build bench
    cmd = ["make", "bench"]
    res = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    if res.returncode != 0:
        print("[!] Build failed:")
        print(res.stderr)
        sys.exit(1)

    # Run bench binary
    bench_bin = os.path.join(root, "bench", "benchmark_suite")
    run_res = subprocess.run([bench_bin], cwd=root, capture_output=True, text=True)
    print(run_res.stdout)
    if run_res.stderr:
        print(run_res.stderr, file=sys.stderr)

    if run_res.returncode != 0:
        sys.exit(run_res.returncode)

if __name__ == "__main__":
    main()
