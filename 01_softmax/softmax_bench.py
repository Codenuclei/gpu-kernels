# Fused softmax fwd + bwd: Custom CUDA vs Triton vs PyTorch
# Features:
#   - Custom CUDA: float4 loads, register-resident, online softmax (m, s), warp XOR shuffle, L=m+log(s) recompute in bwd
#   - Triton: Autotuned fused softmax forward & backward
#   - PyTorch: torch.softmax and torch._softmax_backward_data

import os
import subprocess
import ctypes
import torch
import triton
import triton.language as tl

# ---------------- Compile / Load CUDA Shared Library ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SO_PATH = os.path.join(BASE_DIR, "libfused_softmax.so")
CU_PATH = os.path.join(BASE_DIR, "fused_softmax_lib.cu")

def ensure_cuda_lib():
    if not os.path.exists(SO_PATH):
        print(f"Compiling {CU_PATH} -> {SO_PATH}...")
        cmd = [
            "nvcc", "-O3", "-arch=sm_86", "--shared", "-Xcompiler", "-fPIC",
            CU_PATH, "-o", SO_PATH
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"CUDA compilation failed:\n{res.stderr}")
    
    lib = ctypes.CDLL(SO_PATH)
    lib.softmax_fwd_cuda.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p
    ]
    lib.softmax_bwd_cuda.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p
    ]
    return lib

cuda_lib = ensure_cuda_lib()

def cuda_softmax_fwd(x):
    rows, D = x.shape
    y = torch.empty_like(x)
    L = torch.empty(rows, dtype=x.dtype, device=x.device)
    stream = torch.cuda.current_stream().cuda_stream
    cuda_lib.softmax_fwd_cuda(x.data_ptr(), y.data_ptr(), L.data_ptr(), rows, D, stream)
    return y, L

def cuda_softmax_bwd(x, dy, L):
    rows, D = x.shape
    dx = torch.empty_like(x)
    stream = torch.cuda.current_stream().cuda_stream
    cuda_lib.softmax_bwd_cuda(x.data_ptr(), dy.data_ptr(), L.data_ptr(), dx.data_ptr(), rows, D, stream)
    return dx

# ---------------- Triton Implementation ----------------
@triton.autotune(
    configs=[
        triton.Config({'num_warps': 4}, num_stages=2),
        triton.Config({'num_warps': 4}, num_stages=3),
        triton.Config({'num_warps': 8}, num_stages=2),
        triton.Config({'num_warps': 8}, num_stages=3),
        triton.Config({'num_warps': 16}, num_stages=2),
    ],
    key=['D'],
)
@triton.jit
def _triton_fwd_kernel(
    y_ptr, y_stride_r,
    l_ptr,
    x_ptr, x_stride_r,
    D,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < D
    x = tl.load(x_ptr + row_idx * x_stride_r + col_offsets, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(y_ptr + row_idx * y_stride_r + col_offsets, y, mask=mask)
    if l_ptr is not None:
        tl.store(l_ptr + row_idx, m + tl.log(s))

@triton.autotune(
    configs=[
        triton.Config({'num_warps': 4}, num_stages=2),
        triton.Config({'num_warps': 4}, num_stages=3),
        triton.Config({'num_warps': 8}, num_stages=2),
        triton.Config({'num_warps': 8}, num_stages=3),
        triton.Config({'num_warps': 16}, num_stages=2),
    ],
    key=['D'],
)
@triton.jit
def _triton_bwd_kernel(
    dx_ptr, dx_stride_r,
    dy_ptr, dy_stride_r,
    x_ptr, x_stride_r,
    l_ptr,
    D,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < D
    x = tl.load(x_ptr + row_idx * x_stride_r + col_offsets, mask=mask, other=-float('inf'))
    dy = tl.load(dy_ptr + row_idx * dy_stride_r + col_offsets, mask=mask, other=0.0)
    L = tl.load(l_ptr + row_idx)
    y = tl.exp(x - L)
    dot = tl.sum(dy * y, axis=0)
    dx = y * (dy - dot)
    tl.store(dx_ptr + row_idx * dx_stride_r + col_offsets, dx, mask=mask)

def triton_softmax_fwd(x):
    rows, D = x.shape
    y = torch.empty_like(x)
    L = torch.empty(rows, dtype=x.dtype, device=x.device)
    BLOCK_SIZE = triton.next_power_of_2(D)
    _triton_fwd_kernel[(rows,)](
        y, y.stride(0),
        L,
        x, x.stride(0),
        D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y, L

def triton_softmax_bwd(x, dy, L):
    rows, D = x.shape
    dx = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(D)
    _triton_bwd_kernel[(rows,)](
        dx, dx.stride(0),
        dy, dy.stride(0),
        x, x.stride(0),
        L,
        D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return dx

# ---------------- Benchmark Harness ----------------
def bench(fn, warmup=30, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms

def run_benchmarks():
    torch.manual_seed(0)
    dev = "cuda"
    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")
    print(f"{'='*105}")
    
    shapes = [
        (8192, 1024),
        (16384, 1024),
        (4096, 2048),
        (8192, 2048),
        (4096, 4096),
        (8192, 4096),
        (2048, 8192),
        (4096, 8192),
    ]

    results = []

    for rows, D in shapes:
        x = (torch.rand(rows, D, device=dev) * 20 - 10)
        dy = torch.randn(rows, D, device=dev)

        # Correctness check
        y_cuda, L_cuda = cuda_softmax_fwd(x)
        dx_cuda = cuda_softmax_bwd(x, dy, L_cuda)

        y_triton, L_triton = triton_softmax_fwd(x)
        dx_triton = triton_softmax_bwd(x, dy, L_triton)

        xr = x.clone().requires_grad_(True)
        yr = torch.softmax(xr, dim=-1)
        yr.backward(dy)

        fwd_err_cuda = (y_cuda - yr).abs().max().item()
        bwd_err_cuda = (dx_cuda - xr.grad).abs().max().item()
        fwd_err_triton = (y_triton - yr).abs().max().item()
        bwd_err_triton = (dx_triton - xr.grad).abs().max().item()

        # Benchmarks - FWD
        t_fwd_cuda = bench(lambda: cuda_softmax_fwd(x))
        t_fwd_triton = bench(lambda: triton_softmax_fwd(x))
        t_fwd_torch = bench(lambda: torch.softmax(x, dim=-1))

        # Benchmarks - BWD
        yt = torch.softmax(x, dim=-1)
        t_bwd_cuda = bench(lambda: cuda_softmax_bwd(x, dy, L_cuda))
        t_bwd_triton = bench(lambda: triton_softmax_bwd(x, dy, L_triton))
        t_bwd_torch = bench(lambda: torch._softmax_backward_data(dy, yt, -1, yt.dtype))

        # Bandwidth calculations
        # Fwd: 1 read x, 1 write y -> 2 * N * 4 bytes
        bytes_fwd = 2 * rows * D * 4
        # Bwd: read x, read dy, write dx -> 3 * N * 4 bytes
        bytes_bwd = 3 * rows * D * 4

        bw_fwd_cuda = bytes_fwd / (t_fwd_cuda * 1e-3) / 1e9
        bw_fwd_triton = bytes_fwd / (t_fwd_triton * 1e-3) / 1e9
        bw_fwd_torch = bytes_fwd / (t_fwd_torch * 1e-3) / 1e9

        bw_bwd_cuda = bytes_bwd / (t_bwd_cuda * 1e-3) / 1e9
        bw_bwd_triton = bytes_bwd / (t_bwd_triton * 1e-3) / 1e9
        bw_bwd_torch = bytes_bwd / (t_bwd_torch * 1e-3) / 1e9

        win_fwd_vs_triton = (t_fwd_triton / t_fwd_cuda - 1.0) * 100
        win_fwd_vs_torch = (t_fwd_torch / t_fwd_cuda - 1.0) * 100
        win_bwd_vs_triton = (t_bwd_triton / t_bwd_cuda - 1.0) * 100
        win_bwd_vs_torch = (t_bwd_torch / t_bwd_cuda - 1.0) * 100

        print(f"\n[{rows} x {D}]")
        print(f"  Accuracy (vs PyTorch):")
        print(f"    CUDA   - fwd err: {fwd_err_cuda:.2e}, bwd err: {bwd_err_cuda:.2e}")
        print(f"    Triton - fwd err: {fwd_err_triton:.2e}, bwd err: {bwd_err_triton:.2e}")
        print(f"  FORWARD:")
        print(f"    CUDA   : {t_fwd_cuda:.3f} ms  ({bw_fwd_cuda:6.1f} GB/s)")
        print(f"    Triton : {t_fwd_triton:.3f} ms  ({bw_fwd_triton:6.1f} GB/s)  --> CUDA is {t_fwd_triton/t_fwd_cuda:.2f}x ({win_fwd_vs_triton:+.1f}%)")
        print(f"    PyTorch: {t_fwd_torch:.3f} ms  ({bw_fwd_torch:6.1f} GB/s)  --> CUDA is {t_fwd_torch/t_fwd_cuda:.2f}x ({win_fwd_vs_torch:+.1f}%)")
        print(f"  BACKWARD:")
        print(f"    CUDA   : {t_bwd_cuda:.3f} ms  ({bw_bwd_cuda:6.1f} GB/s)")
        print(f"    Triton : {t_bwd_triton:.3f} ms  ({bw_bwd_triton:6.1f} GB/s)  --> CUDA is {t_bwd_triton/t_bwd_cuda:.2f}x ({win_bwd_vs_triton:+.1f}%)")
        print(f"    PyTorch: {t_bwd_torch:.3f} ms  ({bw_bwd_torch:6.1f} GB/s)  --> CUDA is {t_bwd_torch/t_bwd_cuda:.2f}x ({win_bwd_vs_torch:+.1f}%)")

        results.append({
            "shape": (rows, D),
            "t_fwd_cuda": t_fwd_cuda,
            "t_fwd_triton": t_fwd_triton,
            "t_fwd_torch": t_fwd_torch,
            "bw_fwd_cuda": bw_fwd_cuda,
            "bw_fwd_triton": bw_fwd_triton,
            "bw_fwd_torch": bw_fwd_torch,
            "win_fwd_triton": win_fwd_vs_triton,
            "win_fwd_torch": win_fwd_vs_torch,
            "t_bwd_cuda": t_bwd_cuda,
            "t_bwd_triton": t_bwd_triton,
            "t_bwd_torch": t_bwd_torch,
            "bw_bwd_cuda": bw_bwd_cuda,
            "bw_bwd_triton": bw_bwd_triton,
            "bw_bwd_torch": bw_bwd_torch,
            "win_bwd_triton": win_bwd_vs_triton,
            "win_bwd_torch": win_bwd_vs_torch,
        })

    # Summary Table
    print(f"\n{'='*105}")
    print(f"{'BENCHMARK SUMMARY: CUDA vs TRITON vs PYTORCH':^105}")
    print(f"{'='*105}")
    print(f"{'Shape':<16} | {'FWD CUDA':<9} {'FWD Triton':<10} {'FWD Win vs Tr':<13} | {'BWD CUDA':<9} {'BWD Triton':<10} {'BWD Win vs Tr':<13}")
    print(f"{'-'*105}")
    for r in results:
        shape_str = f"{r['shape'][0]}x{r['shape'][1]}"
        fwd_speedup = f"{r['t_fwd_triton']/r['t_fwd_cuda']:.2f}x ({r['win_fwd_triton']:+.1f}%)"
        bwd_speedup = f"{r['t_bwd_triton']/r['t_bwd_cuda']:.2f}x ({r['win_bwd_triton']:+.1f}%)"
        print(f"{shape_str:<16} | {r['t_fwd_cuda']:6.3f} ms {r['t_fwd_triton']:6.3f} ms {fwd_speedup:<13} | {r['t_bwd_cuda']:6.3f} ms {r['t_bwd_triton']:6.3f} ms {bwd_speedup:<13}")
    print(f"{'='*105}")

if __name__ == "__main__":
    run_benchmarks()
