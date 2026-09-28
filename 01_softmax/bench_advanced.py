import os
import ctypes
import torch
import triton
import triton.language as tl

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SO_PATH = os.path.join(BASE_DIR, "libfused_softmax_advanced.so")

lib = ctypes.CDLL(SO_PATH)
lib.softmax_fwd_fp16_launch.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p
]

def cuda_softmax_fwd_fp16(x):
    rows, D = x.shape
    y = torch.empty_like(x)
    L = torch.empty(rows, dtype=torch.float32, device=x.device)
    stream = torch.cuda.current_stream().cuda_stream
    lib.softmax_fwd_fp16_launch(x.data_ptr(), y.data_ptr(), L.data_ptr(), rows, D, stream)
    return y, L

# Triton FP16 Autotuned Kernel
@triton.autotune(
    configs=[
        triton.Config({'num_warps': 1}, num_stages=2),
        triton.Config({'num_warps': 2}, num_stages=2),
        triton.Config({'num_warps': 4}, num_stages=2),
        triton.Config({'num_warps': 8}, num_stages=2),
        triton.Config({'num_warps': 16}, num_stages=2),
    ],
    key=['D'],
)
@triton.jit
def _triton_fwd_fp16(
    y_ptr, y_stride_r,
    l_ptr,
    x_ptr, x_stride_r,
    D,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < D
    x = tl.load(x_ptr + row_idx * x_stride_r + col_offsets, mask=mask, other=-float('inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(y_ptr + row_idx * y_stride_r + col_offsets, y.to(tl.float16), mask=mask)
    if l_ptr is not None:
        tl.store(l_ptr + row_idx, m + tl.log(s))

def triton_softmax_fwd_fp16(x):
    rows, D = x.shape
    y = torch.empty_like(x)
    L = torch.empty(rows, dtype=torch.float32, device=x.device)
    BLOCK_SIZE = triton.next_power_of_2(D)
    _triton_fwd_fp16[(rows,)](
        y, y.stride(0),
        L,
        x, x.stride(0),
        D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y, L

def bench(fn, warmup=25, iters=150):
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
    return start.elapsed_time(end) / iters

def main():
    torch.manual_seed(0)
    device_name = torch.cuda.get_device_name(0)
    print(f"FP16 Benchmarks on {device_name}")
    print("=" * 105)

    shapes = [
        (65536, 64),    # Micro head dimension
        (32768, 128),   # LLaMA / Mistral attention head
        (16384, 256),   # Wide head
        (8192, 512),    # Sequence softmax
        (8192, 1024),   # Model hidden dimension
        (8192, 2048),   # Large hidden dimension
        (8192, 4096),   # 4K hidden / seq
        (4096, 8192),   # 8K context
    ]

    print(f"{'Shape':<16} | {'Strategy':<13} | {'CUDA FP16':<10} {'Triton FP16':<12} {'PyTorch FP16':<13} | {'Win vs Triton':<13} {'Win vs Torch':<13}")
    print("-" * 105)

    for rows, D in shapes:
        x = (torch.rand(rows, D, device="cuda", dtype=torch.float16) * 20 - 10)

        # Correctness check
        y_cuda, _ = cuda_softmax_fwd_fp16(x)
        y_triton, _ = triton_softmax_fwd_fp16(x)
        y_torch = torch.softmax(x, dim=-1)

        err_cuda = (y_cuda.float() - y_torch.float()).abs().max().item()
        err_triton = (y_triton.float() - y_torch.float()).abs().max().item()

        t_cuda = bench(lambda: cuda_softmax_fwd_fp16(x))
        t_triton = bench(lambda: triton_softmax_fwd_fp16(x))
        t_torch = bench(lambda: torch.softmax(x, dim=-1))

        strategy = "Warp/Row" if D <= 1024 else "Block/Row"
        win_triton = f"{t_triton/t_cuda:.2f}x ({(t_triton/t_cuda-1)*100:+.1f}%)"
        win_torch = f"{t_torch/t_cuda:.2f}x ({(t_torch/t_cuda-1)*100:+.1f}%)"
        shape_str = f"{rows}x{D}"

        print(f"{shape_str:<16} | {strategy:<13} | {t_cuda:6.3f} ms   {t_triton:6.3f} ms     {t_torch:6.3f} ms     | {win_triton:<13} {win_torch:<13}")

    print("=" * 105)

if __name__ == "__main__":
    main()
