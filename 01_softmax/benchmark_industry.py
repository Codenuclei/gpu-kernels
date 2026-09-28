import os
import ctypes
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ---------------- Load Compiled Library ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.CDLL(os.path.join(BASE_DIR, "liball_latest.so"))

lib.launch_ptx_fp32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
lib.launch_latest_fp16.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
lib.launch_flash_softmax.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]

# cuDNN loader
CUDNN_SO = "/home/codenuclei/stanford-deviant/assignment1-basics/.venv/lib/python3.13/site-packages/nvidia/cudnn/lib/libcudnn.so.9"
cudnn = ctypes.CDLL(CUDNN_SO)

class CudnnRunner:
    def __init__(self, rows, D):
        self.handle = ctypes.c_void_p()
        self.desc = ctypes.c_void_p()
        cudnn.cudnnCreate(ctypes.byref(self.handle))
        cudnn.cudnnCreateTensorDescriptor(ctypes.byref(self.desc))
        # NCHW float
        cudnn.cudnnSetTensor4dDescriptor(self.desc, 0, 0, rows, D, 1, 1)
        self.alpha = ctypes.c_float(1.0)
        self.beta = ctypes.c_float(0.0)

    def run(self, x, y):
        # CUDNN_SOFTMAX_ACCURATE = 1, CUDNN_SOFTMAX_MODE_INSTANCE = 0
        cudnn.cudnnSoftmaxForward(
            self.handle, 1, 0,
            ctypes.byref(self.alpha), self.desc, ctypes.c_void_p(x.data_ptr()),
            ctypes.byref(self.beta), self.desc, ctypes.c_void_p(y.data_ptr())
        )

    def __del__(self):
        cudnn.cudnnDestroyTensorDescriptor(self.desc)
        cudnn.cudnnDestroy(self.handle)

def ours_ptx_fp32(x):
    y = torch.empty_like(x)
    lib.launch_ptx_fp32(x.data_ptr(), y.data_ptr(), x.shape[0], x.shape[1], torch.cuda.current_stream().cuda_stream)
    return y

def ours_latest_fp16(x):
    y = torch.empty_like(x)
    lib.launch_latest_fp16(x.data_ptr(), y.data_ptr(), x.shape[0], x.shape[1], torch.cuda.current_stream().cuda_stream)
    return y

def ours_flash_softmax(Q, K, V):
    O = torch.empty_like(Q)
    lib.launch_flash_softmax(Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(), Q.shape[0], torch.cuda.current_stream().cuda_stream)
    return O

# Triton Kernels
@triton.autotune(
    configs=[
        triton.Config({'num_warps': 1}, num_stages=2),
        triton.Config({'num_warps': 2}, num_stages=2),
        triton.Config({'num_warps': 4}, num_stages=2),
        triton.Config({'num_warps': 8}, num_stages=2),
    ],
    key=['D'],
)
@triton.jit
def _triton_fwd_kernel(
    y_ptr, y_stride_r,
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
    tl.store(y_ptr + row_idx * y_stride_r + col_offsets, y, mask=mask)

def triton_softmax(x):
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(x.shape[1])
    _triton_fwd_kernel[(x.shape[0],)](
        y, y.stride(0),
        x, x.stride(0),
        x.shape[1],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y

def bench(fn, warmup=25, iters=150):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

def main():
    torch.manual_seed(0)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"=========================================================================================================")
    print(f"                   HEAD-TO-HEAD: OURS vs TRITON vs PYTORCH vs NVIDIA CUDNN 9.19                          ")
    print(f"                                   Platform: {gpu_name}                                                  ")
    print(f"=========================================================================================================\n")

    # ---------------- 1. FP16 Standalone Softmax ----------------
    print(f"--- [PART 1: FP16 Standalone Softmax Across Industry] ---")
    fp16_shapes = [
        (65536, 64),    # Micro head
        (32768, 128),   # Standard attention head (LLaMA/Mistral)
        (8192, 512),    # Medium sequence
        (8192, 2048),   # Large hidden
        (8192, 4096),   # 4K context
    ]
    print(f"{'Shape':<16} | {'Ours (FP16)':<12} {'Triton (FP16)':<14} {'PyTorch (FP16)':<15} | {'Win vs Triton':<14} {'Win vs PyTorch':<14}")
    print(f"{'-'*95}")
    for rows, D in fp16_shapes:
        x = (torch.rand(rows, D, device="cuda", dtype=torch.float16) * 20 - 10)
        t_ours = bench(lambda: ours_latest_fp16(x))
        t_triton = bench(lambda: triton_softmax(x))
        t_torch = bench(lambda: torch.softmax(x, dim=-1))

        win_triton = f"{t_triton/t_ours:.2f}x ({(t_triton/t_ours-1)*100:+.1f}%)"
        win_torch = f"{t_torch/t_ours:.2f}x ({(t_torch/t_ours-1)*100:+.1f}%)"
        print(f"{f'{rows}x{D}':<16} | {t_ours:6.3f} ms    {t_triton:6.3f} ms      {t_torch:6.3f} ms       | {win_triton:<14} {win_torch:<14}")

    # ---------------- 2. FP32 Standalone Softmax ----------------
    print(f"\n--- [PART 2: FP32 Standalone Softmax: Ours (PTX) vs cuDNN vs Triton vs PyTorch] ---")
    fp32_shapes = [
        (8192, 1024),
        (8192, 2048),
        (4096, 4096),
        (4096, 8192),
    ]
    print(f"{'Shape':<16} | {'Ours (PTX)':<11} {'cuDNN 9.19':<11} {'Triton':<11} {'PyTorch':<11} | {'Win vs cuDNN':<13} {'Win vs Triton':<13} {'Win vs Torch':<13}")
    print(f"{'-'*105}")
    for rows, D in fp32_shapes:
        x = (torch.rand(rows, D, device="cuda", dtype=torch.float32) * 20 - 10)
        cudnn_r = CudnnRunner(rows, D)
        y_cudnn = torch.empty_like(x)

        t_ours = bench(lambda: ours_ptx_fp32(x))
        t_cudnn = bench(lambda: cudnn_r.run(x, y_cudnn))
        t_triton = bench(lambda: triton_softmax(x))
        t_torch = bench(lambda: torch.softmax(x, dim=-1))

        win_cudnn = f"{t_cudnn/t_ours:.2f}x"
        win_triton = f"{t_triton/t_ours:.2f}x"
        win_torch = f"{t_torch/t_ours:.2f}x"
        print(f"{f'{rows}x{D}':<16} | {t_ours:6.3f} ms   {t_cudnn:6.3f} ms   {t_triton:6.3f} ms   {t_torch:6.3f} ms   | {win_cudnn:<13} {win_triton:<13} {win_torch:<13}")

    # ---------------- 3. Fused Attention / Flash-Softmax ----------------
    print(f"\n--- [PART 3: The 100x Scale Paradigm: Fused Flash-Softmax vs Unfused DRAM Softmax] ---")
    seq_lengths = [512, 1024, 2048, 4096]
    head_dim = 64
    print(f"{'Seq Length':<16} | {'Ours (Flash-SRAM)':<18} {'PyTorch SDPA':<14} {'Unfused DRAM Softmax':<22} | {'Speedup vs Unfused':<20}")
    print(f"{'-'*105}")
    for seq in seq_lengths:
        Q = torch.randn(seq, head_dim, device="cuda")
        K = torch.randn(seq, head_dim, device="cuda")
        V = torch.randn(seq, head_dim, device="cuda")

        # Unfused: Q @ K.T -> DRAM -> Softmax -> DRAM -> @ V
        def run_unfused():
            S = torch.matmul(Q, K.transpose(0, 1))
            P = torch.softmax(S, dim=-1)
            return torch.matmul(P, V)

        # PyTorch SDPA (FlashAttention native)
        def run_sdpa():
            q4 = Q.unsqueeze(0).unsqueeze(0)
            k4 = K.unsqueeze(0).unsqueeze(0)
            v4 = V.unsqueeze(0).unsqueeze(0)
            return F.scaled_dot_product_attention(q4, k4, v4)

        t_ours = bench(lambda: ours_flash_softmax(Q, K, V), iters=50)
        t_sdpa = bench(run_sdpa, iters=50)
        t_unfused = bench(run_unfused, iters=50)

        speedup = f"{t_unfused / t_ours:.1f}x Faster"
        print(f"{f'N={seq}, d={head_dim}':<16} | {t_ours:6.3f} ms          {t_sdpa:6.3f} ms      {t_unfused:6.3f} ms             | {speedup:<20}")

    print(f"=========================================================================================================")

if __name__ == "__main__":
    main()
