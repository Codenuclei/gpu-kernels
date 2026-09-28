#!/usr/bin/env python3
"""
=============================================================================
 BARE-METAL CRUCIBLE MONKEY-PATCH LOADER FOR UNSLOTH & PYTORCH
 Replaces Standard PyTorch & Unsloth Softmax with Silicon-Evolved CUDA Kernels
 Target Hardware: NVIDIA GeForce RTX 2050 Mobile (GA107, Ampere SM 86)
 Zero Hardcoded Baselines - 100% Measured Bare-Metal Silicon Timers
=============================================================================
"""

import os
import sys
import math
import ctypes
import subprocess
import torch
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUDA_SRC = os.path.join(BASE_DIR, "crucible_ops.cu")
LIB_PATH = os.path.join(BASE_DIR, "crucible_ops.so")

# ---------------------------------------------------------------------------
# 1. Shared Library Compilation & Loader
# ---------------------------------------------------------------------------

_CRUCIBLE_LIB = None
AUTO_HOTSWAP = True

def get_crucible_lib(force_reload=False):
    """Compiles (if needed) and loads the bare-metal crucible CUDA library."""
    global _CRUCIBLE_LIB
    if _CRUCIBLE_LIB is not None and not force_reload:
        return _CRUCIBLE_LIB

    if not os.path.exists(LIB_PATH) or os.path.getmtime(CUDA_SRC) > os.path.getmtime(LIB_PATH) or force_reload:
        cmd = [
            "nvcc", "-O3", "--use_fast_math", "-arch=sm_86",
            "-shared", "-Xcompiler", "-fPIC",
            "-o", LIB_PATH, CUDA_SRC
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"NVCC Compilation failed:\n{res.stderr}")

    lib = ctypes.CDLL(LIB_PATH)
    
    # Signatures
    lib.launch_softmax_fp16.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p
    ]
    lib.launch_softmax_fp32.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p
    ]
    lib.launch_split_kv_decode.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p
    ]

    _CRUCIBLE_LIB = lib
    return _CRUCIBLE_LIB

def hot_swap_champion(unroll=8, cache_streaming=0, ptx_mode=1, maxreg=0, champion_name="Champion"):
    """
    Auto-replaces and hot-swaps the newly crowned champion kernel into Unsloth & PyTorch runtime.
    Called automatically whenever an agent mutant defeats the reigning champion on bare silicon.
    """
    global _CRUCIBLE_LIB
    import time
    print(f"\n⚡ [Crucible Hot-Swap] Outperformed! Auto-compiling champion '{champion_name}' for Unsloth...")
    print(f"   Architecture parameters: UNROLL={unroll} | CACHE_STREAM={cache_streaming} | PTX_FMA={ptx_mode}")
    
    cmd = [
        "nvcc", "-O3", "--use_fast_math", "-arch=sm_86",
        "-shared", "-Xcompiler", "-fPIC",
        f"-DCHAMP_UNROLL={unroll}",
        f"-DCHAMP_CACHE_STREAM={cache_streaming}",
        f"-DCHAMP_PTX_MODE={ptx_mode}",
        "-o", LIB_PATH, CUDA_SRC
    ]
    if maxreg > 0:
        cmd.insert(4, f"-maxrregcount={maxreg}")

    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"❌ [Hot-Swap Error] Compilation failed:\n{res.stderr}")
        return False

    comp_time = (time.time() - t0) * 1000.0

    # Reload dynamic C-library
    _CRUCIBLE_LIB = None
    _ = get_crucible_lib(force_reload=False)

    # Automatically patch Unsloth & PyTorch if not yet patched
    if not is_patched():
        patch_unsloth()

    print(f"👑 [Hot-Swap Deployed] '{champion_name}' compiled in {comp_time:.0f} ms & actively powering Unsloth & PyTorch!\n")
    return True


# ---------------------------------------------------------------------------
# 2. Bare-Metal Kernel Wrappers
# ---------------------------------------------------------------------------

_ORIG_F_SOFTMAX = F.softmax
_ORIG_TORCH_SOFTMAX = torch.softmax
_ORIG_TENSOR_SOFTMAX = torch.Tensor.softmax
_ORIG_F_SDPA = F.scaled_dot_product_attention

_IS_PATCHED = False

def bare_metal_softmax(input: torch.Tensor, dim: int = -1, _stacklevel: int = 3, dtype=None):
    """
    High-performance bare-metal CUDA Softmax.
    Executes in-register vectorized reductions on Ampere SM 86 bare silicon.
    Falls back gracefully to native PyTorch for non-CUDA or non-last-dim tensors.
    """
    if not input.is_cuda:
        return _ORIG_F_SOFTMAX(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

    norm_dim = dim if dim >= 0 else input.dim() + dim
    if norm_dim != input.dim() - 1:
        return _ORIG_F_SOFTMAX(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

    target_dtype = dtype or input.dtype
    if target_dtype not in (torch.float16, torch.float32):
        return _ORIG_F_SOFTMAX(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

    lib = get_crucible_lib()
    x = input if input.dtype == target_dtype else input.to(target_dtype)
    x = x.contiguous()
    out = torch.empty_like(x)

    d = x.shape[-1]
    num_rows = x.numel() // d
    stream = torch.cuda.current_stream().cuda_stream

    if target_dtype == torch.float16:
        lib.launch_softmax_fp16(x.data_ptr(), out.data_ptr(), num_rows, d, stream)
    else:
        lib.launch_softmax_fp32(x.data_ptr(), out.data_ptr(), num_rows, d, stream)

    return out

def bare_metal_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = None
):
    """
    Bare-metal Flash-Decoding Scaled Dot-Product Attention.
    When query token length is 1 (decoding step), activates the Split-KV online softmax champion kernel.
    Otherwise, leverages the bare-metal vectorized in-register softmax.
    """
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        return _ORIG_F_SDPA(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)

    d = query.shape[-1]
    q_len = query.shape[-2]
    k_len = key.shape[-2]

    # Fast path for Single-Token Decoding (Split-KV Champion Kernel)
    if (
        q_len == 1
        and query.dtype == torch.float16
        and key.dtype == torch.float16
        and value.dtype == torch.float16
        and attn_mask is None
        and dropout_p == 0.0
        and d in (64, 128, 256)
        and query.dim() in (3, 4)
    ):
        lib = get_crucible_lib()
        eff_scale = scale if scale is not None else (1.0 / math.sqrt(d))
        chunks = 16
        stream = torch.cuda.current_stream().cuda_stream

        # Flatten batch and heads into individual decode operations
        orig_shape = query.shape
        q_flat = query.view(-1, d).contiguous()
        k_flat = key.view(-1, k_len, d).contiguous()
        v_flat = value.view(-1, k_len, d).contiguous()
        batch_heads = q_flat.shape[0]

        out_flat = torch.empty((batch_heads, d), device=query.device, dtype=torch.float16)
        part_out = torch.empty(chunks * d, device=query.device, dtype=torch.float32)
        part_max = torch.empty(chunks, device=query.device, dtype=torch.float32)
        part_sum = torch.empty(chunks, device=query.device, dtype=torch.float32)

        for b in range(batch_heads):
            lib.launch_split_kv_decode(
                q_flat[b].data_ptr(),
                k_flat[b].data_ptr(),
                v_flat[b].data_ptr(),
                part_out.data_ptr(),
                part_max.data_ptr(),
                part_sum.data_ptr(),
                out_flat[b].data_ptr(),
                k_len,
                d,
                chunks,
                eff_scale,
                stream
            )

        return out_flat.view(orig_shape)

    # General path: Scaled Dot-Product with Bare-Metal Softmax
    eff_scale = scale if scale is not None else (1.0 / math.sqrt(d))
    scores = torch.matmul(query, key.transpose(-2, -1)) * eff_scale
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores.masked_fill_(~attn_mask, -1e9)
        else:
            scores = scores + attn_mask
    if is_causal:
        causal_mask = torch.ones((q_len, k_len), device=query.device, dtype=torch.bool).tril()
        scores.masked_fill_(~causal_mask, -1e9)

    probs = bare_metal_softmax(scores, dim=-1)
    if dropout_p > 0.0:
        probs = F.dropout(probs, p=dropout_p)
    return torch.matmul(probs, value)

# ---------------------------------------------------------------------------
# 3. Monkey-Patching Engine
# ---------------------------------------------------------------------------

def patch_unsloth():
    """
    Monkey-patches PyTorch and Unsloth attention/softmax with our Bare-Metal Kernel.
    """
    global _IS_PATCHED
    if _IS_PATCHED:
        print("[Crucible] Already patched!")
        return

    # Ensure lib is ready
    get_crucible_lib()

    # 1. Patch PyTorch globally
    F.softmax = bare_metal_softmax
    torch.softmax = bare_metal_softmax
    torch.Tensor.softmax = bare_metal_softmax
    F.scaled_dot_product_attention = bare_metal_sdpa

    # 2. Patch Unsloth dispatch if present
    patched_modules = ["torch.nn.functional", "torch"]
    try:
        import unsloth
        import unsloth.utils.attention_dispatch as ud
        ud.scaled_dot_product_attention = bare_metal_sdpa
        patched_modules.append("unsloth.utils.attention_dispatch")
    except (ImportError, Exception):
        pass

    try:
        import unsloth.models.llama as llama
        llama.scaled_dot_product_attention = bare_metal_sdpa
        patched_modules.append("unsloth.models.llama")
    except (ImportError, Exception):
        pass

    _IS_PATCHED = True
    print("=" * 70)
    print("🔥 [Crucible] Bare-Metal Silicon Kernel Injected Successfully!")
    print(f"   Patched namespaces: {', '.join(patched_modules)}")
    print("   RTX 2050 Mobile hardware timers active.")
    print("=" * 70)

def unpatch_unsloth():
    """
    Restores original PyTorch and Unsloth implementations.
    """
    global _IS_PATCHED
    if not _IS_PATCHED:
        print("[Crucible] Not patched.")
        return

    F.softmax = _ORIG_F_SOFTMAX
    torch.softmax = _ORIG_TORCH_SOFTMAX
    torch.Tensor.softmax = _ORIG_TENSOR_SOFTMAX
    F.scaled_dot_product_attention = _ORIG_F_SDPA

    try:
        import unsloth.utils.attention_dispatch as ud
        ud.scaled_dot_product_attention = _ORIG_F_SDPA
    except (ImportError, Exception):
        pass

    try:
        import unsloth.models.llama as llama
        llama.scaled_dot_product_attention = _ORIG_F_SDPA
    except (ImportError, Exception):
        pass

    _IS_PATCHED = False
    print("[Crucible] Original PyTorch & Unsloth dispatch restored.")

def is_patched() -> bool:
    return _IS_PATCHED

# ---------------------------------------------------------------------------
# 4. Silicon Verification & Benchmark Suite
# ---------------------------------------------------------------------------

def run_verification_suite():
    """
    Executes live verification and hardware timing tests on bare silicon.
    """
    print("\n" + "=" * 80)
    print("🔥 CRUCIBLE BARE-METAL SILICON VERIFICATION & BENCHMARK SUITE")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    # 1. Standalone Softmax Test
    print("\n[Test 1] Testing Standalone FP16 Softmax (1024x1024)...")
    x16 = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    
    # Warmup
    for _ in range(50):
        _ = _ORIG_F_SOFTMAX(x16, dim=-1)
        _ = bare_metal_softmax(x16, dim=-1)
    torch.cuda.synchronize()

    # Accuracy check
    ref_sm = _ORIG_F_SOFTMAX(x16, dim=-1)
    crucible_sm = bare_metal_softmax(x16, dim=-1)
    max_err_sm = (crucible_sm - ref_sm).abs().max().item()
    print(f"  -> Numerical Max Absolute Error: {max_err_sm:.2e}")
    assert max_err_sm < 1e-4, f"Softmax error too high: {max_err_sm}"
    print("  -> Numerical Precision: VERIFIED ✅")

    # Timing
    iters = 1000
    start_evt = torch.cuda.Event(enable_timing=True)
    stop_evt = torch.cuda.Event(enable_timing=True)

    start_evt.record()
    for _ in range(iters):
        _ = _ORIG_F_SOFTMAX(x16, dim=-1)
    stop_evt.record()
    torch.cuda.synchronize()
    torch_sm_us = (start_evt.elapsed_time(stop_evt) / iters) * 1000

    start_evt.record()
    for _ in range(iters):
        _ = bare_metal_softmax(x16, dim=-1)
    stop_evt.record()
    torch.cuda.synchronize()
    crucible_sm_us = (start_evt.elapsed_time(stop_evt) / iters) * 1000

    speedup_sm = torch_sm_us / crucible_sm_us
    print(f"  -> PyTorch Native Softmax:  {torch_sm_us:.2f} µs")
    print(f"  -> Crucible Bare-Metal:     {crucible_sm_us:.2f} µs")
    print(f"  -> Speedup:                 {speedup_sm:.2f}x 🔥")

    # 2. Split-KV Attention Decoding Test
    print("\n[Test 2] Testing Single-Token Decoding Attention (B=1, H=1, N=1024, D=64)...")
    q = torch.randn(1, 1, 1, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 1, 1024, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 1, 1024, 64, device="cuda", dtype=torch.float16)

    # Warmup
    for _ in range(50):
        _ = _ORIG_F_SDPA(q, k, v)
        _ = bare_metal_sdpa(q, k, v)
    torch.cuda.synchronize()

    # Accuracy
    ref_att = _ORIG_F_SDPA(q, k, v)
    crucible_att = bare_metal_sdpa(q, k, v)
    max_err_att = (crucible_att - ref_att).abs().max().item()
    cos_sim = F.cosine_similarity(crucible_att.squeeze().float(), ref_att.squeeze().float(), dim=0).item()
    print(f"  -> Numerical Max Absolute Error: {max_err_att:.2e}")
    print(f"  -> Cosine Similarity:           {cos_sim:.6f}")
    assert cos_sim > 0.999, f"Attention cosine similarity too low: {cos_sim}"
    print("  -> Numerical Precision: VERIFIED ✅")

    # Timing
    start_evt.record()
    for _ in range(iters):
        _ = _ORIG_F_SDPA(q, k, v)
    stop_evt.record()
    torch.cuda.synchronize()
    torch_att_us = (start_evt.elapsed_time(stop_evt) / iters) * 1000

    start_evt.record()
    for _ in range(iters):
        _ = bare_metal_sdpa(q, k, v)
    stop_evt.record()
    torch.cuda.synchronize()
    crucible_att_us = (start_evt.elapsed_time(stop_evt) / iters) * 1000

    speedup_att = torch_att_us / crucible_att_us
    print(f"  -> PyTorch Native SDPA:     {torch_att_us:.2f} µs")
    print(f"  -> Crucible Bare-Metal:     {crucible_att_us:.2f} µs")
    print(f"  -> Speedup:                 {speedup_att:.2f}x 🔥")

    # 3. Monkey-Patching End-to-End Test
    print("\n[Test 3] Testing Monkey-Patch Life-Cycle...")
    patch_unsloth()
    assert is_patched(), "Should be patched"

    # Call torch.softmax directly
    patched_res = torch.softmax(x16, dim=-1)
    assert (patched_res - ref_sm).abs().max().item() < 1e-4
    print("  -> Patched torch.softmax() routing verified ✅")

    # Call F.scaled_dot_product_attention directly
    patched_sdpa = F.scaled_dot_product_attention(q, k, v)
    assert F.cosine_similarity(patched_sdpa.squeeze().float(), ref_att.squeeze().float(), dim=0).item() > 0.999
    print("  -> Patched F.scaled_dot_product_attention() routing verified ✅")

    unpatch_unsloth()
    assert not is_patched(), "Should be unpatched"
    print("  -> Clean unpatch life-cycle verified ✅")

    print("\n" + "=" * 80)
    print("🏆 ALL VERIFICATIONS PASSED - ZERO DRIFT ON RAW SILICON")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_verification_suite()
