# Challenge 04: Attention Kernels (MHA, FlashAttention, Flash-Decoding)

Production-grade Transformer attention implementations in CUDA C++, solving the quadratic memory wall ($O(N^2)$) and autoregressive decoding underutilization.

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d}}\right) \mathbf{V}$$

---

## Kernels Implemented

### 1. Naive Multi-Head Attention (`mha_naive.cu`)
- 3 distinct kernel passes:
  1. $S = Q K^T / \sqrt{d}$
  2. $P = \text{softmax}(S)$
  3. $O = P V$
- **The Failure**: Materializes $N \times N$ attention matrix in slow global DRAM ($O(N^2)$ footprint).
- Measured: $2.138\text{ ms}$ for $N=1024, d=64$.

### 2. Fused FlashAttention Forward Kernel (`flash_attention.cu`)
- Full implementation of Tri Dao's FlashAttention algorithm from scratch.
- Tiles $Q$ into $B_r \times d$ and $K, V$ into $B_c \times d$ in fast SRAM (shared memory).
- Tracks online softmax running max $m$ and normalizer $l$ per row.
- Rescales running output accumulator:
  $$O^{(new)} = O^{(old)} \cdot e^{m^{(old)} - m^{(new)}} + P_{ij} V_j$$
- **Zero Global Memory Materialization**: Footprint shrinks from $O(N^2)$ ($4\text{ MB}$) to $O(N)$ ($256\text{ KB}$).
- Causal autoregressive masking support.
- Measured: **0.738 ms** ($2.9\times$ faster than naive MHA).

### 3. Flash-Decoding / Split-KV (`flash_decoding.cu`)
- Solves the single-token generation bottleneck ($q=1$, large $N$).
- **The Problem**: In standard decoding, batch=1 / q=1 can only launch 1 thread block, leaving 15 out of 16 SMs idle.
- **The Solution**: Partitions the KV cache across 16 parallel splits:
  - **Stage 1 Kernel**: Computes partial attention across splits, emitting partial outputs, partial maxes, and partial sums.
  - **Stage 2 Kernel**: Log-sum-exp reduction combines partial split outputs into final output vector.
- Measured: **36.2 microseconds** for $N=4096$ context on RTX 2050!
