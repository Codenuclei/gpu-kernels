# Challenge 02: Parallel Reduction & Prefix Scan

Exhaustive implementations of collective reduction patterns and work-efficient prefix scan algorithms optimized for NVIDIA Ampere architecture.

---

## 1. Parallel Reduction Hierarchy (`reduce.cu`)

Adapted from Mark Harris' seminal optimization guide and modernized for SM 8.6 warp shuffle instructions:

1. **Stage 1: Interleaved Addressing with Divergent Branching**
   - Uses `if (tid % (2 * s) == 0)` causing extreme warp branch divergence.
   - Measured: $1.970\text{ ms}$, $34.1\text{ GB/s}$.
2. **Stage 2: Contiguous Threads**
   - Resolves warp divergence via `index = 2 * s * tid`, but suffers 32-way shared memory bank conflicts.
   - Measured: $1.191\text{ ms}$, $56.4\text{ GB/s}$ ($1.65\times$).
3. **Stage 3: Sequential Addressing**
   - Replaces strided addressing with stride halving (`s = blockDim.x / 2; s > 0; s >>= 1`), fully eliminating shared memory bank conflicts.
   - Measured: $1.146\text{ ms}$, $58.6\text{ GB/s}$ ($1.72\times$).
4. **Stage 4: First Add During Global Memory Load**
   - Halves the number of thread blocks and cuts global memory load latency in half by performing the first addition during global read.
   - Measured: $0.781\text{ ms}$, $85.9\text{ GB/s}$ ($2.52\times$).
5. **Stage 5: Warp Synchronous Unrolling**
   - Avoids `__syncthreads()` overhead when `s <= 32` since threads within a warp execute in SIMT lockstep.
6. **Stage 6: Template-Based Loop Unrolling**
   - Complete compile-time loop unrolling using template parameter `blockSize`.
7. **Stage 7: Hardware Warp Shuffle (`__shfl_down_sync`)**
   - Completely eliminates shared memory for intra-warp reduction by communicating directly via register shuffles.
   - Measured: $0.781\text{ ms}$, $86.0\text{ GB/s}$ ($2.52\times$ over naive).

---

## 2. Parallel Prefix Scan (`prefix_scan.cu`)

1. **Hillis-Steele Algorithm**:
   - Step-efficient parallel scan ($O(N \log N)$ total additions).
   - Fast for small arrays, requires double buffering in shared memory.
2. **Blelloch Work-Efficient Tree Scan**:
   - $O(N)$ total operations.
   - Phase 1 (Up-Sweep): builds balanced reduction tree.
   - Phase 2 (Down-Sweep): propagates running prefix sums back down the tree.
   - Includes shared memory padding to eliminate bank conflicts.
3. **Warp Shuffle Scan (`__shfl_up_sync`)**:
   - Ultra-fast intra-warp prefix scan in pure registers ($0.0020\text{ ms}$ for 512 elements).
