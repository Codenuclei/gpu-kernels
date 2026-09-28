#include <cstdio>
#include <cuda_runtime.h>

// Inline PTX for 16-byte cache-streaming load (ld.global.cs.v4.u32)
__device__ __forceinline__ float4 ptx_load_cs(const float4* addr) {
    float4 ret;
    asm volatile(
        "ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(*reinterpret_cast<unsigned*>(&ret.x)),
          "=r"(*reinterpret_cast<unsigned*>(&ret.y)),
          "=r"(*reinterpret_cast<unsigned*>(&ret.z)),
          "=r"(*reinterpret_cast<unsigned*>(&ret.w))
        : "l"(addr)
    );
    return ret;
}

// Inline PTX for 16-byte cache-streaming store (st.global.cs.v4.u32)
__device__ __forceinline__ void ptx_store_cs(float4* addr, float4 val) {
    asm volatile(
        "st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
        :: "l"(addr),
           "r"(*reinterpret_cast<unsigned*>(&val.x)),
           "r"(*reinterpret_cast<unsigned*>(&val.y)),
           "r"(*reinterpret_cast<unsigned*>(&val.z)),
           "r"(*reinterpret_cast<unsigned*>(&val.w))
        : "memory"
    );
}

// Inline PTX for SFU base-2 exponentiation with flush-to-zero (ex2.approx.ftz.f32)
// exp(x) = ex2(x * log2(e))
__device__ __forceinline__ float ptx_exp(float x) {
    constexpr float LOG2E = 1.4426950408889634f;
    float t = x * LOG2E;
    float res;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(res) : "f"(t));
    return res;
}

// Inline PTX for butterfly warp shuffle (shfl.sync.bfly.b32)
__device__ __forceinline__ float ptx_shfl_xor(float val, int mask) {
    float res;
    asm volatile(
        "shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
        : "=f"(res)
        : "f"(val), "r"(mask)
    );
    return res;
}

__global__ void test_ptx_k(const float4* in, float4* out) {
    float4 v = ptx_load_cs(in);
    v.x = ptx_exp(v.x);
    v.y = ptx_exp(v.y);
    v.z = ptx_exp(v.z);
    v.w = ptx_exp(v.w);
    v.x = ptx_shfl_xor(v.x, 16);
    pt_store:
    ptx_store_cs(out, v);
}

int main() {
    printf("PTX compilation succeeded!\n");
    return 0;
}
