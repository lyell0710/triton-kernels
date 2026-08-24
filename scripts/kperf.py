# SPDX-License-Identifier: MIT
"""kperf:无性能计数器环境(RmProfilingAdminOnly=1)的 kernel 性能观测工具。

NCU 四大件的平替(方法论见 docs/theory/04_perf_without_ncu.md):
1. 计时      → CUDA event / wall(本文件 bench)      替 Duration
2. roofline  → 手算 bytes/flops ÷ 实测 → %峰值        替 Speed-of-Light(SOL)
3. occupancy → Triton 编译产物 n_regs/shared + 公式    替 Occupancy section
4. 归因      → 变体对照(控制变量)+ nsys 时间线        替 stall 分解(近似)

用法: python scripts/kperf.py   (对本仓三个 kernel 各出一张观测卡)
"""

import sys
import time
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PEAK_BW_GBS = 1008.0          # RTX 4090 GDDR6X
PEAK_FP16_TFLOPS = 165.2      # 4090 fp16 tensor core(non-sparsity)
SM_COUNT = 128
REGS_PER_SM = 65536
SMEM_PER_SM = 102400          # 100KB(Ada 每 SM 可配 shared)
MAX_THREADS_PER_SM = 1536     # Ada(sm_89)


def bench(fn, iters=200, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def triton_occupancy(jit_fn, num_warps: int):
    """从 Triton 编译缓存读 n_regs/shared,按 Ada 资源上限算理论 occupancy。
    kernel 必须已被调用过一次(JIT 缓存命中)。"""
    kernels = list(jit_fn.device_caches.values()) if hasattr(jit_fn, "device_caches") else []
    compiled = None
    for dc in kernels:
        cache = dc[0] if isinstance(dc, tuple) else dc
        if cache:
            compiled = next(iter(cache.values()))
            break
    if compiled is None:
        return None
    n_regs = getattr(compiled, "n_regs", None)
    n_spills = getattr(compiled, "n_spills", 0)
    smem = getattr(compiled.metadata, "shared", 0)
    threads = num_warps * 32
    lim_regs = REGS_PER_SM // max(n_regs * threads, 1) if n_regs else 99
    lim_smem = SMEM_PER_SM // smem if smem else 99
    lim_thr = MAX_THREADS_PER_SM // threads
    blocks = max(1, min(lim_regs, lim_smem, lim_thr))
    occ = blocks * threads / MAX_THREADS_PER_SM
    limiter = min((lim_regs, "regs"), (lim_smem, "smem"), (lim_thr, "threads"))[1]
    return {"n_regs": n_regs, "n_spills": n_spills, "smem_B": smem,
            "blocks_per_sm": blocks, "occupancy": round(occ, 2),
            "limiter": limiter}


def card(name, ms, bytes_moved, flops, occ):
    bw = bytes_moved / (ms / 1e3) / 1e9
    tf = flops / (ms / 1e3) / 1e12
    bound = "memory-bound" if bw / PEAK_BW_GBS > tf / PEAK_FP16_TFLOPS \
        else "compute-bound"
    print(f"\n== {name}")
    print(f"   时间 {ms:.4f} ms | 带宽 {bw:.0f} GB/s({bw/PEAK_BW_GBS*100:.0f}% 峰值)"
          f" | 算力 {tf:.1f} TFLOPS({tf/PEAK_FP16_TFLOPS*100:.0f}% 峰值)"
          f" → {bound}")
    if occ:
        print(f"   occupancy {occ['occupancy']*100:.0f}%"
              f"(每 SM {occ['blocks_per_sm']} block,限制因子={occ['limiter']};"
              f" regs/thread={occ['n_regs']}, spills={occ['n_spills']},"
              f" smem={occ['smem_B']//1024}KB)")


def main():
    torch.manual_seed(0)
    from elementwise_kernels import softmax, _softmax_kernel
    from fa2_fwd import fa2_forward, _fa2_fwd_kernel
    from gemm_pipelined import gemm, _gemm_kernel

    # softmax 8192²(带宽主导)
    x = torch.randn(8192, 8192, device="cuda", dtype=torch.float32)
    ms = bench(lambda: softmax(x))
    card("softmax 8192² fp32", ms, 2 * x.numel() * 4, 5 * x.numel(),
         triton_occupancy(_softmax_kernel, num_warps=8))

    # FA2 4K(算力主导)
    B, Hq, Hkv, S, D = 1, 32, 8, 4096, 128
    q = torch.randn(B, Hq, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.float16)
    ms = bench(lambda: fa2_forward(q, k, v, causal=True), iters=50)
    flops = 4.0 * B * Hq * S * S * D / 2
    bytes_m = (q.numel() + k.numel() + v.numel() + q.numel()) * 2
    card("FA2 fwd S=4096 fp16", ms, bytes_m, flops,
         triton_occupancy(_fa2_fwd_kernel, num_warps=8))

    # GEMM 4096³(算力主导)
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    ms = bench(lambda: gemm(a, b), iters=50)
    card("GEMM 4096³ fp16 (stages=3)", ms,
         3 * 4096 * 4096 * 2, 2 * 4096 ** 3,
         triton_occupancy(_gemm_kernel, num_warps=8))


if __name__ == "__main__":
    main()
