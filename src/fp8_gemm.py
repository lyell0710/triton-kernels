# SPDX-License-Identifier: MIT
"""FP8 GEMM(per-block scaling),对标 DeepGEMM 的缩放策略,sm_89 实现。

缩放布局(DeepSeek/DeepGEMM 风格):
- 权重 B(K,N):128×128 块各持一个 scale(块内 absmax/448,e4m3 满量程)
- 激活 A(M,K):每行每 128-K 组一个 scale(per-token-group)
GEMM 主循环 BLOCK_K=128 与缩放组对齐:每组做一次 fp8 tl.dot(fp32 累加),
乘上 sa(行向量)×sb(该 K 组该 N 块的标量)后并入总累加——
**反量化融合在累加里,不物化 fp16 权重**。

与 DeepGEMM 的界线(面试点,详见 docs/theory/06):DeepGEMM 本体是
Hopper-only(wgmma 异步矩阵指令 + TMA 搬运);sm_89(Ada)只有同步 mma 与
cp.async,本实现即"mma 路线"——同一缩放代数,不同指令世代。
"""

import torch
import triton
import triton.language as tl

FP8_MAX = 448.0                    # e4m3 最大正规值
GROUP = 128


@torch.no_grad()
def quant_fp8_block(w: torch.Tensor):
    """(K,N) fp16/bf16 → fp8 e4m3 + scale (K/128, N/128) fp32。"""
    K, N = w.shape
    assert K % GROUP == 0 and N % GROUP == 0
    wf = w.float().reshape(K // GROUP, GROUP, N // GROUP, GROUP)
    amax = wf.abs().amax(dim=(1, 3)).clamp(min=1e-8)       # (K/g, N/g)
    scale = amax / FP8_MAX
    q = (wf / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX)
    return q.reshape(K, N).to(torch.float8_e4m3fn), scale


@torch.no_grad()
def quant_fp8_token_group(a: torch.Tensor):
    """(M,K) → fp8 + scale (M, K/128)。"""
    M, K = a.shape
    assert K % GROUP == 0
    af = a.float().reshape(M, K // GROUP, GROUP)
    amax = af.abs().amax(dim=2).clamp(min=1e-8)
    scale = amax / FP8_MAX
    q = (af / scale[:, :, None]).clamp(-FP8_MAX, FP8_MAX)
    return q.reshape(M, K).to(torch.float8_e4m3fn), scale


@triton.jit
def _fp8_gemm_kernel(A, B, C, SA, SB,
                     M, N, K,
                     stride_am, stride_ak, stride_bk, stride_bn,
                     stride_cm, stride_cn,
                     stride_sam, stride_sak, stride_sbk, stride_sbn,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     GROUP_M: tl.constexpr):
    BLOCK_K: tl.constexpr = 128            # 与缩放组硬对齐
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_in_group = GROUP_M * num_pid_n
    gid = pid // num_in_group
    first_m = gid * GROUP_M
    gsz = tl.minimum(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % num_in_group) % gsz
    pid_n = (pid % num_in_group) // gsz

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kg in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0.0)
        part = tl.dot(a, b)                                  # fp8→fp32
        sa = tl.load(SA + offs_m * stride_sam + kg * stride_sak,
                     mask=offs_m < M, other=0.0)             # (BLOCK_M,)
        sb = tl.load(SB + kg * stride_sbk
                     + (pid_n * BLOCK_N // 128) * stride_sbn)  # 标量
        acc += part * sa[:, None] * sb
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(C.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def fp8_gemm_prequant(a_fp8, sa, b_fp8, sb, out_dtype=torch.float16,
                      block_m=128, group_m=8, num_warps=8, num_stages=3):
    """量化好的输入直接 GEMM(bench 用,不含量化成本)。BLOCK_N 固定 128
    与权重块对齐(sb 取标量的前提)。"""
    M, K = a_fp8.shape
    N = b_fp8.shape[1]
    c = torch.empty(M, N, device=a_fp8.device, dtype=out_dtype)
    BLOCK_N = 128
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, BLOCK_N),)
    _fp8_gemm_kernel[grid](a_fp8, b_fp8, c, sa, sb, M, N, K,
                           a_fp8.stride(0), a_fp8.stride(1),
                           b_fp8.stride(0), b_fp8.stride(1),
                           c.stride(0), c.stride(1),
                           sa.stride(0), sa.stride(1),
                           sb.stride(0), sb.stride(1),
                           BLOCK_M=block_m, BLOCK_N=BLOCK_N, GROUP_M=group_m,
                           num_warps=num_warps, num_stages=num_stages)
    return c


def fp8_gemm(a: torch.Tensor, b: torch.Tensor, **cfg):
    """端到端:fp16/bf16 输入,内部量化(在线 A 量化是真实 serving 形态)。"""
    a_fp8, sa = quant_fp8_token_group(a)
    b_fp8, sb = quant_fp8_block(b)
    return fp8_gemm_prequant(a_fp8, sa, b_fp8, sb, out_dtype=a.dtype, **cfg)
