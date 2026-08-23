# SPDX-License-Identifier: MIT
"""Flash Attention 2 forward(简化版,Triton)。causal + GQA,fp16 输入 fp32 累加。

为什么快(一句话):不物化 S×S 的注意力矩阵——K/V 分块流过片上,
softmax 用 online 规约(m/l 两个跑动统计量)边算边修正,HBM 读写量从
O(S²) 降到 O(S·D)。逐步推导见 docs/theory/01_flashattention.md。

简化范围(相对官方 FA2,如实声明):只有 forward;无 dropout/alibi/paged;
seq 维边界用 mask 处理;每个 program 处理一个 (batch, q_head, M块)。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fa2_fwd_kernel(
    Q, K, V, O,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    n_ctx,
    NUM_Q_HEADS: tl.constexpr,
    GQA_GROUP: tl.constexpr,          # q_heads // kv_heads
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)          # 第几个 Q 行块
    pid_bh = tl.program_id(1)         # batch*q_head 扁平索引
    b = pid_bh // NUM_Q_HEADS
    hq = pid_bh % NUM_Q_HEADS
    hkv = hq // GQA_GROUP             # GQA:多个 q head 共享一个 kv head

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = (Q + b * stride_qb + hq * stride_qh
              + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < n_ctx, other=0.0)

    # online softmax 的三个跑动量:行最大值 m、行和 l、输出累加 acc(全 fp32)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # causal 时本行块最多看到自己所在对角块;末块含对角,用逐元素 mask
    hi = tl.minimum((pid_m + 1) * BLOCK_M, n_ctx) if IS_CAUSAL else n_ctx

    for start_n in range(0, hi, BLOCK_N):
        curr_n = start_n + offs_n
        k_ptrs = (K + b * stride_kb + hkv * stride_kh
                  + curr_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
        v_ptrs = (V + b * stride_vb + hkv * stride_vh
                  + curr_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        k = tl.load(k_ptrs, mask=curr_n[:, None] < n_ctx, other=0.0)
        v = tl.load(v_ptrs, mask=curr_n[:, None] < n_ctx, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale          # (M, N) fp32
        qk = tl.where(curr_n[None, :] < n_ctx, qk, float("-inf"))
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= curr_n[None, :], qk, float("-inf"))

        # online 更新:新块并入后,旧的 exp 统一乘 alpha 校正到新基准 m_new
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]

    o_ptrs = (O + b * stride_ob + hq * stride_oh
              + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < n_ctx)


def fa2_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                causal: bool = True, sm_scale: float | None = None,
                block_m: int = 128, block_n: int = 64,
                num_warps: int = 8, num_stages: int = 2) -> torch.Tensor:
    """q: (B, Hq, S, D); k/v: (B, Hkv, S, D),Hq 必须是 Hkv 的整数倍。"""
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    assert Hq % Hkv == 0 and D in (64, 128) and q.is_cuda
    if sm_scale is None:
        sm_scale = D ** -0.5
    o = torch.empty_like(q)
    grid = (triton.cdiv(S, block_m), B * Hq)
    _fa2_fwd_kernel[grid](
        q, k, v, o, sm_scale,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        S,
        NUM_Q_HEADS=Hq, GQA_GROUP=Hq // Hkv, HEAD_DIM=D,
        BLOCK_M=block_m, BLOCK_N=block_n, IS_CAUSAL=causal,
        num_warps=num_warps, num_stages=num_stages,
    )
    return o
