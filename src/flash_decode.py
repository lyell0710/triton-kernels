# SPDX-License-Identifier: MIT
"""Flash-Decoding(decode 阶段 Sq=1 的 attention,split-K 并行)。

FA2 的 M-tile 在 Sq=1 时全废(EXP-T01 §7)——decode 的并行度必须来自
KV 序列维:把 KV 切成 num_splits 段,每段独立算 (m_p, l_p, acc_p) 三个
部分统计量(与 FA 的 online softmax 同一套代数),再用一次归并
  m = max(m_p);  l = Σ l_p·e^{m_p−m};  out = Σ acc_p·e^{m_p−m} / l
合成精确结果——**softmax 的可归并性用了两次**(块内 online、块间归并)。
详见 docs/theory/05_flash_decoding.md。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fd_partial_kernel(
    Q, K, V, Mp, Lp, Accp,
    sm_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_mb, stride_mh, stride_ms,
    stride_ab, stride_ah, stride_as, stride_ad,
    n_ctx, split_size,
    NUM_Q_HEADS: tl.constexpr, GQA_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_s = tl.program_id(0)              # 第几个 KV 分段
    pid_bh = tl.program_id(1)
    b = pid_bh // NUM_Q_HEADS
    hq = pid_bh % NUM_Q_HEADS
    hkv = hq // GQA_GROUP

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + b * stride_qb + hq * stride_qh + offs_d * stride_qd)
    q = q.to(tl.float32)

    lo = pid_s * split_size
    hi = tl.minimum(lo + split_size, n_ctx)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    offs_n = tl.arange(0, BLOCK_N)

    for start in range(lo, hi, BLOCK_N):
        curr = start + offs_n
        kmask = curr < hi
        k = tl.load(K + b * stride_kb + hkv * stride_kh
                    + curr[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                    mask=kmask[:, None], other=0.0).to(tl.float32)
        v = tl.load(V + b * stride_vb + hkv * stride_vh
                    + curr[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                    mask=kmask[:, None], other=0.0).to(tl.float32)
        s = tl.sum(k * q[None, :], axis=1) * sm_scale       # (BLOCK_N,)
        s = tl.where(kmask, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    base = b * stride_mb + hq * stride_mh + pid_s * stride_ms
    tl.store(Mp + base, m_i)
    tl.store(Lp + base, l_i)
    tl.store(Accp + b * stride_ab + hq * stride_ah + pid_s * stride_as
             + offs_d * stride_ad, acc)


@triton.jit
def _fd_combine_kernel(
    Mp, Lp, Accp, O,
    stride_mb, stride_mh, stride_ms,
    stride_ab, stride_ah, stride_as, stride_ad,
    stride_ob, stride_oh, stride_od,
    NUM_Q_HEADS: tl.constexpr, NUM_SPLITS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    b = pid_bh // NUM_Q_HEADS
    h = pid_bh % NUM_Q_HEADS
    offs_s = tl.arange(0, NUM_SPLITS)
    offs_d = tl.arange(0, HEAD_DIM)

    m = tl.load(Mp + b * stride_mb + h * stride_mh + offs_s * stride_ms)
    l = tl.load(Lp + b * stride_mb + h * stride_mh + offs_s * stride_ms)
    acc = tl.load(Accp + b * stride_ab + h * stride_ah
                  + offs_s[:, None] * stride_as + offs_d[None, :] * stride_ad)

    m_g = tl.max(m, axis=0)
    scale = tl.exp(m - m_g)                       # 空段 m=-inf → scale=0
    l_g = tl.sum(l * scale, axis=0)
    out = tl.sum(acc * scale[:, None], axis=0) / l_g
    tl.store(O + b * stride_ob + h * stride_oh + offs_d * stride_od,
             out.to(O.dtype.element_ty))


def flash_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 sm_scale: float | None = None,
                 num_splits: int | None = None,
                 block_n: int = 128) -> torch.Tensor:
    """q: (B, Hq, 1, D); k/v: (B, Hkv, Skv, D)。decode 专用(无 causal mask:
    最后一个 token 可看全部 KV)。"""
    B, Hq, sq, D = q.shape
    Hkv, Skv = k.shape[1], k.shape[2]
    assert sq == 1 and Hq % Hkv == 0
    if sm_scale is None:
        sm_scale = D ** -0.5
    if num_splits is None:
        # 填满 SM:B*Hq*splits ≳ 2×128;段长别小于 block_n
        want = max(1, (2 * 128) // max(B * Hq, 1))
        num_splits = min(max(want, 1), triton.cdiv(Skv, block_n))
    num_splits = triton.next_power_of_2(num_splits)
    split_size = triton.cdiv(Skv, num_splits)

    mp = torch.empty(B, Hq, num_splits, device=q.device, dtype=torch.float32)
    lp = torch.empty_like(mp)
    accp = torch.empty(B, Hq, num_splits, D, device=q.device,
                       dtype=torch.float32)
    o = torch.empty(B, Hq, 1, D, device=q.device, dtype=q.dtype)

    q2 = q.reshape(B, Hq, D).contiguous()
    _fd_partial_kernel[(num_splits, B * Hq)](
        q2, k, v, mp, lp, accp, sm_scale,
        *q2.stride(), *k.stride(), *v.stride(),
        *mp.stride(), *accp.stride(),
        Skv, split_size,
        NUM_Q_HEADS=Hq, GQA_GROUP=Hq // Hkv,
        HEAD_DIM=D, BLOCK_N=block_n, num_warps=4)
    o2 = o.reshape(B, Hq, D)
    _fd_combine_kernel[(B * Hq,)](
        mp, lp, accp, o2,
        *mp.stride(), *accp.stride(), *o2.stride(),
        NUM_Q_HEADS=Hq, NUM_SPLITS=num_splits, HEAD_DIM=D, num_warps=4)
    return o
