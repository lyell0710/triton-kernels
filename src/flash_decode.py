# SPDX-License-Identifier: MIT
"""Flash-Decoding(decode 阶段 Sq=1 的 attention,split-K 并行)。

解决什么问题:FA2 的 M-tile 在 Sq=1 时全废(EXP-T01（Triton FA2 forward）§7)——grid 只剩
B*Hq 个 program,长上下文时大量 SM 闲置。decode 的并行度必须改从 KV 序列
维取:把 KV 切成 num_splits 段,每段独立算 (m_p, l_p, acc_p) 三个部分
统计量(与 FA 的 online softmax 同一套代数),再用一次归并
  m = max(m_p);  l = Σ l_p·e^{m_p−m};  out = Σ acc_p·e^{m_p−m} / l
合成精确结果——**softmax 的可归并性用了两次**(块内 online、块间归并)。
详见 docs/theory/05_flash_decoding.md。

数据布局(两 kernel 之间的接力面,全 fp32):
    Mp/Lp: (B, Hq, num_splits)      每段的部分 max / 部分行和
    Accp:  (B, Hq, num_splits, D)   每段未归一化输出
    fp32 是硬要求:归并要对部分量做 e^{m_p-m} 换基,fp16 存部分和会把
    split 间的舍入误差放大进最终输出。

为什么两段式而非单 pass 原子(面试点):归并必须先拿到全局 max 才能换基,
这是一次跨段规约依赖——log-sum-exp 不能交换成朴素原子加;两个 kernel 的
边界就是这次全局同步(CUDA 里 grid 级同步的最便宜写法)。

性能特征(EXP-T04（Flash-Decoding）,4090):32K 上下文 kernel 级 2.39× vs naive
(0.147 vs 0.352 ms,单轮口径);GQA 原生读 KV(不物化 repeat)是另一半
收益;llm-engine 接入后 fp32 probe PASS(5.96e-5)。
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
    pid_s = tl.program_id(0)              # 第几个 KV 分段(split-K 并行轴)
    pid_bh = tl.program_id(1)
    b = pid_bh // NUM_Q_HEADS
    hq = pid_bh % NUM_Q_HEADS
    hkv = hq // GQA_GROUP                 # GQA 原生:索引换算直读共享 kv head

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + b * stride_qb + hq * stride_qh + offs_d * stride_qd)
    # decode 是带宽瓶颈(整条 KV 只读一次,算术强度 O(1)),tensor core
    # 帮不上忙且 tl.dot 要求 M≥16 得 pad——全程 fp32 SIMT 反而干净
    q = q.to(tl.float32)

    lo = pid_s * split_size
    hi = tl.minimum(lo + split_size, n_ctx)   # 末段截断到真实上下文长度

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    offs_n = tl.arange(0, BLOCK_N)

    for start in range(lo, hi, BLOCK_N):
        curr = start + offs_n
        kmask = curr < hi                 # 上界用 hi 而非 n_ctx:段界越界与
        k = tl.load(K + b * stride_kb + hkv * stride_kh   # 序列越界一并兜住
                    + curr[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                    mask=kmask[:, None], other=0.0).to(tl.float32)
        v = tl.load(V + b * stride_vb + hkv * stride_vh
                    + curr[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                    mask=kmask[:, None], other=0.0).to(tl.float32)
        # Sq=1:qk^T 退化为广播乘 + 行内规约,(BLOCK_N,) 个分数
        s = tl.sum(k * q[None, :], axis=1) * sm_scale       # (BLOCK_N,)
        s = tl.where(kmask, s, float("-inf"))   # 越界列 exp=0,不污染 l/acc
        # 与 FA2 同一套 m/l/alpha online 更新,标量版(行数=1)
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    # 只写部分统计量、不归一化——除法必须推迟到 combine,否则块间代数不可结合
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
    # 每个 (b,h) 一个 program 串行收全部段:NUM_SPLITS 很小(≤ 数百),
    # 一层树规约或原子都比不过单 program 直读——也顺便保证求和顺序确定
    pid_bh = tl.program_id(0)
    b = pid_bh // NUM_Q_HEADS
    h = pid_bh % NUM_Q_HEADS
    offs_s = tl.arange(0, NUM_SPLITS)     # NUM_SPLITS 须为 2 的幂(tl.arange
    offs_d = tl.arange(0, HEAD_DIM)       # 编译期约束,launcher 已 next_pow2)

    m = tl.load(Mp + b * stride_mb + h * stride_mh + offs_s * stride_ms)
    l = tl.load(Lp + b * stride_mb + h * stride_mh + offs_s * stride_ms)
    acc = tl.load(Accp + b * stride_ab + h * stride_ah
                  + offs_s[:, None] * stride_as + offs_d[None, :] * stride_ad)

    # 面试点:归并代数。段 p 的不变量是 l_p = Σ_{i∈p} e^{s_i-m_p},
    # acc_p = Σ_{i∈p} e^{s_i-m_p}·v_i;取 m_g = max_p m_p,每段乘换基因子
    # e^{m_p-m_g} 后:Σ_p l_p·e^{m_p-m_g} = Σ_i e^{s_i-m_g}(全局行和),
    # Σ_p acc_p·e^{m_p-m_g} = Σ_i e^{s_i-m_g}·v_i——与单 pass 结果**精确
    # 相等**,不是近似
    m_g = tl.max(m, axis=0)
    scale = tl.exp(m - m_g)                       # 空段 m=-inf → scale=0:
    l_g = tl.sum(l * scale, axis=0)               # 贡献自动湮灭,无需特判;
    out = tl.sum(acc * scale[:, None], axis=0) / l_g  # -inf 初值在此承担语义
    tl.store(O + b * stride_ob + h * stride_oh + offs_d * stride_od,
             out.to(O.dtype.element_ty))


def flash_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 sm_scale: float | None = None,
                 num_splits: int | None = None,
                 block_n: int = 128) -> torch.Tensor:
    """q: (B, Hq, 1, D); k/v: (B, Hkv, Skv, D)。decode 专用(无 causal mask:
    最后一个 token 可看全部 KV,mask 退化为空)。"""
    B, Hq, sq, D = q.shape
    Hkv, Skv = k.shape[1], k.shape[2]
    assert sq == 1 and Hq % Hkv == 0
    if sm_scale is None:
        sm_scale = D ** -0.5
    if num_splits is None:
        # 填满 SM 的启发式:4090 有 128 个 SM,目标 B*Hq*splits ≳ 2×128
        # (每 SM 至少 2 个 CTA 才有延迟切换余地);上限 cdiv(Skv, block_n)
        # 保证段长不小于一个 BLOCK_N(再细切只剩空转)。未扫参(EXP-T04 §7)
        want = max(1, (2 * 128) // max(B * Hq, 1))
        num_splits = min(max(want, 1), triton.cdiv(Skv, block_n))
    # combine 里 NUM_SPLITS 喂 tl.arange,必须 2 的幂;向上取整可能造出
    # 空段,由 combine 的 scale=0 兜底
    num_splits = triton.next_power_of_2(num_splits)
    split_size = triton.cdiv(Skv, num_splits)

    # 中间量全 fp32(见文件头:换基精度要求),额外显存仅 O(B·Hq·splits·D)
    mp = torch.empty(B, Hq, num_splits, device=q.device, dtype=torch.float32)
    lp = torch.empty_like(mp)
    accp = torch.empty(B, Hq, num_splits, D, device=q.device,
                       dtype=torch.float32)
    o = torch.empty(B, Hq, 1, D, device=q.device, dtype=q.dtype)

    # (B,Hq,1,D) 压成 (B,Hq,D):kernel 签名只吃 3 维 stride
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
