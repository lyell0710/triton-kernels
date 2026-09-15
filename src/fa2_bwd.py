# SPDX-License-Identifier: MIT
"""Flash Attention 2 backward(简化版,Triton)。causal + GQA,fp16 输入 fp32 累加。

解决什么问题:forward 靠"不物化 S×S 的 P 矩阵"省了 HBM;backward 若照搬教科书
就要存/读 P —— 那就把 forward 省的又要回来。本 kernel 用 **重算(recomputation)**
换存储:forward 只存行统计量 LSE = m + log(l)(O(S) 而非 O(S²)),backward 在需要
P 的时候用 Q·Kᵀ 现场重算。这是 FlashAttention 论文里 recomputation 的来源——
不是"优化技巧",而是算法设计的一部分。

算法一句话(三行公式,白板必背):
    Δ_i    = Σ_d O[i,d]·dO[i,d]                每行的"输出已收敛度"(preprocess 一次算完)
    dS_ij  = P_ij · (dP_ij − Δ_i),  dP = dO·Vᵀ  softmax 的雅可比:减去行内加权均值
    dV = Pᵀ·dO      dQ = dS·K      dK = dSᵀ·Q
其中 P_ij = exp(s_ij − LSE_i) 是**已归一化**的注意力权重(用 LSE 直接得,省一次除)。

并行布局(grid 两维,与 forward 对称):
    dK/dV 内核: axis0 = K/V 的 N 行块, axis1 = batch*kv_head
                每个 program 独占一个 (b, hkv, N块),把该块可见的所有 M 块串行流过,
                并在 GQA 组内对多个 q head 求和 → **累加器驻寄存器、写回一次、零原子**
    dQ 内核:    axis0 = Q 的 M 行块, axis1 = batch*q_head
                镜像做一遍(对 N 方向循环)→ 同样零原子

为什么不做原子版:原子加在 fp32 上要求 dK/dV 先清零再累加,既多一次 HBM 往返,
又让浮点求和顺序不确定(不可复现)。本仓的复现纪律要求 3 轮逐位可比。

接口契约:与 forward 同形状;额外需要 forward 的 (o, lse);
返回 (dq, dk, dv),dtype 与输入一致。

简化范围(相对官方 FA2,如实声明):只有 forward+backward,无 dropout/alibi/paged;
未做 split-K(长序列下 dK/dV 的 M 循环会成为串行瓶颈,见 EXP-T08 §7);
GQA 组内求和走串行循环而非并行归约。
"""

import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# ① preprocess:Δ_i = Σ_d O[i,d]·dO[i,d]
#    只沿 head_dim 归约,一个 program 处理一个 (b, hq, M块) 的行块
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _fa2_bwd_preprocess_kernel(
    O, dO, Delta,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_dob, stride_doh, stride_dom, stride_dod,
    n_ctx,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // NUM_Q_HEADS
    hq = pid_bh % NUM_Q_HEADS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < n_ctx

    o_ptrs = (O + b * stride_ob + hq * stride_oh
              + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    do_ptrs = (dO + b * stride_dob + hq * stride_doh
               + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod)
    # 越界行补 0:O 与 dO 都是 0 → Δ=0,backward 里那些行整体被 mask 掉
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + pid_bh * n_ctx + offs_m, delta, mask=mask_m)


# ─────────────────────────────────────────────────────────────────────────────
# ② dK / dV:每个 program 独占一个 (b, hkv, N块),GQA 组内串行求和
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _fa2_bwd_dkdv_kernel(
    Q, K, V, dO, LSE, Delta, dK, dV,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_lb, stride_lh,
    n_ctx,
    NUM_Q_HEADS: tl.constexpr,
    GQA_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bhkv = tl.program_id(1)
    b = pid_bhkv // (NUM_Q_HEADS // GQA_GROUP)
    hkv = pid_bhkv % (NUM_Q_HEADS // GQA_GROUP)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_n = offs_n < n_ctx

    # K/V 这一块整个 kernel 只载一次(N 块驻留),GQA 组内所有 q head 共用
    k_ptrs = (K + b * stride_kb + hkv * stride_kh
              + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    v_ptrs = (V + b * stride_vb + hkv * stride_vh
              + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

    # 累加器驻寄存器:全程不落 smem、不写 HBM,循环跑完才写一次 → 零原子
    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    # causal 下本 N 块只可能被 m ≥ n0 的行看到,故 M 循环从 n0 所在块起步
    # (对照 forward:那边是 N 循环的**上界**收缩,这边是**下界**抬升——同一个
    #  三角性,两个方向各用一次,合计省掉一半算力)
    lo_m = 0
    if IS_CAUSAL:
        lo_m = (pid_n * BLOCK_N) // BLOCK_M

    for g in range(GQA_GROUP):
        hq = hkv * GQA_GROUP + g
        for blk_m in range(lo_m, tl.cdiv(n_ctx, BLOCK_M)):
            offs_m = blk_m * BLOCK_M + tl.arange(0, BLOCK_M)
            mask_m = offs_m < n_ctx

            q_ptrs = (Q + b * stride_qb + hq * stride_qh
                      + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
            do_ptrs = (dO + b * stride_dob + hq * stride_doh
                       + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod)
            q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
            do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
            # LSE 越界行取 0(而非 -inf):下面对无效行显式置 -inf,见 valid
            # 【易错】LSE 是 (B,Hq,S) 的独立连续张量,必须用**它自己的** stride;
            # 借用 q 的 stride 会放大 D 倍直接越界(本 kernel 初版就栽在这)
            lse = tl.load(LSE + b * stride_lb + hq * stride_lh + offs_m,
                          mask=mask_m, other=0.0)
            delta = tl.load(Delta + b * stride_lb + hq * stride_lh + offs_m,
                            mask=mask_m, other=0.0)

            # 重算 S = Q·Kᵀ(backward 的核心动作:不存 P,现场算)
            qk = tl.dot(q, tl.trans(k)) * sm_scale

            # 【关键·易错】mask 必须把「行越界」一起并进来。
            # 只 mask 列(像 forward 那样)会让越界行的 qk 停在 0,而它们的
            # lse 被 load 成 0 → p = exp(0-0) = 1,整行假权重污染 dK/dV。
            # 把 mask_m 并入 valid 后,无效行 qk = -inf → p = 0,自动免疫。
            valid = mask_m[:, None] & mask_n[None, :]
            if IS_CAUSAL:
                valid = valid & (offs_m[:, None] >= offs_n[None, :])
            qk = tl.where(valid, qk, float("-inf"))

            # P 直接用 LSE 得归一化权重(exp(s - m - log l) = exp(s-m)/l)
            p = tl.exp(qk - lse[:, None])
            dp = tl.dot(do, tl.trans(v))              # dP = dO·Vᵀ

            # softmax 雅可比:dS = P ⊙ (dP − Δ),Δ 就是"行内 P 加权均值"那一项
            ds = p * (dp - delta[:, None])
            ds = ds.to(K.dtype.element_ty)

            dv += tl.dot(tl.trans(p.to(V.dtype.element_ty)), do)
            # 【易错·本实现初版就栽在这】s = scale·(q·kᵀ),而 qk 在上一行已经乘过
            # scale,所以 ds 是对**已缩放**分数求的导 → ∂s/∂q = scale·k,于是
            # dq/dk 必须再乘一次 scale。漏掉它梯度会整体放大 1/scale = √D 倍
            # (D=64 时正好 8 倍,与实测误差比完全吻合)。dv 不经过 scale,故不受影响
            # ——这正是"dv 对而 dq/dk 错"这个指纹的来源。
            dk += sm_scale * tl.dot(tl.trans(ds), q)

    dk_ptrs = (dK + b * stride_dkb + hkv * stride_dkh
               + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd)
    dv_ptrs = (dV + b * stride_dvb + hkv * stride_dvh
               + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd)
    tl.store(dk_ptrs, dk.to(dK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(dV.dtype.element_ty), mask=mask_n[:, None])


# ─────────────────────────────────────────────────────────────────────────────
# ③ dQ:镜像做一遍,对 N 方向循环
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _fa2_bwd_dq_kernel(
    Q, K, V, dO, LSE, Delta, dQ,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    stride_lb, stride_lh,
    n_ctx,
    NUM_Q_HEADS: tl.constexpr,
    GQA_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // NUM_Q_HEADS
    hq = pid_bh % NUM_Q_HEADS
    hkv = hq // GQA_GROUP

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < n_ctx

    # Q/dO/LSE/Δ 这一块只载一次(与 dKdV 内核镜像:k 那边驻 K/V,这边驻 Q/dO)
    q_ptrs = (Q + b * stride_qb + hq * stride_qh
              + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    do_ptrs = (dO + b * stride_dob + hq * stride_doh
               + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod)
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    lse = tl.load(LSE + b * stride_lb + hq * stride_lh + offs_m, mask=mask_m, other=0.0)
    delta = tl.load(Delta + b * stride_lb + hq * stride_lh + offs_m, mask=mask_m, other=0.0)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # causal 下可见列的上确界 = (pid_m+1)·BM(与 forward 完全同一处推导)
    hi = tl.minimum((pid_m + 1) * BLOCK_M, n_ctx) if IS_CAUSAL else n_ctx

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_ctx

        k_ptrs = (K + b * stride_kb + hkv * stride_kh
                  + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
        v_ptrs = (V + b * stride_vb + hkv * stride_vh
                  + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale
        valid = mask_m[:, None] & mask_n[None, :]
        if IS_CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        qk = tl.where(valid, qk, float("-inf"))

        p = tl.exp(qk - lse[:, None])
        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - delta[:, None])).to(K.dtype.element_ty)

        dq += sm_scale * tl.dot(ds, k)   # 同 dk:补回 ∂s/∂q 的 scale 因子

    dq_ptrs = (dQ + b * stride_dqb + hq * stride_dqh
               + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd)
    tl.store(dq_ptrs, dq.to(dQ.dtype.element_ty), mask=mask_m[:, None])


def fa2_backward(q, k, v, o, do, lse, causal=True, sm_scale=None,
                 block_m=64, block_n=64, num_warps=4, num_stages=2):
    """FA2 backward(简化版)。q/k/v/o/do 同形状约定见 forward;lse 来自
    `fa2_forward(..., return_lse=True)[1]`。返回 (dq, dk, dv)。

    重算而非存储:本函数不读任何 S×S 张量,多出来的 FLOPs 就是 recomputation
    的代价(总 FLOPs 约为 forward 的 2.5×),换来的是 HBM 从 O(S²) 回到 O(S·D)。
    """
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    assert Hq % Hkv == 0 and D in (64, 128) and q.is_cuda
    if sm_scale is None:
        sm_scale = D ** -0.5
    for t in (q, k, v, o, do):
        assert t.is_contiguous(), "本实现要求连续张量(后向 stride 按连续推)"

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    delta = torch.empty(B, Hq, S, device=q.device, dtype=torch.float32)

    grid_pre = (triton.cdiv(S, 128), B * Hq)
    _fa2_bwd_preprocess_kernel[grid_pre](
        o, do, delta,
        *o.stride(), *do.stride(), S,
        NUM_Q_HEADS=Hq, HEAD_DIM=D, BLOCK_M=128,
        num_warps=4,
    )

    grid_kv = (triton.cdiv(S, block_n), B * Hkv)
    _fa2_bwd_dkdv_kernel[grid_kv](
        q, k, v, do, lse, delta, dk, dv, sm_scale,
        *q.stride(), *k.stride(), *v.stride(), *do.stride(),
        *dk.stride(), *dv.stride(), lse.stride(0), lse.stride(1), S,
        NUM_Q_HEADS=Hq, GQA_GROUP=Hq // Hkv, HEAD_DIM=D,
        BLOCK_M=block_m, BLOCK_N=block_n, IS_CAUSAL=causal,
        num_warps=num_warps, num_stages=num_stages,
    )

    grid_q = (triton.cdiv(S, block_m), B * Hq)
    _fa2_bwd_dq_kernel[grid_q](
        q, k, v, do, lse, delta, dq, sm_scale,
        *q.stride(), *k.stride(), *v.stride(), *do.stride(),
        *dq.stride(), lse.stride(0), lse.stride(1), S,
        NUM_Q_HEADS=Hq, GQA_GROUP=Hq // Hkv, HEAD_DIM=D,
        BLOCK_M=block_m, BLOCK_N=block_n, IS_CAUSAL=causal,
        num_warps=num_warps, num_stages=num_stages,
    )
    return dq, dk, dv
