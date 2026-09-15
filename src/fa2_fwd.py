# SPDX-License-Identifier: MIT
"""Flash Attention 2 forward(简化版,Triton)。causal + GQA,fp16 输入 fp32 累加。

解决什么问题:标准 attention 要物化 S×S 的 P 矩阵,长序列下 HBM 读写量
O(S²) 成为瓶颈;本 kernel 把它降到 O(S·D),单 kernel 完成。

算法一句话:K/V 分块流过片上,softmax 用 online 规约(m/l 两个跑动统计量)
边算边修正,任何时刻都不存在完整的一行注意力权重。推导见
docs/theory/01_flashattention.md。

并行布局(grid 两维):
    axis0 = Q 的 M 行块(cdiv(S, BLOCK_M) 个)  axis1 = batch*q_head 扁平
    每个 program 独占一个 (b, hq, M块):Q tile 只载一次驻寄存器,K/V 整条
    流过 → program 间零通信零同步,输出行块互不重叠,天然无原子。

接口契约:q (B,Hq,S,D),k/v (B,Hkv,S,D),Hq % Hkv == 0,D ∈ {64,128},
S 任意(非整除由 mask 兜底);返回与 q 同 dtype/形状的 O。

性能特征(EXP-T01（Triton FA2 forward）,4090):S=4K(B1·H32/8·D128 fp16)达 SDPA-flash 的 87%
(1.118±0.002 vs 0.975±0.002 ms,123 TFLOPS);kperf 观测算力利用 74%、
occupancy 仅 17%(regs 213/线程,终端级证据登记于 EXP-T06（FP8 GEMM）§7)——
compute-bound,靠寄存器 ILP 而非高 occupancy 藏延迟;与官方 FA2 差的
~12% 在 tensor-core 布局/双缓冲/warp 专业化这层抽象税。

简化范围(相对官方 FA2,如实声明):只有 forward;无 dropout/alibi/paged;
seq 维边界用 mask 处理;每个 program 处理一个 (batch, q_head, M块)。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fa2_fwd_kernel(
    Q, K, V, O, LSE,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lb, stride_lh, stride_lm,
    n_ctx,
    NUM_Q_HEADS: tl.constexpr,
    GQA_GROUP: tl.constexpr,          # q_heads // kv_heads
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IEEE_DOT: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    pid_m = tl.program_id(0)          # 第几个 Q 行块
    pid_bh = tl.program_id(1)         # batch*q_head 扁平索引(两维并行压一维,免 3D grid)
    b = pid_bh // NUM_Q_HEADS
    hq = pid_bh % NUM_Q_HEADS
    hkv = hq // GQA_GROUP             # GQA:多个 q head 共享一个 kv head,直接换算
                                      # 索引读原 KV,不物化 repeat(省 HBM 与显存)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # Q tile 整个 kernel 只载这一次(M 块驻留),越界行补 0——补 0 行算出的
    # 结果是垃圾,但末尾 store 的行 mask 保证它们永不落地
    q_ptrs = (Q + b * stride_qb + hq * stride_qh
              + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < n_ctx, other=0.0)

    # 面试点:online softmax 的三个跑动量(全 fp32,fp16 累加长序列会丢位)。
    # 处理完前 n 列后的不变量——
    #   m_i = max(s[:n])                 当前见过的行最大值(只增不减)
    #   l_i = Σ exp(s[:n] - m_i)         以 m_i 为基准的未归一化行和
    #   acc = Σ exp(s[:n] - m_i) · V[:n] 同基准的未归一化输出
    # 任意时刻 acc/l_i 即"只看前 n 列"的精确 attention 输出;m_i 初值 -inf
    # 使首块的 alpha 换基自然退化为直接赋值
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 面试点:causal 循环上界推导——本块行号 ∈ [pid_m·BM, (pid_m+1)·BM),
    # 行 m 只可见列 n ≤ m,故可见列的上确界是 (pid_m+1)·BM;对角块之下的
    # 整块全可见(无需 mask),整块不可见的根本不进循环(算力直接省一半),
    # 只有行列区间相交的对角块需要循环内的逐元素 mask
    hi = tl.minimum((pid_m + 1) * BLOCK_M, n_ctx) if IS_CAUSAL else n_ctx

    for start_n in range(0, hi, BLOCK_N):
        curr_n = start_n + offs_n
        k_ptrs = (K + b * stride_kb + hkv * stride_kh
                  + curr_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
        v_ptrs = (V + b * stride_vb + hkv * stride_vh
                  + curr_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        # 越界列先补 0 载入,真正的剔除交给下面的 -inf 列 mask
        k = tl.load(k_ptrs, mask=curr_n[:, None] < n_ctx, other=0.0)
        v = tl.load(v_ptrs, mask=curr_n[:, None] < n_ctx, other=0.0)

        if IEEE_DOT:   # fp32 复跑证明算法精确性时关 TF32(10 位尾数)
            qk = tl.dot(q, tl.trans(k), input_precision="ieee") * sm_scale
        else:
            qk = tl.dot(q, tl.trans(k)) * sm_scale      # (M, N) fp32
        # 越界列必须置 -inf 而非 0:exp(-inf)=0 才不污染 l 与 acc
        # (置 0 会给每个越界列贡献 e^{-m} 的假质量)
        qk = tl.where(curr_n[None, :] < n_ctx, qk, float("-inf"))
        if IS_CAUSAL:
            # 对角块的逐元素 mask;对已整块可见的块此条件恒真——
            # 无条件套用省掉"是否对角块"的控制流分支,谓词开销可忽略
            qk = tl.where(offs_m[:, None] >= curr_n[None, :], qk, float("-inf"))

        # online 更新:基准从 m_i 抬到 m_new,旧的 l/acc 统一乘
        # alpha = e^{m_i-m_new} ≤ 1 换基——数学恒等而非近似;m 单调不减
        # 保证所有 exp 参数 ≤ 0,不会上溢
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if IEEE_DOT:
            acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        else:
            # p 降回 fp16 走 tensor core;精度损失由 6 形状 gate(<2e-2)兜底
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # 面试点:归一化只做这一次(FA2 对 FA1 的关键改进——循环内维护未归一化
    # acc,省掉每块一次的除法/重缩放)。有效行 l_i 恒 >0(causal 下每行至少
    # 可见对角元自身);越界填充行全列被 mask → l_i=0 → 0/0=nan,但被下方
    # store 的行 mask 拦截,nan 不落地
    acc = acc / l_i[:, None]

    o_ptrs = (O + b * stride_ob + hq * stride_oh
              + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < n_ctx)

    if WRITE_LSE:
        # backward 需要的行统计量:LSE = m + log(l)。等价于官方 FA2 的
        # softmax_lse;有了它 backward 里 p = exp(s - LSE) 直接就是归一化
        # 后的 P(省掉一次除)。越界填充行 l_i=0 → log(0)=-inf → LSE=-inf,
        # backward 侧对所有越界行整体做 mask,不会污染。
        # 【易错】用 LSE 自己的 stride,不能借用 O 的:O 是 (B,Hq,S,D) 而 LSE 是
        # (B,Hq,S),借用会把偏移放大 D 倍 → 越界写(初版即因此触发 illegal access)
        tl.store(LSE + b * stride_lb + hq * stride_lh + offs_m * stride_lm,
                 m_i + tl.log(l_i), mask=offs_m < n_ctx)


def fa2_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                causal: bool = True, sm_scale: float | None = None,
                block_m: int | None = None, block_n: int = 64,
                num_warps: int | None = None,
                num_stages: int | None = None,
                return_lse: bool = False):
    """q: (B, Hq, S, D); k/v: (B, Hkv, S, D),Hq 必须是 Hkv 的整数倍。

    默认 tile BM128/BN64/w8/s2 来自 EXP-T01 扫描:4K 上 BM128 比 BM64 +17%,
    BN=128 撞 shared memory 上限 OOM。tile 按 dtype 自适应:fp32 的 tile
    字节翻倍,BM128 会超 Ada 100KB shared memory 上限(EXP-T01 实测),故
    fp32 降 BM32/w4/s1(校验路线,只求精确不求峰值)。

    return_lse=True 时额外返回 LSE = m + log(l)(形状 (B, Hq, S), fp32)——
    这是 backward 需要的行统计量,等价于官方 FA2 存的 softmax_lse。
    默认 False 保持与 EXP-T01 既有接口/性能口径完全一致。
    """
    B, Hq, S, D = q.shape
    if block_m is None:
        block_m = 32 if q.dtype == torch.float32 else 128
    if num_warps is None:
        num_warps = 4 if q.dtype == torch.float32 else 8
    if num_stages is None:
        num_stages = 1 if q.dtype == torch.float32 else 2
    Hkv = k.shape[1]
    assert Hq % Hkv == 0 and D in (64, 128) and q.is_cuda
    if sm_scale is None:
        sm_scale = D ** -0.5
    o = torch.empty_like(q)
    lse = torch.empty(B, Hq, S, device=q.device, dtype=torch.float32) if return_lse else None
    grid = (triton.cdiv(S, block_m), B * Hq)
    _fa2_fwd_kernel[grid](
        q, k, v, o, lse if return_lse else o, sm_scale,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        *(lse.stride()[:2] + (lse.stride(2),) if return_lse
          else (o.stride(0), o.stride(1), o.stride(2))),
        S,
        NUM_Q_HEADS=Hq, GQA_GROUP=Hq // Hkv, HEAD_DIM=D,
        BLOCK_M=block_m, BLOCK_N=block_n, IS_CAUSAL=causal,
        IEEE_DOT=(q.dtype == torch.float32),
        WRITE_LSE=return_lse,
        num_warps=num_warps, num_stages=num_stages,
    )
    return (o, lse) if return_lse else o
