# SPDX-License-Identifier: MIT
"""MoE Permute/Unpermute(dispatch-combine 的单卡版,Triton)。

MoE 的 grouped GEMM 要求"同专家的 token 连续"——于是每层要做两次搬运:
  permute:  (T,D) 按 topk 展开并按专家排序 → (T*topk, D)
  unpermute:算完的行按 router 权重加权收回 → (T,D)
索引构建(排序/直方图)走 torch(与 vLLM moe_align 的 CUDA 索引核同职责);
**数据搬运走 Triton**:permute=gather;unpermute=每 token 收拢自己的 topk 行
再加权求和——gather 式实现,**天然无原子加**(对照:scatter 式必须 atomic)。
与 DeepEP 的关系见 docs/theory/07(单卡内存重排 ↔ 跨卡 all-to-all,同一
dispatch-combine 代数,不同传输介质)。

数据流(T=token 数,E=专家数):
  x (T,D) --permute(src_row)--> y (T*topk,D) --grouped GEMM(counts 给段界)-->
  y' (T*topk,D) --unpermute(pos, weights)--> out (T,D)

性能特征(EXP-T07,T4096/D2048/E60/top4,4090):unpermute 0.0845±0.0001 ms
vs torch 参考 1.053±0.002 = **12.5×**(来源=torch 路径四趟 kernel/中间张量
收成单 kernel);permute 1.8×(0.081 vs 0.146);索引构建(torch argsort+
bincount)0.266 ms——比两次数据搬运之和还贵,这就是 vLLM 为它写专用
kernel(moe_align_block_size)的存在理由。往返一致性 = bf16 噪声级。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_rows_kernel(X, IDX, Y, D: tl.constexpr, BLOCK_D: tl.constexpr):
    # 一行一个 program:读侧随机(按专家序取原 token 行)、写侧完全合并;
    # 每个输出行唯一属主 → 无写冲突。"乱读"不产生竞态,"乱写"才产生
    row = tl.program_id(0)
    src = tl.load(IDX + row)              # 间接寻址:排序位 → 源 token 行
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D                       # D 非 2 的幂时截到真实宽度
    v = tl.load(X + src * D + offs, mask=mask)
    tl.store(Y + row * D + offs, v, mask=mask)


@triton.jit
def _unpermute_kernel(Y, POS, W, OUT, TOPK: tl.constexpr,
                      D: tl.constexpr, BLOCK_D: tl.constexpr):
    # 面试点:gather 式无原子论证。scatter-add 视角是"每个专家输出行加回
    # out[所属 token 行]"——同一 token 的 topk 行并发撞同一目的行,必须
    # atomicAdd,且加法顺序随调度漂移(数值不可复现)。翻转成 gather 后:
    # program t 独占输出行 t,按 pos 自取 topk 行——写集合按构造两两不相交
    # (一行一属主),读的是只读缓冲无竞争,求和顺序固定 j=0..TOPK-1 →
    # 逐位可复现。代价只是读侧变随机,而随机读不构成 hazard
    t = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)   # 加权和固定 fp32:bf16 直接
    for j in range(TOPK):                         # 累加 topk 项会丢有效位
        p = tl.load(POS + t * TOPK + j)       # 该 (t,j) 行在排序缓冲中的位置
        w = tl.load(W + t * TOPK + j)
        acc += tl.load(Y + p * D + offs, mask=mask, other=0.0).to(tl.float32) * w
    tl.store(OUT + t * D + offs, acc.to(OUT.dtype.element_ty), mask=mask)


def build_indices(topk_ids: torch.Tensor, num_experts: int):
    """索引构建(torch):返回 src_row(排序位→原 token 行)、
    pos(每 (t,j) → 排序位)、每专家计数(grouped GEMM 的段界)。
    实测 0.266 ms(EXP-T07),比两次搬运之和还贵——专用化留作对照结论。"""
    T, topk = topk_ids.shape
    flat = topk_ids.reshape(-1)
    # stable:同专家内保持 token 原序 → pos 映射唯一确定,往返测试可逐位比对
    order = torch.argsort(flat, stable=True)          # 排序位 → flat 序号
    src_row = order // topk                           # flat 序号 = t*topk+j,
    pos = torch.empty_like(order)                     # 整除即还原 token 行 t
    # 逆置换:order 是"排序位→flat 序号",unpermute 端要的是反向映射
    pos[order] = torch.arange(order.numel(), device=order.device)
    counts = torch.bincount(flat, minlength=num_experts)
    return src_row.to(torch.int64), pos.reshape(T, topk).to(torch.int64), counts


def permute(x: torch.Tensor, src_row: torch.Tensor):
    """(T,D) → (T*topk, D) 按专家排序的展开。"""
    T2 = src_row.numel()
    D = x.shape[1]
    y = torch.empty(T2, D, device=x.device, dtype=x.dtype)
    BLOCK_D = triton.next_power_of_2(D)   # tl.arange 要求编译期 2 的幂
    # 宽行(≥2048 元素)给 8 warps 才能把行内 load 流水打满;窄行 4 warps
    # 免过度切分
    _gather_rows_kernel[(T2,)](x, src_row, y, D=D, BLOCK_D=BLOCK_D,
                               num_warps=8 if BLOCK_D >= 2048 else 4)
    return y


def unpermute(y: torch.Tensor, pos: torch.Tensor, weights: torch.Tensor,
              out_dtype=None):
    """(T*topk, D) + router 权重 → (T,D)。gather 式,无原子。"""
    T, topk = pos.shape
    D = y.shape[1]
    out = torch.empty(T, D, device=y.device, dtype=out_dtype or y.dtype)
    BLOCK_D = triton.next_power_of_2(D)
    # weights 统一转 fp32 contiguous:kernel 内累加是 fp32,权重先行对齐口径
    _unpermute_kernel[(T,)](y, pos, weights.float().contiguous(), out,
                            TOPK=topk, D=D, BLOCK_D=BLOCK_D,
                            num_warps=8 if BLOCK_D >= 2048 else 4)
    return out
