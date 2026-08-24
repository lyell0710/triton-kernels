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
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_rows_kernel(X, IDX, Y, D: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    src = tl.load(IDX + row)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    v = tl.load(X + src * D + offs, mask=mask)
    tl.store(Y + row * D + offs, v, mask=mask)


@triton.jit
def _unpermute_kernel(Y, POS, W, OUT, TOPK: tl.constexpr,
                      D: tl.constexpr, BLOCK_D: tl.constexpr):
    t = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for j in range(TOPK):
        p = tl.load(POS + t * TOPK + j)       # 该 (t,j) 行在排序缓冲中的位置
        w = tl.load(W + t * TOPK + j)
        acc += tl.load(Y + p * D + offs, mask=mask, other=0.0).to(tl.float32) * w
    tl.store(OUT + t * D + offs, acc.to(OUT.dtype.element_ty), mask=mask)


def build_indices(topk_ids: torch.Tensor, num_experts: int):
    """索引构建(torch):返回 src_row(排序位→原 token 行)、
    pos(每 (t,j) → 排序位)、每专家计数。"""
    T, topk = topk_ids.shape
    flat = topk_ids.reshape(-1)
    order = torch.argsort(flat, stable=True)          # 排序位 → flat 序号
    src_row = order // topk                           # gather 的源 token 行
    pos = torch.empty_like(order)
    pos[order] = torch.arange(order.numel(), device=order.device)
    counts = torch.bincount(flat, minlength=num_experts)
    return src_row.to(torch.int64), pos.reshape(T, topk).to(torch.int64), counts


def permute(x: torch.Tensor, src_row: torch.Tensor):
    """(T,D) → (T*topk, D) 按专家排序的展开。"""
    T2 = src_row.numel()
    D = x.shape[1]
    y = torch.empty(T2, D, device=x.device, dtype=x.dtype)
    BLOCK_D = triton.next_power_of_2(D)
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
    _unpermute_kernel[(T,)](y, pos, weights.float().contiguous(), out,
                            TOPK=topk, D=D, BLOCK_D=BLOCK_D,
                            num_warps=8 if BLOCK_D >= 2048 else 4)
    return out
