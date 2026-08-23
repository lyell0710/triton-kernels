# SPDX-License-Identifier: MIT
"""RMSNorm / 行 Softmax / INT8 per-channel quantize 的 Triton 版。

三个都是"一行一个 program"的 row-parallel 模式:整行载入寄存器 → 行内规约
→ 逐元素写回。与 CUDA 版的对应关系(讲解点):
- CUDA 里要手写 两级规约(warp shuffle + shared memory);Triton 的 tl.max/
  tl.sum 一句生成同等规约树——你写的是"要什么",编译器写"怎么做"。
- CUDA 的 float4 向量化访存 ≈ Triton 编译器对连续 tl.load 的自动向量化。
- 代价:CUDA 能控制 bank conflict / 具体指令;Triton 控不了——性能差距
  (若有)来自这里,见 EXP-T03 的同尺寸对比。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(X, W, Y, n_cols, eps,
                    stride_row, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(X + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / n_cols
    inv = 1.0 / tl.sqrt(ms + eps)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * stride_row + offs, (x * inv * w).to(Y.dtype.element_ty),
             mask=mask)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
    shp = x.shape
    x2 = x.reshape(-1, shp[-1]).contiguous()
    y = torch.empty_like(x2)
    BLOCK = triton.next_power_of_2(shp[-1])
    _rmsnorm_kernel[(x2.shape[0],)](x2, w, y, shp[-1], eps, x2.stride(0),
                                    BLOCK=BLOCK,
                                    num_warps=8 if BLOCK >= 2048 else 4)
    return y.reshape(shp)


@triton.jit
def _softmax_kernel(X, Y, n_cols, stride_row, BLOCK: tl.constexpr,
                    EXACT: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    if EXACT:            # 整除:无 mask → 编译器可发 128bit 向量化访存
        x = tl.load(X + row * stride_row + offs).to(tl.float32)
        mask = offs < BLOCK
    else:
        mask = offs < n_cols
        x = tl.load(X + row * stride_row + offs, mask=mask,
                    other=float("-inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    y = num / tl.sum(num, axis=0)
    tl.store(Y + row * stride_row + offs, y.to(Y.dtype.element_ty), mask=mask)


def softmax(x: torch.Tensor):
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    y = torch.empty_like(x2)
    BLOCK = triton.next_power_of_2(x.shape[-1])
    _softmax_kernel[(x2.shape[0],)](x2, y, x.shape[-1], x2.stride(0),
                                    BLOCK=BLOCK, EXACT=(BLOCK == x.shape[-1]),
                                    num_warps=8 if BLOCK >= 2048 else 4)
    return y.reshape(x.shape)


@triton.jit
def _int8_quant_kernel(X, Q, S, n_cols, stride_row, BLOCK: tl.constexpr,
                       EXACT: tl.constexpr):
    # per-channel(行)对称量化:scale = absmax/127,q = round(x/scale)
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    if EXACT:
        x = tl.load(X + row * stride_row + offs).to(tl.float32)
        mask = offs < BLOCK
    else:
        mask = offs < n_cols
        x = tl.load(X + row * stride_row + offs, mask=mask,
                    other=0.0).to(tl.float32)
    scale = tl.max(tl.abs(x), axis=0) / 127.0
    scale = tl.maximum(scale, 1e-8)
    q = tl.extra.cuda.libdevice.rint(x / scale)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(Q + row * stride_row + offs, q.to(tl.int8), mask=mask)
    tl.store(S + row, scale)


def int8_quantize(x: torch.Tensor):
    """x: (rows, cols) fp32/fp16 → (int8 q, fp32 per-row scale)。"""
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    q = torch.empty_like(x2, dtype=torch.int8)
    s = torch.empty(x2.shape[0], device=x.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(x.shape[-1])
    _int8_quant_kernel[(x2.shape[0],)](x2, q, s, x.shape[-1], x2.stride(0),
                                       BLOCK=BLOCK, EXACT=(BLOCK == x.shape[-1]),
                                       num_warps=8 if BLOCK >= 2048 else 4)
    return q.reshape(x.shape), s
