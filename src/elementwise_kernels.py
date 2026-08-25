# SPDX-License-Identifier: MIT
"""RMSNorm / 行 Softmax / INT8 per-channel quantize 的 Triton 版。

三个都是"一行一个 program"的 row-parallel 模式:整行载入寄存器 → 行内规约
→ 逐元素写回。与 CUDA 版的对应关系(讲解点):
- CUDA 里要手写 两级规约(warp shuffle + shared memory);Triton 的 tl.max/
  tl.sum 一句生成同等规约树——你写的是"要什么",编译器写"怎么做"。
- CUDA 的 float4 向量化访存 ≈ Triton 编译器对连续 tl.load 的自动向量化。
- 代价:CUDA 能控制 bank conflict / 具体指令;Triton 控不了——性能差距
  (若有)来自这里,见 EXP-T03 的同尺寸对比。

性能特征——必须拆三个口径引用(EXP-T03/T05,4090):
- 设备侧:softmax 8192² Triton 917 vs torch 922 GB/s,同速,双双贴
  roofline 91%(kperf:带宽 91%,occ 67%);
- launch 层:1024² 的 4.4× "差距"全在主机侧(8×8 纯开销 37.4 vs 8.0µs,
  Triton Python 分发 > torch C++ 分发);终局解是 CUDA Graph:每调用
  36.2±0.1 → 3.11µs,**11.6×** 塌缩,graph 后 Triton 反超 torch eager
  (EXP-T05);
- 端到端:int8 三数字口径不得混引(LEDGER 红线)——5.9µs(裸 CUDA v4,
  scale 预置,EXP-K01)/ 65.1µs(ext 绑定端到端,含 3 次前置 launch)/
  41.6~52µs(本文件 triton 单 kernel 融合,随系统状态波动引用带区间);
  融合数(launch 数)比单 kernel 快慢更重要。
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
    # 规约统一升 fp32:fp16 平方和几千元素就开始丢位,均值会系统性偏低
    x = tl.load(X + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    # other=0.0 使越界列对平方和零贡献;分母用真实 n_cols 而非 BLOCK
    ms = tl.sum(x * x, axis=0) / n_cols
    inv = 1.0 / tl.sqrt(ms + eps)         # eps 在 sqrt 内:全零行防除 0
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * stride_row + offs, (x * inv * w).to(Y.dtype.element_ty),
             mask=mask)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
    shp = x.shape
    x2 = x.reshape(-1, shp[-1]).contiguous()
    y = torch.empty_like(x2)
    BLOCK = triton.next_power_of_2(shp[-1])   # tl.arange 要求编译期 2 的幂
    # 宽行(≥2048)8 warps 才够把行内 load/规约流水打满;窄行 4 warps 免切碎
    _rmsnorm_kernel[(x2.shape[0],)](x2, w, y, shp[-1], eps, x2.stride(0),
                                    BLOCK=BLOCK,
                                    num_warps=8 if BLOCK >= 2048 else 4)
    return y.reshape(shp)


@triton.jit
def _softmax_kernel(X, Y, n_cols, stride_row, BLOCK: tl.constexpr,
                    EXACT: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    # 面试点:EXACT 快路径的来历——最初假设"mask load 阻断 128bit 向量化"
    # 导致 Triton 慢,于是加了整除免 mask 分支;对照实验证伪:数字纹丝不动,
    # 真正的差距在主机侧 launch(EXP-T03 §7)。快路径语义无害故保留,
    # 作为"猜测必须交给对照实验"的物证
    if EXACT:            # 整除:无 mask load(假设已证伪,见上)
        x = tl.load(X + row * stride_row + offs).to(tl.float32)
        mask = offs < BLOCK               # 恒真,仅为与慢路径共用 store 签名
    else:
        mask = offs < n_cols
        # 越界补 -inf:exp(-inf)=0,不进分母(补 0 会污染归一化)
        x = tl.load(X + row * stride_row + offs, mask=mask,
                    other=float("-inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)             # 减行 max 防 exp 上溢(最大指数归 0)
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
    # per-channel(行)对称量化:scale = absmax/127,q = round(x/scale)。
    # absmax 规约 + 缩放 + 舍入一趟完成 → 单次 launch;对照 ext-CUDA 路径
    # 的"3 次 scale 前置 launch + kernel",这正是 41.6~52 vs 65.1µs 端到端
    # 反转的机理(EXP-T03)
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    if EXACT:            # 快路径同 _softmax_kernel:假设已证伪,语义无害保留
        x = tl.load(X + row * stride_row + offs).to(tl.float32)
        mask = offs < BLOCK
    else:
        mask = offs < n_cols
        x = tl.load(X + row * stride_row + offs, mask=mask,
                    other=0.0).to(tl.float32)   # 补 0:不影响 absmax(≥0)
    scale = tl.max(tl.abs(x), axis=0) / 127.0
    scale = tl.maximum(scale, 1e-8)       # 全零行防除 0
    q = tl.extra.cuda.libdevice.rint(x / scale)   # rint=就近舍入;直接 to(int8)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)  # 是截断,会引入半 LSB 偏差
    # clamp ±127(弃 -128):对称量化保证 q 与 -q 都可表示
    tl.store(Q + row * stride_row + offs, q.to(tl.int8), mask=mask)
    tl.store(S + row, scale)              # per-row scale,下游反量化用


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
