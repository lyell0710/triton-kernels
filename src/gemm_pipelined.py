# SPDX-License-Identifier: MIT
"""流水线(双缓冲)GEMM,Triton 版。

「双缓冲」在两个世界里的同一件事(讲解主线,详见 docs/theory/02):
- CUDA 手写:两块 shared memory 交替——计算 buf[0] 的同时用 cp.async 预取
  下一 K 块进 buf[1],循环末交换。隐藏的是 global→shared 的搬运延迟。
- Triton:同一循环体,`num_stages=N` 让编译器把 tl.load 与 tl.dot 软件流水
  化(N=1 无重叠,N=2 即双缓冲,N≥3 更深流水,shared memory 占用 ∝ N)。
本文件用同一个 kernel 扫 num_stages,把"双缓冲带来多少"变成一个可测数字。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(A, B, C,
                 M, N, K,
                 stride_am, stride_ak, stride_bk, stride_bn,
                 stride_cm, stride_cn,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                 BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
                 IEEE_DOT: tl.constexpr):
    pid = tl.program_id(0)
    # L2 友好的 grouped 调度:同组内先走 M 方向,提高 B 块复用
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M)
                    & (offs_k[None, :] + k0 < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k0 < K)
                    & (offs_n[None, :] < N), other=0.0)
        if IEEE_DOT:
            acc += tl.dot(a, b, input_precision="ieee")
        else:
            acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(C.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def gemm(a: torch.Tensor, b: torch.Tensor,
         block_m=128, block_n=128, block_k=64, group_m=8,
         num_warps=8, num_stages=None) -> torch.Tensor:
    """a: (M,K), b: (K,N),fp16 输入 fp32 累加输出 fp16。"""
    if num_stages is None:
        num_stages = 2 if a.dtype == torch.float32 else 3
    if a.dtype == torch.float32:
        block_n = min(block_n, 64)
    M, K = a.shape
    K2, N = b.shape
    assert K == K2 and a.is_cuda
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    _gemm_kernel[grid](a, b, c, M, N, K,
                       a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                       c.stride(0), c.stride(1),
                       BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
                       GROUP_M=group_m, IEEE_DOT=(a.dtype == torch.float32),
                       num_warps=num_warps, num_stages=num_stages)
    return c


def linear(x: torch.Tensor, weight: torch.Tensor, bias=None,
           **cfg) -> torch.Tensor:
    """nn.Linear 语义:y = x @ W^T + b。W: (out,in)——供 llm-engine D16 接入。
    小 M(decode)自适应缩 tile:BLOCK_M=128 在 M=1 时 127/128 全废。"""
    shp = x.shape
    x2 = x.reshape(-1, shp[-1]).contiguous()
    if "block_m" not in cfg:
        cfg["block_m"] = 128 if x2.shape[0] >= 128 else (
            32 if x2.shape[0] >= 32 else 16)
    y = gemm(x2.to(weight.dtype), weight.t().contiguous(), **cfg)
    if bias is not None:
        y = y + bias
    return y.reshape(*shp[:-1], weight.shape[0]).to(x.dtype)
