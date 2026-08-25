# SPDX-License-Identifier: MIT
"""流水线(双缓冲)GEMM,Triton 版。

「双缓冲」在两个世界里的同一件事(讲解主线,详见 docs/theory/02):
- CUDA 手写:两块 shared memory 交替——计算 buf[0] 的同时用 cp.async 预取
  下一 K 块进 buf[1],循环末交换。隐藏的是 global→shared 的搬运延迟。
- Triton:同一循环体,`num_stages=N` 让编译器把 tl.load 与 tl.dot 软件流水
  化(N=1 无重叠,N=2 即双缓冲,N≥3 更深流水,shared memory 占用 ∝ N)。
本文件用同一个 kernel 扫 num_stages,把"双缓冲带来多少"变成一个可测数字。

接口契约:gemm(a (M,K), b (K,N)) → (M,N),fp16 输入 fp32 累加输出 fp16;
linear() 提供 nn.Linear 语义供 llm-engine 接入。

性能特征(EXP-T02,4090 fp16):4096³ 随 stages 1→4 = 131.9/133.5/160.5/
157.1 TFLOPS,最优 stages=3 打平 cuBLAS 159.8(3 轮复现 159.4±1.2 vs
160.0±0.7);Qwen3-8B up_proj 形状 154.4 vs 147.3 反超 4.8%。kperf 观测:
算力 98% 而 occupancy 仅 17%(regs 170/线程,终端级证据登记于 EXP-T06 §7)。

面试点(两个反直觉):
1. stages=2(经典双缓冲)只 +1%,stages=3 才 +21%——流水深度要按
   「一次 BLOCK_K 搬运延迟 / 一轮 dot 计算时长」配;Ada 上该比值 >1,
   双缓冲只遮住一段,拷贝仍在关键路径上,必须再加一级才把 load 摘出去。
2. occupancy 17% 却打出 98% 峰值——tensor core kernel 靠寄存器堆 ILP
   藏延迟,比高 occupancy 更值钱;"occ 低"只在延迟藏不住时才是嫌疑人。
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
    # L2 友好的 grouped 调度:把线性 pid 重映射成"先在 M 方向排满 GROUP_M 行
    # 再换 N 列"——时间上相邻的 CTA 命中同一批 B 列块,B tile 的 L2 复用
    # 距离从 num_pid_n 缩到 GROUP_M;朴素 row-major 顺序下相邻 CTA 各取
    # 各的 B 块,大 N 时 B 反复走 HBM
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    # min 处理 M 方向最后一个残组(不足 GROUP_M 行时组内映射要按实际行数取模)
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # 主循环:每轮搬一个 BLOCK_K 条带并 dot 进累加器。num_stages=N 时编译器
    # 就在这个循环上做软件流水:分配 N 份 smem 缓冲,dot 消费第 i 份的同时
    # cp.async 预取第 i+N-1 份——CUDA 手写双缓冲在这里退化为一个 launch
    # 参数;代价是 smem 占用 ∝ N,stages 过深反过来压 CTA 并发
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        # K 尾块 mask:越界补 0,对 dot 零贡献,免去主循环外单写一个收尾循环
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M)
                    & (offs_k[None, :] + k0 < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k0 < K)
                    & (offs_n[None, :] < N), other=0.0)
        if IEEE_DOT:
            # fp32 校验路线:禁 TF32(10 位尾数),换取与参考实现可比的精度
            acc += tl.dot(a, b, input_precision="ieee")
        else:
            # acc 作第三参数:直接映射 mma 指令的累加寄存器,免独立 add 一趟
            acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(C.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def gemm(a: torch.Tensor, b: torch.Tensor,
         block_m=128, block_n=128, block_k=64, group_m=8,
         num_warps=8, num_stages=None) -> torch.Tensor:
    """a: (M,K), b: (K,N),fp16 输入 fp32 累加输出 fp16。

    默认 128×128×64/w8:Ada 寄存器预算内的最大方形 tile(仅 acc 就占
    128·128 fp32 / 256 线程 = 64 regs/线程,kperf 实测总 170);
    num_stages 默认 3 = EXP-T02 扫描最优;fp32 输入 tile 字节翻倍,
    降 BN64/stages2(校验路线,不追峰值)。"""
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
    小 M(decode)自适应缩 tile:BLOCK_M=128 在 M=1 时 mma 行利用率 1/128,
    127/128 的 tensor core 算力全废,故按实际行数降到 32/16。"""
    shp = x.shape
    x2 = x.reshape(-1, shp[-1]).contiguous()
    if "block_m" not in cfg:
        cfg["block_m"] = 128 if x2.shape[0] >= 128 else (
            32 if x2.shape[0] >= 32 else 16)
    y = gemm(x2.to(weight.dtype), weight.t().contiguous(), **cfg)
    if bias is not None:
        y = y + bias
    return y.reshape(*shp[:-1], weight.shape[0]).to(x.dtype)
