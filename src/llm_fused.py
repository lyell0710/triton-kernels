"""LLM 前向里的三个融合逐元素算子:fused_add_rmsnorm / rope / silu_and_mul。

这三个算子在 pre-norm decoder 的每一层各出现 1-2 次,全部是访存主导:
每元素只有几次乘加,却要搬 6-10 字节。它们是「融合免搬运」这一类优化的
最纯粹样本 —— 没有 Tensor Core 可用,唯一的杠杆就是砍掉中间量的显存往返。

与 Kernel_Optimazation 下的手写 CUDA 版本一一对应,两边在同一个 harness 下受测
(Kernel_Optimazation/<op>/bench.py 的 triton_* 臂)。这条对照是本项目
「什么时候该用 CUDA」判断曲线的第三个点:
  - 计算主导的 GEMM:手写 CUDA wmma 够到真 cuBLAS 的 86%(Kernel#EXP-K02)
  - 融合型 attention:同一套 wmma 只够到自家 Triton 的 28%(Kernel#EXP-K03)
  - 访存主导的融合逐元素算子:本文件 —— 预期两边都贴带宽墙、打平
前两点已测,第三点由本文件补上。

Triton 在这类算子上的结构优势:一个 program 处理一行,整行常驻寄存器,
归约由编译器生成 —— 手写 CUDA 需要显式写成「寄存器缓存版」(v4)才能达到
同样的访存次数,而 Triton 的自然写法天然就是那一版。
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# 1) fused_add_rmsnorm:residual += x; out = rmsnorm(residual) * w
# ---------------------------------------------------------------------------
@triton.jit
def _fused_add_rmsnorm_kernel(X, RES, W, OUT, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols                      # BLOCK 取 2 的幂,尾部靠 mask 关掉
    base = row * n_cols

    # fp32 中间量:bf16 尾数 8 位,H=4096 个平方在 bf16 上累加会吃掉低位
    # (与 CUDA 版 include/fused_norm.h 的精度约定同一条理由)
    x = tl.load(X + base + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(RES + base + cols, mask=mask, other=0.0).to(tl.float32)
    s = x + r

    # 立刻写回:下一层的残差流要用它。注意后面归一化用的是寄存器里的 s,
    # 不是重新 load 回来的值 —— 这就是手写 CUDA 版 v4 的「寄存器缓存」,
    # 在 Triton 里是默认行为,不需要额外写。
    tl.store(RES + base + cols, s.to(tl.bfloat16), mask=mask)

    rstd = tl.rsqrt(tl.sum(s * s, axis=0) / n_cols + eps)
    w = tl.load(W + cols, mask=mask, other=0.0)
    # 舍入顺序与 vLLM layernorm.cu / CUDA 版保持一致:先舍到 bf16 再乘权重
    y = (s * rstd).to(tl.bfloat16) * w
    tl.store(OUT + base + cols, y, mask=mask)


def fused_add_rmsnorm(x, residual, w, eps=1e-6, out=None):
    """residual 就地更新为 residual+x;返回 rmsnorm(residual)*w。"""
    assert x.is_contiguous() and residual.is_contiguous()
    H = x.shape[-1]
    x2 = x.view(-1, H)
    r2 = residual.view(-1, H)
    out = torch.empty_like(x) if out is None else out
    BLOCK = triton.next_power_of_2(H)
    # num_warps 按 BLOCK 给:一行越长越需要更多 warp 分摊,否则单 warp 要
    # 循环很多次。这条经验值与 src/elementwise_kernels.py 的行核一致。
    num_warps = max(4, min(16, BLOCK // 256))
    _fused_add_rmsnorm_kernel[(x2.shape[0],)](
        x2, r2, w, out.view(-1, H), H, eps, BLOCK=BLOCK, num_warps=num_warps)
    return out


# ---------------------------------------------------------------------------
# 2) rope:按「前后半段」布局做旋转位置编码(HF llama/Qwen 约定,
#    即 vLLM 的 is_neox=True 分支)。q 与 k 在同一个 kernel 里转完。
# ---------------------------------------------------------------------------
@triton.jit
def _rope_kernel(P, COS, SIN, Hh, D: tl.constexpr, n_pair,
                 HALF: tl.constexpr, BLOCK: tl.constexpr):
    """单张量就地旋转;每个 program 负责 BLOCK 个「(token, head, 频率) 对」。

    粒度与形态在这个 kernel 上试错了三轮,每轮都是访存形态的问题,值得记下来:
      ① 一个 program 一个 (token, head):T=32768/HQ=32 时发 100 万个 program,
         每个只搬 64 个元素,调度开销吃掉一半设备时间。
      ② 一个 program 一个 token、内部 [BQ, HALF] 二维 tile:program 数降到 T,
         但行跨度是 D(读 128B 跳 128B),访存拆成半事务。
      ③ q/k 合进一个 kernel、用 mask 选择:实测有效带宽恰好只有手写 CUDA 版的
         一半 —— 「恰好一半」说明搬了两倍字节,即两路 masked load 都产生了
         实际访存。Triton 的 mask 保证语义正确,不保证被 mask 掉的那路不发事务。
    现在这版一次只喂一张张量、下标一维连续,q 与 k 分两次 launch。
    多一次 launch 的代价在 prefill 可忽略、在 decode 会显出来(见 bench 的
    decode 区间)—— 这正是手写 CUDA v2「q/k 合并 launch」那一级在 Triton 侧
    做不到的东西,也是两种语言表达力差异的一个具体样本。
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_pair
    half = D // 2

    # 【关键两行】offs 经过 % 与 // 之后,编译器无法再推断地址的连续性,
    # 会把访存退化成逐元素的 16 位加载 —— 一个 warp 只覆盖 32*2=64B,
    # 半个 128B 事务被浪费,实测有效带宽恰好只有向量化版本的一半。
    # tl.max_contiguous / tl.multiple_of 是把「我知道它连续」这件事显式告诉
    # 编译器的规范手段:i 在每 half 个 lane 内是 0..half-1 的连续序列,
    # hp*D 则是 D 的整数倍。给出这两条断言后向量化才会生效。
    i = tl.max_contiguous(tl.multiple_of(offs % half, HALF), HALF)
    hp = offs // half          # 第几个 (token, head)
    tok = hp // Hh
    base = tl.multiple_of(hp * D, D) + i

    cs = tl.load(COS + tok * D + i, mask=mask, other=0.0).to(tl.float32)
    sn = tl.load(SIN + tok * D + i, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(P + base, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(P + base + half, mask=mask, other=0.0).to(tl.float32)

    # 复数乘 (x1 + i*x2)*(c + i*s);两个输出都算完再写,读写依赖收在 program 内,
    # 不存在手写 CUDA v0 那种「一线程一元素 + 就地更新」的跨线程覆盖问题。
    tl.store(P + base, (x1 * cs - x2 * sn).to(P.dtype.element_ty), mask=mask)
    tl.store(P + base + half, (x2 * cs + x1 * sn).to(P.dtype.element_ty), mask=mask)


def rope(q, k, cos, sin):
    """q:[T,HQ,D] k:[T,HK,D] cos/sin:[T,D];就地旋转。

    cos/sin 的前后半段是重复的同一组频率(见 llm-engine src/layers.py
    precompute_rope 的 cat((freqs, freqs))),所以只取前 D/2 即可。
    """
    assert q.is_contiguous() and k.is_contiguous()
    T, HQ, D = q.shape
    HK = k.shape[1]
    BLOCK = 1024
    for t, Hh in ((q, HQ), (k, HK)):
        n_pair = T * Hh * (D // 2)
        _rope_kernel[(triton.cdiv(n_pair, BLOCK),)](
            t, cos, sin, Hh, D, n_pair, HALF=D // 2, BLOCK=BLOCK, num_warps=4)
    return q, k


# ---------------------------------------------------------------------------
# 3) silu_and_mul:out = silu(gate) * up(SwiGLU 的逐元素部分)
# ---------------------------------------------------------------------------
@triton.jit
def _silu_and_mul_kernel(G, U, OUT, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U + offs, mask=mask, other=0.0).to(tl.float32)
    # silu(g) = g * sigmoid(g);在 fp32 里算 sigmoid 再乘,与 PyTorch 的
    # bf16 opmath=float 一致。直接在 bf16 上算 exp 会在 |g| 较大时明显失真。
    y = g * tl.sigmoid(g) * u
    tl.store(OUT + offs, y.to(OUT.dtype.element_ty), mask=mask)


def silu_and_mul(gate, up, out=None):
    assert gate.shape == up.shape and gate.is_contiguous() and up.is_contiguous()
    out = torch.empty_like(gate) if out is None else out
    n = gate.numel()
    BLOCK = 1024
    _silu_and_mul_kernel[(triton.cdiv(n, BLOCK),)](gate, up, out, n,
                                                   BLOCK=BLOCK, num_warps=4)
    return out
