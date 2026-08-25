# 讲义 01 · 从 softmax 的减 max 推到 FlashAttention-2,再推到 flash-decoding

> 读者:准备校招面试的作者本人,以及第一次动手写 attention kernel 的工程师。
> 读法:不跳步。每个论断后面跟着它的证据锚(EXP 编号 / 文件:行号 / raw 路径),
> 所有数字与仓内现行口径逐字一致,来源见 records/ 与 data/derived/。

## 1. 这一篇回答什么问题

一个 attention kernel 从"数值稳定的 softmax"一路长成"不物化 S×S 的 FA2",
再长成"decode 专用的 split-K flash-decoding",中间每一步都有一个可以手推的
理由。读完你应当能:

- 手推三件事:①softmax 为什么必须减 max、减完为什么**精确**等价而不是近似;②分块
  softmax 的可归并性代数(换基引理 + 合并公式),以及同一套代数为什么能既做"块内
  online"又做"块间 reduce";③FA1→FA2 的循环反转为什么能把 $(m,l,\mathrm{acc})$
  整段留在寄存器里。
- 说清"87%"的**四要素限定**:简化版、仅 forward、4K 形状(B1·H32/8·D128)、
  对照 = SDPA flash 后端——缺一不引(本仓措辞约定);并把差掉的那部分拆到
  "算力利用率"这一层。
- 答上"flash-decoding 到底快几倍":**2.24±0.11×(naive repeat 预置口径)/
  5.17±0.24×(含 repeat 实体化口径)**,32K 上下文(EXP-T04),并主动说出反面——
  **Skv ≤ 8K 反而只有 0.86-0.88×**,以及这个反亏为什么与算法无关。

## 2. 直觉与第一性原理

**没有 FlashAttention 的世界**:$O = \mathrm{softmax}(QK^\top/\sqrt{d})V$ 照字面写,中间
必然出现一个 $S\times S$ 分数矩阵。本仓 bench 形状 B=1、Hq=32、S=4096、fp16 下它是
$4096^2\times32\times2\,\mathrm{B}=1.07\,\mathrm{GB}$,一次前向至少写一遍、读一遍、写回
一遍、再读一遍,约 $3.2\,\mathrm{GB}$ 的 HBM 往返,在 4090 的 1008 GB/s 上就是
$3.2\,\mathrm{ms}$;而算力账只有 $2BH_qS^2D = 137.4\,\mathrm{GFLOP}$(causal 折半后),
在 165 TFLOPS 的 fp16 tensor core 上是 $0.83\,\mathrm{ms}$。**物化 $S\times S$ 把一个
本该 compute-bound 的算子按成 memory-bound,而且按得很深。**

仓内量尺(EXP-T01,3 轮):S=2048 上 naive fp32 参考 $6.694\pm0.003\,\mathrm{ms}$,本仓
kernel $0.3296\pm0.0007\,\mathrm{ms}$(data/derived/exp-t01_stability_3rounds.csv)——
20 倍的差距里算法一个 FLOP 都没少,少的全是那趟 HBM 往返。

**为什么"不物化"是可能的**:softmax 的分母是求和,天生可分块累加;唯一障碍是分子里的
$e^{x-\max}$ 需要"整行的 max",而分块时你只有"到目前为止的 max"。FA 的全部魔法就是:
**旧 max 算出来的东西乘一个标量就能换到新 max 的基准上**(§3.2 换基引理)。

**日常类比与它的失效点**:像连续称重时用"目前最重的一件"当基准记录相对重量,
来了更重的就把之前所有记录统一乘一个折算系数。类比在两处失效:

1. 类比里基准换晚了只是记录难看,attention 里换晚了是 $e^x$ **溢出成 inf**、接着
   $\mathrm{inf}/\mathrm{inf}=\mathrm{nan}$ 污染整行——"结果作废"而非"精度差一点"。
2. 类比里的合并是普通加法(可交换、可结合、可原子累加),softmax 的合并是**带换基
   因子的加法**——每项在加之前都要乘一个依赖全局 max 的标量。这正是 flash-decoding
   不能用 atomicAdd 归并、必须拆两个 kernel 的根本原因(§3.6)。

**为什么选"减 max"而不是"换更宽的浮点"**:换宽只把溢出点往后推(fp16 溢出线
$x>11.09$,fp32 是 $x>88.7$),不改变"存在会炸的输入"这个事实;减 max 把指数参数
压到 $\le 0$,**无条件**安全且不花额外带宽。优先选把问题消掉的变换,不是推远的变换。

## 3. 完整推导与机制

### 3.1 减 max 的必要性:精确恒等式,不是技巧

逐步推,每步写清"凭什么可以这么做":

1. 定义 $\mathrm{softmax}(x)_i = e^{x_i}/\sum_j e^{x_j}$——这是定义,没有自由度。
2. 对任意常数 $c$,分子分母同乘 $e^{-c}$:
   $$\frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i - c}}{\sum_j e^{x_j - c}}$$
   ——凭的是 $e^{a+b} = e^a e^b$ 与"分子分母同乘非零数不改变商"。这是**恒等式**,
   对任何 $c$ 都精确成立,一位有效数字都不损失。
3. 于是 $c$ 成了自由参数,选它的标准只剩数值范围。取 $c=\max_j x_j$ 时所有
   $x_i-c\le0$,故 $e^{x_i-c}\in(0,1]$ **永不上溢**;且至少有一项等于 1,分母
   $\ge1$,**永不下溢成 0**。比 max 小的 $c$ 可能上溢,大很多的 $c$ 会让整行下溢——
   max 是同时把两侧余量做到最大的那个选择。
4. 代价:要拿到 $\max_j x_j$ 就得先扫一遍整行 → 朴素实现是**三趟**(求 max、求
   $\sum e$、归一化)。online softmax 把三趟压成**一趟**,代价是每一步维护"当前基准"
   并在基准变化时校正历史。这就是 §3.2。

仓内落点:非分块版就是 `x = x - tl.max(x, axis=0)`(src/elementwise_kernels.py:75,
讲义 03 第 1 段走读);FA2 里它变成跑动版
`m_new = tl.maximum(m_i, tl.max(qk, 1))`(src/fa2_fwd.py:110)。同一条数学,区别只在
"扫一遍就知道 max"还是"边扫边修正 max"。

### 3.2 分块可归并性:换基引理与合并算子

**部分统计量**(第 $p$ 块 K/V,分数 $s_j = q\cdot k_j\cdot\text{scale}$):

$$m_p = \max_{j\in p} s_j,\qquad l_p = \sum_{j\in p} e^{s_j - m_p},\qquad
\mathrm{acc}_p = \sum_{j\in p} e^{s_j - m_p}\, v_j$$

注意 $\mathrm{acc}_p$ **没有除以** $l_p$——这是全部推导的枢纽。

**换基引理**:$e^{s-m'} = e^{s-m}\cdot e^{m-m'}$,凭的还是 $e^{a+b}=e^ae^b$;关键在于
因子 $e^{m-m'}$ **与 $j$ 无关**,可以从求和号里提出来——一个标量乘法就把整块历史换到
新基准。

**合并算子**:定义 $(m_A,l_A,\mathrm{acc}_A) \oplus (m_B,l_B,\mathrm{acc}_B)$ 为

$$m = \max(m_A, m_B),\quad
l = l_A e^{m_A-m} + l_B e^{m_B-m},\quad
\mathrm{acc} = \mathrm{acc}_A e^{m_A-m} + \mathrm{acc}_B e^{m_B-m}$$

代入定义即可验证:合并结果恰好等于"把 A∪B 当一块直接算"的三元组——凭的是换基引理把
两块换到同一基准后,求和退化为普通加法。三条性质,每条都在实现里被用到:

- **可结合、可交换**:$\oplus$ 只依赖 $\max$ 与加法。所以分块方式、归并顺序、树形
  还是线性,数学结果同一个(浮点舍入除外)——这就是"块内 online 顺序扫"与"块间
  一次性 reduce"能共用一套代数的原因。
- **单位元** $(-\infty, 0, \mathbf{0})$:空块的贡献 $e^{-\infty-m}=0$ 自动湮灭,
  不需要任何 if 分支——src/flash_decode.py:121-123 直接吃这条性质。
- **除法必须最后做**:若每块先算 $\mathrm{acc}_p/l_p$ 就丢了 $l_p$ 这个权重,合并得写成
  加权平均
  $\big(\sum_p l_p e^{m_p-m}\cdot\tfrac{\mathrm{acc}_p}{l_p}\big)/\sum_p l_p e^{m_p-m}$,
  $l_p$ 还是得带着走。"最后再除"是**可归并性的前提**,不是省除法的小优化。

**同一代数用两次**(白板卡 docs/talk/whiteboard_card_fa2_algebra.md 的主线):块内
online 是"(已累积的历史) ⊕ (新的 K/V 块)"(src/fa2_fwd.py:107-119),块间 reduce 是
"(段 0) ⊕ (段 1) ⊕ … ⊕ (段 p)"(src/flash_decode.py:115-123)。

### 3.3 HBM 流量账:被消掉的到底是哪一部分

一个常见的半对说法是"FA 把 HBM 流量从 $O(S^2)$ 降到 $O(S\cdot D)$"。精确一点:

- **被消掉的**:$S\times S$ 分数矩阵的**写回 + 读取**——1.07 GB 对 100 KB 级
  shared memory,无论如何进不了片上,是纯粹不可缓存的 HBM 往返。
- **没有被消掉的**:K/V 的**重读**。每个 Q 行块都要把可见的那段 K/V 整条流过一遍,
  S=4096、BLOCK_M=128 时有 $S/\mathrm{BM}=32$ 个 Q 行块,causal 下平均各读一半:
  $$0.5 \times 32 \times \underbrace{(8 \times 4096 \times 128 \times 2\,\mathrm{B}) \times 2}_{K+V=16.8\,\mathrm{MB}} \approx 268\ \mathrm{MB}$$
  在 1008 GB/s 上约 $0.266\,\mathrm{ms}$,占实测 $1.1184\,\mathrm{ms}$ 的 24%;同形状
  的算力时间是 $137.4/165 = 0.83\,\mathrm{ms}$,占 74%。**compute-bound 成立,但不是
  因为"访存量降到 $O(S\cdot D)$",而是因为剩下的访存量正好被算力盖住。**

这条账还解释了 tile 扫描的主要结果:BLOCK_M 从 64 提到 128,4K 上
$1.362\to1.126\,\mathrm{ms}$(+17%,EXP-T01 §5,tile 扫描为终端级证据)。BM 翻倍
同时做了两件事:K/V 的重读**次数减半**,以及每块 softmax 簿记被更多 Q 行摊薄。
反方向的硬约束同样实测过:BLOCK_N 提到 128 直接 OOM(需求 160 KB 超过 Ada 每 CTA
的 shared memory 上限)。**tile 不是越大越好,片上资源是硬墙**——这堵墙在讲义 02
的 num_stages 上会以另一种形式再撞一次。

### 3.4 FA1 → FA2:循环反转到底省了什么

FA1 的循环是**外层 K/V、内层 Q**,而 $(m,l,\mathrm{acc})$ 是**按 Q 行**维护的,所以每
处理一个 K/V 块就要把所有 Q 行块的三元组从 HBM 读出、更新、写回。按本仓形状粗算
(BM=128、BN=64、S=4096、每行 $2+D=130$ 个 fp32):

$$\underbrace{\frac{S}{\mathrm{BN}}}_{64\ \text{个 K 块}} \times
\underbrace{\frac{S}{\mathrm{BM}}}_{32\ \text{个 Q 块}} \times
\underbrace{128 \times 130 \times 4\,\mathrm{B}}_{\text{一个 Q 块的三元组}}
\approx 136\ \mathrm{MB}\ (\text{每 } (b,h)\ \text{读写各一遍})$$

乘上 $B\cdot H_q=32$ 就是 GB 级——**与被消掉的 $S^2$ 物化同一个量级**。所以循环反转
不是锦上添花:FA1 消掉了 $S\times S$ 物化却在中间量上还回去一大半,FA2 把外层换成 Q、
内层换成 K/V,三元组从头到尾**只活在寄存器里**,这一半才真正消掉。(以上为按本仓形状
的推断算式,非实测;本仓无 FA1 实现。)

FA2 相对 FA1 的另外两处改动,在本仓 kernel 里都能指到行:

- **非 matmul FLOPs 削减**:FA1 每块做一次 $\mathrm{diag}(l)^{-1}$ 重缩放,FA2 只在
  循环外除一次(src/fa2_fwd.py:125)。为什么值钱:matmul 走 tensor core(165 TFLOPS
  量级),逐元素 fp32 除法走 SIMT 通路,单位 FLOP 贵一个量级——**省下的非 matmul 操作
  要按 tensor core 的加速比折算**才是它的真实价值。
- **并行度**:seq 维进 grid(src/fa2_fwd.py:155 的
  `grid = (triton.cdiv(S, block_m), B * Hq)`),4K 形状下 $32\times32=1024$ 个 program,
  对 128 个 SM 绰绰有余。记住这个数字——§3.6 里它会塌成 16。

### 3.5 GQA 头映射:一行寻址换来的一半带宽

GQA 让多个 q head 共享一个 kv head。kernel 里改的只有一行:
`hkv = hq // GQA_GROUP`(src/fa2_fwd.py:55)。

- **为什么可以这么写**:GQA 的定义就是**连续分组**共享(第 $g$ 组 q head 是
  $[gG,(g+1)G)$),整数除法正是那个映射。
- **收益在哪**:KV 的 HBM 读取量按 $H_{kv}/H_q$ 缩小(本仓 8/32 = 1/4),而且
  **不物化 repeat**——参考实现要 `repeat_interleave` 造一份 4 倍大的 KV
  (scripts/test_fa2.py:25-26),kernel 只换个索引读原张量。这份"免掉的拷贝"在 §5 的
  三臂口径里被单独标价。
- **改错会怎样**:写成 `hq % Hkv` 就把连续分组改成轮转分组——不 nan、不越界、性能
  一模一样,只是**悄悄算错**。正确性 gate 里的 `(1, 16, 4, 777, 128, True)`
  (scripts/test_fa2.py:41)就是为抓这类"安静的错误"设的格。

### 3.6 decode 的并行度塌陷:为什么必须换一根并行轴

把 §3.4 的 grid 代入 decode:$S_q=1$ 时第一维 $=\lceil 1/\mathrm{BM}\rceil=1$,program
总数塌成 $B\cdot H_q$。本仓 EXP-T04 协议 B=1、Hq=16(scripts/test_flash_decode.py:37),
就是 **16 个 program 面对 128 个 SM**,87.5% 的 SM 全程闲置——这不是调参能救的。

**split-K 的构造**:把 KV 切成 `num_splits` 段,grid 变成 `(num_splits, B*Hq)`。每段
独立跑与 FA2 完全相同的 online softmax,只是**不做最后那次除法**,而把
$(m_p,l_p,\mathrm{acc}_p)$ 写出去;第二个 kernel 用 §3.2 的合并算子一次归并所有段。

**splits 怎么选**(src/flash_decode.py:139-147):$\text{want} = 2\times128/(B H_q)$
(每 SM 至少 2 个 CTA 才有延迟切换余地),上界 $\lceil S_{kv}/\mathrm{BLOCK\_N}\rceil$
(段长不小于一个 BLOCK_N,再细切只剩空转),最后 `next_power_of_2`(combine 里
`tl.arange(0, NUM_SPLITS)` 要求编译期 2 的幂)。代进 32K 那格:want = 256/16 = 16,
上界 256,故 splits = 16、split_size = 2048。启发式未扫参(EXP-T04 §7)。

**为什么必须两个 kernel**:归并每一项都要乘 $e^{m_p-m_g}$,而 $m_g=\max_p m_p$ 要等
**所有段**算完才知道;atomicAdd 只能做无状态的可交换加法,做不了"先等全局 max 再逐项
换基"的非线性归约。两个 kernel 的边界就是那次跨 CTA 同步(CUDA 里 grid 级同步最便宜的
写法)。代价是**多一次 launch**,这笔账会在 §5 的短上下文那格原样出现。

## 4. 代码逐段走读:src/fa2_fwd.py 全核 + src/flash_decode.py 两段式

按执行顺序走读(引用为仓内真实代码逐字拷贝,标 文件:起-止行)。

**第 1 段 · 启动器:形状契约、tile 默认值与 grid**(src/fa2_fwd.py:143-155)

```python
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
    grid = (triton.cdiv(S, block_m), B * Hq)
```

角色:kernel 的全部外部约定都在这 13 行里定死。①**tile 按 dtype 分叉**:fp16 走
BM128/w8/s2(EXP-T01 扫描最优),fp32 降 BM32/w4/s1——fp32 的 tile 字节翻倍、BM128 撞
Ada 的 shared memory 上限;fp32 只是"校验路线",与 `IEEE_DOT` 配套(第 5 段)。
②**`assert Hq % Hkv == 0`** 是 GQA 连续分组映射(§3.5)的前提,不满足不是"算慢了"而是
"算错了",所以用 assert 而非 fallback。③改错会怎样:grid 只写 `(B*Hq,)` 就退化成
§3.6 的 decode 惨状。

**第 2 段 · program 定位与 GQA 一行映射**(src/fa2_fwd.py:51-56)

```python
    pid_m = tl.program_id(0)          # 第几个 Q 行块
    pid_bh = tl.program_id(1)         # batch*q_head 扁平索引(两维并行压一维,免 3D grid)
    b = pid_bh // NUM_Q_HEADS
    hq = pid_bh % NUM_Q_HEADS
    hkv = hq // GQA_GROUP             # GQA:多个 q head 共享一个 kv head,直接换算
                                      # 索引读原 KV,不物化 repeat(省 HBM 与显存)
```

角色:把线性 program id 翻译成 $(b,h_q,\text{M块})$ 三元坐标,顺手完成 GQA 头映射。
**这里没有任何原子操作与跨 program 通信**——每个 program 独占一个输出行块,这是循环
反转(§3.4)的结构性好处,也是三元组能常驻寄存器的前提。改错会怎样:`GQA_GROUP` 若
不是 `tl.constexpr`,整数除法从编译期常量折叠变成运行期指令,寻址开销进热路径。

**第 3 段 · Q tile 常驻 + 三个跑动量的不变量**(src/fa2_fwd.py:62-77)

```python
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
```

角色:建立主循环的**不变量**。注释里那三行不是装饰,是循环正确性的证明骨架——任意
时刻 $\mathrm{acc}/l_i$ 就是"只看前 $n$ 列"的精确输出,所以循环在任何一块之后停下都
自洽。三个决策:①**Q tile 只载一次**,它在整条 K/V 流里被复用 $S/\mathrm{BN}$ 次,是
算术强度的分子;②**全 fp32**,$l_i$ 是几千项求和,fp16 丢的位会直接乘进输出;
③**$m_i$ 初值 $-\infty$** 使首块 $\alpha=0$、"换基"退化成"直接赋值",正是 §3.2 单位元
在代码里的样子。改错会怎样:初值写 0 则首块分数全为负时 $l_i$ 偏小、输出整体错。

**第 4 段 · causal 上界与循环体:三种 mask 各司其职**(src/fa2_fwd.py:79-105)

```python
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
```

角色:这一段同时处理**算力**与**正确性**,且刻意用了三种不同的越界处理。`hi` 是算力
那件事:本块行号 $\in[\mathrm{pid}_m\mathrm{BM},(\mathrm{pid}_m+1)\mathrm{BM})$,
causal 下行 $m$ 只可见列 $n\le m$,故可见列的上确界是 $(\mathrm{pid}_m+1)\mathrm{BM}$
——对角线以下的整块**根本不进循环**,FLOPs 直接减半。这是"mask 不只是填 $-\infty$,
更是根本不算"的实现。

三种越界处理的分工必须分清:①K/V 的 `tl.load(..., other=0.0)` 补 0 只为让访存合法;
②`qk = tl.where(curr_n < n_ctx, qk, -inf)` 才是真正的剔除,**必须是 $-\infty$ 不是 0**
——$e^{0-m}$ 会给每个越界列贡献一份假质量进 $l_i$ 与 $\mathrm{acc}$;③causal 的逐元素
mask 无条件套用,不判"是不是对角块",谓词开销远小于一个控制流分支。改错会怎样:②
若省掉,S=777 这类非整除形状会**安静地算错**(多算 7 列假质量),正确性 gate 里
S=777 那格(scripts/test_fa2.py:41)就是为它设的。

**第 5 段 · online 更新的五行数学**(src/fa2_fwd.py:107-119)

```python
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
```

角色:§3.2 的合并算子在 kernel 里的样子,整篇讲义的核心五行。逐行对应:`m_new` =
$\max(m_A,m_B)$、`alpha` = 换基因子 $e^{m_A-m}$、`p` = 新块在新基准下的指数、`l_i` 与
`acc` 就是合并公式的两条。三个细节:①`alpha` 恒 $\le1$ 且 $m$ 单调不减,保证所有
`tl.exp` 的参数 $\le0$——**不会上溢是被结构保证的,不是运气**;②`p.to(v.dtype)` 把
概率降回 fp16 走 tensor core,精度损失由 gate 兜底,是明确接受的交易;③`IEEE_DOT`
关掉 TF32 走真 fp32,只在校验路线用。改错会怎样:`l_i` 与 `acc` 少乘一个 `alpha`,
输出不会 nan 只会偏,只有与 fp32 精算逐元素比对才抓得住。

**第 6 段 · 归一化与写回:nan 为什么不落地**(src/fa2_fwd.py:121-129)

```python
    # 面试点:归一化只做这一次(FA2 对 FA1 的关键改进——循环内维护未归一化
    # acc,省掉每块一次的除法/重缩放)。有效行 l_i 恒 >0(causal 下每行至少
    # 可见对角元自身);越界填充行全列被 mask → l_i=0 → 0/0=nan,但被下方
    # store 的行 mask 拦截,nan 不落地
    acc = acc / l_i[:, None]

    o_ptrs = (O + b * stride_ob + hq * stride_oh
              + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < n_ctx)
```

角色:FA2 相对 FA1 的"非 matmul FLOPs 削减"就落在这一次除法上(§3.4)。注释里那条 nan
推理链要能背下来:越界填充行全部列被 mask 成 $-\infty$ → $l_i=0$ → $0/0=\mathrm{nan}$,
但 store 的行 mask 保证这些行永不写出。**"内部允许出现 nan 但保证它出不去"是需要显式
论证的设计。** 改错会怎样:mask 漏掉则输出尾部出现 nan,且只在非整除形状上出现,常规
shape 的测试全绿。

**第 7 段 · flash-decoding partial kernel:同一套代数,标量版**(src/flash_decode.py:57-83)

```python
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
```

角色:split-K 的"分"。与第 4/5 段逐行对照即知:**这是同一套 online softmax,只是行数
从 BLOCK_M 变成 1**。三处刻意差异:①`hi = min(lo + split_size, n_ctx)` 把段边界与序列
边界一起兜住,`kmask = curr < hi` 一个条件管两件事;②用广播乘 + 行内规约替代
`tl.dot`——$S_q=1$ 时 tensor core 的 M 维要 pad 到 16、15/16 的算力全废,而 decode 本就
是带宽瓶颈,SIMT 反而干净;③全程 fp32,部分量要参与跨段换基。改错会怎样:`hi` 换回
`n_ctx` 则每段都扫完整条 KV,split-K 变成 splits 倍的重复劳动,**而结果依然正确**——
最难查的一类性能 bug。

**第 8 段 · combine kernel:块间归并与空段湮灭**(src/flash_decode.py:115-125)

```python
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
```

角色:split-K 的"合",§3.2 代数的第二次使用。注释里那段推导要能当场写出来:
$\sum_p l_p e^{m_p-m_g} = \sum_i e^{s_i-m_g}$ 是**全局行和**,
$\sum_p \mathrm{acc}_p e^{m_p-m_g} = \sum_i e^{s_i-m_g}v_i$ 是**全局未归一化输出**,
相除即精确结果,与单 pass 同源而非近似。两个工程细节:①`tl.exp(m - m_g)` 对空段自动
给 0,`next_power_of_2` 造出的空段不需特判(单位元性质在此兑现);②每个 $(b,h)$ 用
**一个 program 串行收全部段**而非树规约或原子——splits 至多几百,直读更快且**求和顺序
确定**、数值可复现。改错会怎样:改成 atomicAdd 则 $m_g$ 未知无法换基(§3.6);
`NUM_SPLITS` 传非 2 的幂则 `tl.arange` 编译期报错,launcher 的 `next_power_of_2`
(src/flash_decode.py:147)就是这条约束的兑付。

## 5. 实验数据怎么读

### 5.1 fig1:FA2 vs SDPA-flash

figures/fig1_fa2_vs_sdpa.png(脚本 scripts/plot_readme_figures.py:59-90)。

- **轴与口径**:横向条形图,x = 时延(ms,越短越好),y = 序列长
  $S\in\{512,1024,2048,4096\}$;两条 = 本仓 FA2 简化版 / torch SDPA(flash 后端);
  误差条 = **3 轮 std**(非轮内 100 次迭代的分布)。源数据 exp-t01_stability_3rounds.csv。
- **百分比标签的定义**:`eff = sdpa_ms / ours_ms * 100`(plot_readme_figures.py:64),
  是**效率**不是加速比;读反了会把"达到官方的 87%"说成"比官方慢 87%"。四格效率
  88/75/86/87% 非单调,不要挑数字讲(整表见 docs/theory/01_flashattention.md §3;
  512 那格为何不可当真见 §6 第 2 条)。
- **87% 的四要素**(本仓措辞约定,缺一不引):**简化版 / 仅 forward / 4K 形状
  (B1·H32/8·D128,fp16)/ 对照 = SDPA flash 后端**。3 轮口径 $1.1184\pm0.0015$ vs
  $0.9749\pm0.0024$ ms → **87.2%**;存盘单轮 1.119 vs 0.979 → 87.45%,**不进位**记
  87%(EXP-T01 §5)。
- **机理账(可以心算的那种)**:
  $$\mathrm{FLOPs} = \frac{4\,B H_q S^2 D}{2}\Big|_{\text{causal}} = 2\times1\times32\times4096^2\times128 = 137.4\ \mathrm{GFLOP}$$
  (口径就写在 scripts/test_fa2.py:71-73)。
  除以 1.1184 ms 得 **122.9 TFLOPS**(与 derived 的 122.867 对上);对 4090 fp16
  tensor core 峰值 165 TFLOPS(docs/theory/04 §2 口径)= **74%**,与 kperf 卡片的
  "算力 74%、occupancy 17%(regs 213)"逐点吻合(终端级证据,登记于 EXP-T06 §7)。
  SDPA 侧 141 TFLOPS = 85%。**所以"差 13%"的准确说法是算力利用率 74% vs 85%**,
  不来自算法差异(两边都是 FA2),而来自 tensor-core 布局微调、cp.async 双缓冲、
  warp 专业化这三层抽象税(docs/theory/01 §4)。
- **这个设计防了哪些坑**:①**正确性 gate 先于性能**——6 形状(MHA、GQA 2:1/4:1、
  S=777 非整除、双 head_dim、非 causal)max abs err ≤ 2e-3 全过,参考是 fp32 精算
  (scripts/test_fa2.py:22-31);②**对照物命名诚实**——显式
  `sdpa_kernel(SDPBackend.FLASH_ATTENTION)`(scripts/test_fa2.py:85-89)把后端钉死,
  否则 SDPA 可能落到 math 后端,那就成了和另一个算法比;③**naive fp32 臂**给出量尺,
  S>2048 时如实记 NaN 不补估计值;④**3 轮 std** 除 S=512 那格外全部 ≤0.3%,所以
  87.2% 与 87.45% 的差异是轮次噪声。

### 5.2 flash-decoding 的三臂表:2.24× 与 5.17× 差在哪

data/derived/exp-t04_stability_3rounds.csv(3 轮,协议 B1·Hq16/Hkv8·D128·bf16,
scripts/test_flash_decode.py:37):

| Skv | flash_decode (ms) | naive[repeat 预置] | 提速 | naive[含 repeat] | 提速 |
|---|---|---|---|---|---|
| 512 | 0.0907±0.0022 | 0.0779 | **0.86±0.02×** | 0.1243 | 1.37±0.01× |
| 2048 | 0.0941±0.0024 | 0.0816 | **0.87±0.01×** | 0.1271 | 1.35±0.02× |
| 8192 | 0.0918±0.0026 | 0.0804 | **0.88±0.01×** | 0.1726 | 1.88±0.05× |
| 32768 | 0.1515±0.0073 | 0.3393 | **2.24±0.11×** | 0.7822 | **5.17±0.24×** |

**为什么要拆三臂**:GQA 原生 kernel 免掉的正是 `repeat_interleave` 那份实体化拷贝
(scripts/test_flash_decode.py:16-20);算进对照物得 5.17×,不算进得 2.24×。**两个都对,
但报数字必须带口径**——这就是拆三臂而不是选一个好看的报的原因。

**机理账(把加速比算回字节比)**:三臂在 32K 那格全部带宽受限,加速比应当约等于搬运
字节比。逐臂列式(bf16,2 B/元素):

- flash_decode:只读未 repeat 的 KV,$2\times(8\times32768\times128)\times2\,\mathrm{B}
  = 134.2\,\mathrm{MB}$(中间量 Accp 仅 131 KB,可忽略);$/0.1515\,\mathrm{ms}
  = 886\ \mathrm{GB/s}$ = 1008 峰值的 **88%**。
- naive[repeat 预置]:读 16 头 KV = 268.4 MB + 物化分数矩阵往返约 4 MB
  $\approx 272.6\,\mathrm{MB}$;$/0.3393\,\mathrm{ms} = 803\ \mathrm{GB/s}$ = **80%**。
- naive[含 repeat]:再加 repeat 的写 268.4 MB + 读 134.2 MB,共 $\approx675\,\mathrm{MB}$;
  $/0.7822\,\mathrm{ms} = 863\ \mathrm{GB/s}$ = **86%**。

字节比 $272.6/134.2=2.03$(实测 2.24)、$675/134.2=5.03$(实测 5.17):**两个口径的差
就是 repeat 那份拷贝的读写**;拆开看,2× 来自 GQA 不 repeat($H_q/H_{kv}=2$),剩下的
2.5× 来自 repeat 本身。(字节数按协议推算、时间为 3 轮实测;算式与实测 10% 内的偏差
来自小张量 kernel 的效率差异,未单独隔离。)

**短上下文反亏 0.86-0.88× 的机理账**:fd 在 512/2048/8192 三格几乎不变(0.0907 /
0.0941 / 0.0918 ms),naive 同段也平(0.0779 / 0.0816 / 0.0804)。**两条线都平,说明
这一段的时间根本不在设备上**,都是主机侧地板:fd 每次要发 2 次 Triton launch
(partial + combine),naive 是 3 个 torch 算子但走 C++ 分发,而同机实测的分发成本是
Triton eager $36.2\pm0.1\,\mu s$ vs torch $8.03\pm0.74\,\mu s$(EXP-T05 3 轮)。
所以 0.86-0.88 是**两条地板之比,与 KV 长度无关**——不是算法输了,是这一档 launch
口径输了。正解在讲义 03:CUDA Graph 把 launch 塌缩 11.6×。

**另一组形状变体**(Qwen3-8B 形状 H32/fp16,3 轮,raw 在 data/raw/EXP-T04/):32K 上
$8.96\pm0.13\times$(记 9.0×),口径是 **repeat 内计**。头数 16→32、dtype 换 fp16,
数字就换一套——"数字必须带形状定语"的又一个实例。

## 6. 误区与边界

至少踩过一次才写得出来的错误直觉(第 3 条是本仓自己被证伪的假设):

1. **"FA 把 HBM 流量降到 $O(S\cdot D)$"**——半对。被消掉的是不可缓存的 $S\times S$
   写回;K/V 仍被 $S/\mathrm{BM}$ 个 Q 行块各读一遍(4K/BM128 下 32 遍、causal 折半约
   268 MB,§3.3)。正确的调优方向是"BM 越大重读越少",这才和实测的 BM 64→128 +17%
   对得上。
2. **"S=512 那格 88%,说明小序列上我们也接近官方"**——错。0.0390 ms 恰是同机实测的
   Triton 每调用 launch 地板($36.2\pm0.1\,\mu s$,EXP-T05),kernel 本体被盖住,
   **该点的 kernel 级差距在本仓协议下不可测**;要测就得先上 CUDA Graph 或改用设备侧
   计时。
3. **"flash-decoding 总比 naive 快"**——**被本仓自己的复测证伪**。EXP-T04 §5 原表里
   Skv ≤ 8192 的各行已全部作废(未存脚本的混合口径,不可复现),3 轮实测是
   **0.86-0.88× 的反亏**——它的正确定位是**长上下文武器**。方法论提炼:**旧数字不可
   复现时,作废它比解释它更诚实**;拆成三臂口径后,同一批数据同时给出了"反亏"与
   "5.17×"两个真相。
4. **"online softmax 是近似算法"**——不是。换基是精确恒等式(§3.2),误差只来自浮点
   舍入:flash_decode 在 Skv=512/2048 两格与 fp32 精算的 max abs err **恰为 0.0**
   (data/raw/EXP-T04/20260825T152434_flash_decode_stability_r1.json),32K 也只有 6.1e-5。
5. **"decode 用同一个 FA2 kernel 就行"**——不行。$S_q=1$ 时 M 维 tile 全废,grid 塌成
   $B\cdot H_q$(§3.6),且 `tl.dot` 要求 M ≥ 16、pad 之后 15/16 的 tensor core 算力
   空转。换并行轴不是优化,是换算法结构。

**适用边界**:全部数字来自单卡 RTX 4090、fp16/bf16、合成随机输入、**仅 forward**;
87% 不含 backward/dropout/alibi/paged,且只在 B1·H32/8·D128·S=4096 一个形状上成立;
flash-decoding 的 2.24×/5.17× 限 B1·H16/8·D128·bf16 协议的 32K 那一点,H32/fp16 变体
另有一套数字;splits 启发式未扫参(EXP-T04 §7);kperf 卡片是终端级证据(本容器无
性能计数器权限,docs/theory/04)。

## 7. 连环追问

1. **Q:softmax 减 max 是为了精度还是为了不溢出?**
   为了不溢出——减 max 在实数域上是**精确恒等式**(§3.1 第 2 步),顺带保证分母
   $\ge1$、不出现 $0/0$。fp16 的溢出线只有 $x>11.09$,真实模型的 attention logits
   轻易越过它,这不是理论洁癖。
2. **Q:分块之后为什么还能算对?**
   因为 $e^{s-m'}=e^{s-m}e^{m-m'}$,换基因子与求和下标无关、可提到求和号外;于是
   "旧部分量乘一个标量"就换到新基准,合并退化为普通加法(§3.2)。
3. **Q:为什么最后才除 $l$?**
   先除就丢了 $l_p$ 权重,合并要写成加权平均、$l_p$ 还得带着走(§3.2)——"最后再除"
   是可归并性的前提。代码落点 src/fa2_fwd.py:125。
4. **Q:FA2 相对 FA1 改了什么?**
   三处:①循环反转(外 Q 内 K/V)使三元组常驻寄存器,省掉 FA1 那笔与 $S^2$ 同量级的
   中间量往返(§3.4);②非 matmul FLOPs 削减;③seq 维进 grid,并行度从 $B\cdot H$
   涨到 $B\cdot H\cdot S/\mathrm{BM}$。
5. **Q:GQA 在 kernel 里改了几行?收益从哪来?**
   一行 `hkv = hq // GQA_GROUP`(src/fa2_fwd.py:55)。收益是 KV 读取量按 $H_{kv}/H_q$
   缩小且**不物化 repeat**——这份"免掉的拷贝"在 EXP-T04 里被单独标价(2.24× 与 5.17×
   的差就是它)。
6. **Q:你的 87% 具体差在哪一层?**
   算力利用率 74% vs 85%(§5.1 机理账),不是算法差异;缺口在 tensor-core 布局微调、
   cp.async 双缓冲、warp 专业化。本仓立场:讲得清的 87% 好过讲不清的 100%。
7. **Q:decode 为什么不能复用 FA2 kernel?**
   $S_q=1$ → grid 第一维 = 1 → program 数 $=B\cdot H_q$,本仓协议下 16 个对 128 SM
   (§3.6);且 `tl.dot` 的 M 维要 pad 到 16。必须换并行轴到 KV 维。
8. **Q:split-K 为什么要两个 kernel?能不能用 atomicAdd 省一次 launch?**
   不能。归并每项要乘 $e^{m_p-m_g}$,而 $m_g$ 要等所有段算完;atomicAdd 只能做无状态
   的可交换加法(§3.6)。两个 kernel 的边界就是这次 grid 级同步。splits 也不是越多
   越好:上界是 $\lceil S_{kv}/\mathrm{BLOCK\_N}\rceil$,再细切只剩空转,而且本仓的
   启发式**未扫参**(EXP-T04 §7)。
9. **压力问 Q:2.24× 和 5.17× 你到底该报哪个?会不会是挑了个好看的?**
   诚实答:两个都要报,并说清它们的差就是 `repeat_interleave` 的写+读(§5.2 的字节账
   把这点算死了)。若上游引擎的 attention 契约已把 repeat 好的 KV 传进来(本仓
   llm-engine 接线当时正是如此,EXP-T04 §6),GQA 原生那部分收益**根本兑现不了**,
   该报 2.24×;能控制契约时 5.17× 才真实可得。所以正确说法是"口径 A 下 2.24×、口径 B
   下 5.17×",而不是二选一。
10. **压力问 Q:87% 换个形状还成立吗?换成 backward 呢?**
    不能承诺。同一份代码在 S=1024 上只有 75%,S=512 那格的 88% 甚至不可测(§6 第 2 条)
    ——**87% 是 4K 那一个点的值**,四要素限定就是为此存在。backward 更是另一件事:要
    重算 P(存不下),且 dQ 与 dK/dV 的归约方向不同(行向 vs 列向)。本仓只做 forward
    并如实声明,不外推。

## 8. 工业对照与延伸

与生产实现的差距,各在哪一层:

- **官方 FlashAttention(CUDA)**:算法同构,差在实现层——tensor-core 的 swizzle/布局
  微调、cp.async 双缓冲、warp 专业化(producer/consumer 分工)。本仓把这三层交给
  Triton 编译器,代价就是 §5.1 里那段算力利用率差。
- **vLLM paged attention**:本 kernel 的数学 + block table 间接寻址。本仓 K/V 指针是
  连续 stride 寻址(src/fa2_fwd.py:87-90),paged 版换成"先查页表再算页内偏移",
  数学一行不改。
- **vLLM / SGLang 的 decode kernel**:paged 读 + split 归并的合体,本仓 flash_decode
  只做后半。两者正交:paged 解决"KV 在哪",flash-decoding 解决"并行度从哪来"。
- **引擎接线这一层的坑**:kernel 快不等于引擎快。本仓接进 llm-engine 时首版对 KV cache
  切片做了 `.contiguous()`,每层每步整拷一遍 KV,TPOT 反而变差(EXP-T04 §6)——
  **隐藏拷贝藏在调用约定里。**

延伸阅读(源码/文档锚):

1. src/fa2_fwd.py:107-125 —— online 更新五行 + 唯一一次归一化,整篇讲义的核心。
2. src/flash_decode.py:115-123 —— 块间归并代数与空段湮灭的注释推导。
3. docs/theory/01_flashattention.md §2 第 3 步(FA1→FA2 三处改动)与 §3(完整效率表)。
4. docs/talk/whiteboard_card_fa2_algebra.md —— 白板推导卡,含"为什么不能每块先除"。
5. records/EXP-T04_flash_decoding.md §5-§7 —— 三臂口径拆分与小 Skv 行作废的完整过程。
