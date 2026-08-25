# 讲义 02 · GEMM 的流水线:从访存墙推到 num_stages,再推到 FP8 per-block

> 读者:准备校招面试的作者本人,以及第一次给 GEMM 调 tile/流水的工程师。
> 读法:不跳步。每个论断后面跟着它的证据锚(EXP 编号 / 文件:行号 / raw 路径),
> 所有数字与仓内现行口径逐字一致,来源见 records/ 与 data/derived/。

## 1. 这一篇回答什么问题

"GEMM 要做双缓冲"这句话在教科书里是结论,在本仓是一个**被实测打了折的假设**:
同一个 kernel 只改 `num_stages`,2 级(经典双缓冲)只带来 +1%,3 级才 +20%。
读完你应当能:

- 手推**访存墙与 tile 复用账**:为什么不分块的 GEMM 只能吃到峰值的千分之几、
  $\mathrm{BM}\times\mathrm{BN}$ 的 tile 把算术强度抬到多少、以及为什么"抬到够用"
  这件事**单靠寄存器和 shared memory 做不到**,必须把 L2 和调度顺序也算进来。
- 解释 `num_stages=N` 在 Triton 里到底生成了什么、它的 shared memory 代价怎么算、
  以及"最优深度"为什么会随形状在 3 与 4 之间摇摆(本仓明确不宣称唯一最优深度)。
- 说清"打平 cuBLAS"的完整口径:**限两测形状 fp16、cuBLAS = torch.matmul dispatch
  (cuBLASLt)**;square4k 3 轮 159.4±1.2 vs 160.0±0.7 TFLOPS(打平),8B up_proj
  形状单轮 154.4 vs 147.3(反超 4.8%)——以及为什么这个"反超"在 3 轮口径下要
  说得更保守。
- 讲清 FP8 per-block 的缩放代数怎么和 GEMM 主循环对齐,以及 **228.1±1.3 TFLOPS
  是预量化孤立 GEMM 的口径,不是端到端推理提速**(在线量化端到端只有 72.9)。

## 2. 直觉与第一性原理

**没有分块的 GEMM 会怎样**:$C = AB$,朴素写法是每个输出元素读 $A$ 的一行和 $B$
的一列。每个输出 $2K$ 次 FLOP、读 $2K$ 个元素($4K$ 字节,fp16),算术强度
$$I_{\text{naive}} = \frac{2K}{4K} = 0.5\ \text{FLOP/Byte}$$
4090 的机器平衡点是 $165\,\mathrm{TFLOPS} / 1008\,\mathrm{GB/s} \approx 164$
FLOP/Byte。$0.5 \ll 164$,意味着算力只能吃到峰值的 $0.5/164 \approx 0.3\%$——
**GEMM 的第一堵墙从来不是算力,是访存。**

**tile 把强度抬起来**:改成每个 CTA 算一块 $\mathrm{BM}\times\mathrm{BN}$ 的 $C$,
沿 K 维流式读入。这块 $C$ 需要 $\mathrm{BM}\times K$ 的 A 与 $K\times\mathrm{BN}$
的 B,做 $2\cdot\mathrm{BM}\cdot\mathrm{BN}\cdot K$ 次 FLOP:
$$I_{\text{tile}} = \frac{2\,\mathrm{BM}\,\mathrm{BN}\,K}{2(\mathrm{BM}+\mathrm{BN})K}
= \frac{\mathrm{BM}\cdot\mathrm{BN}}{\mathrm{BM}+\mathrm{BN}}$$
本仓默认 $128\times128$ 给出 **64 FLOP/Byte**。注意:64 仍然小于机器平衡点 164。
反推一下"要靠 HBM 单独吃满算力需要多大的方 tile":$b/2 \ge 164 \Rightarrow b\ge328$,
而仅累加器就要 $328^2\times4\,\mathrm{B} \approx 430\,\mathrm{KB}$ 每 CTA——**寄存器
和 shared memory 根本装不下**。所以结论是:**tile 只能把强度抬到"够 L2 接手"的
程度,剩下的靠缓存局部性与调度顺序补**(§3.1 的 grouped 调度就是补这一段)。

**日常类比与失效点**:tile 像"把仓库里的一批货一次搬到工位旁的小推车上,做完这批
再换"。类比在两处失效:①小推车的容量(shared memory / 寄存器)不是可以无限加大的
自由变量,它和"同时能开工几条产线"(occupancy)是同一份预算——本仓 GEMM 的
occupancy 只有 17%(regs 170/线程,kperf 终端级证据,登记于 EXP-T06 §7),正是为
tile 付的钱;②类比里"搬货"和"干活"天然可以同时进行,GPU 上却必须**显式**安排
(双缓冲 / cp.async / 软件流水),否则计算单元就是干等——这正是 §3.2 的主题。

## 3. 完整推导与机制

### 3.1 grouped 调度:先把 L2 复用距离压下来

流水线不是第一条腿。先看**同一时刻在跑的那些 CTA 到底在读什么**:朴素的 row-major
线性映射下,相邻 pid 沿 N 方向铺开,于是相邻 CTA 各读各的 B 列块;等到 N 方向绕完
一圈回到同一批 B 列块时,中间已经流过 $\mathrm{num\_pid\_n}$ 个块的数据,L2 早被冲掉。

grouped 调度(src/gemm_pipelined.py:41-53)把线性 pid 重映射成"先在 M 方向排满
$\mathrm{GROUP\_M}$ 行、再换 N 列":同一组内的 CTA 命中同一批 B 列块,**B tile 的
L2 复用距离从 $\mathrm{num\_pid\_n}$ 缩到 $\mathrm{GROUP\_M}$**(本仓 = 8)。
它和流水线正交,一起构成"打平 cuBLAS"的两条腿(docs/theory/02 §2)。

用 §2 的算式核一下这条腿有多重要:tile 模型给出的 HBM 流量上界是
$$2KMN\left(\frac{1}{\mathrm{BM}}+\frac{1}{\mathrm{BN}}\right)
= 2\cdot4096^3\cdot\frac{2}{128} \approx 2.15\ \mathrm{GB}$$
在 1008 GB/s 上要 $2.13\,\mathrm{ms}$,而 stages=3 的实测总时长只有
$0.862\,\mathrm{ms}$(存盘 raw)。**实测比"每个 tile 都从 HBM 拿"的上界快 2.5 倍,
说明大部分重读根本没走到 HBM。** 4096² fp16 的 A、B 各 33.5 MB,合计 67 MB,在
AD102 数十 MB 量级的 L2 里基本装得下——这既解释了 grouped 调度为什么值钱,也提醒
你:**这个形状的数字自带"操作数进得了 L2"这个前提**,换成权重远大于 L2 的形状,
访存账要重算(推断,本仓未测该形状族)。

### 3.2 为什么 2 级双缓冲只 +1%:从数据反推延迟/计算比

主循环每轮做两件事:把一个 $\mathrm{BLOCK\_K}$ 条带从 global 搬到片上,以及用
tensor core 做一轮 `tl.dot`。串行执行时 tensor core 在等数;重叠的收益上限是
$$\text{省下的时间} \le \min(T_{\text{搬运}},\ T_{\text{计算}})$$
教科书默认 $T_{\text{搬运}} \ll T_{\text{计算}}$,于是"两块缓冲交替"就够把搬运整段
藏进计算——这就是经典双缓冲的适用前提。

**本仓的数据否证了这个默认前提**(EXP-T02,4096³ fp16,存盘单轮值):

| num_stages | 1(无重叠) | 2(双缓冲) | 3 | 4 |
|---|---|---|---|---|
| TFLOPS | 131.9 | 133.5 | **160.5** | 157.1 |

1→2 只 +1%,2→3 跳 +20%,整段流水化相对无重叠是 +21%。**反推**:如果搬运远短于
计算,2 级就该吃满收益;实测 2 级几乎白给,说明 Ada 上一次 BLOCK_K 搬运的时长
**不短于**一轮 dot,两级流水只藏住其中一段,必须再加一级才把 load 整个摘出关键路径
(推断:从数据反推机制,本容器无性能计数器可直接验证 stall 归因,docs/theory/04)。

顺带算一下这一轮到底搬多少:每 CTA 每轮载入
$(\mathrm{BM}+\mathrm{BN})\times\mathrm{BLOCK\_K}\times2\,\mathrm{B}
= 256\times64\times2 = 32\ \mathrm{KB}$。全 kernel 1024 个 CTA × 64 轮 × 32 KB
= 2.15 GB,在 0.862 ms 内完成 → 聚合 2.5 TB/s,远超 HBM 峰值——**再次证明这些搬运
主要由 L2 供给**(与 §3.1 同一结论的另一种算法)。

### 3.3 深度的代价:shared memory 与 CTA 并发是同一份预算

`num_stages=N` 的代价是**片上缓冲 $\propto N$**:每级要一份 tile 的 shared memory,
按上式即 32 KB/级。于是

- $N=2$:64 KB;$N=3$:96 KB,已经逼近 Ada 每 CTA 约 100 KB 的 shared memory 上限;
- $N=4$:名义 128 KB 已经超限。

这解释了两件事:①**最优深度必然停在 3/4 附近而不是继续加深**——不是收益没了,是
装不下了;②square4k 上 stages=4 反而略低于 3(3 轮 158.0±3.52 vs 159.4±1.23,注意
它的 std 是四档里最大的一档),而 qwen8b 形状最优在 4。所以本仓的措辞是
**"最优 stage 在 3/4 间随形状摇摆,不宣称唯一最优深度"**(EXP-T02 §6)。

同一堵墙在讲义 01 里出现过一次:FA2 把 BLOCK_N 提到 128 时 shared memory 需求
160 KB 直接 OOM(EXP-T01 §5)。**tile 大小、流水深度、occupancy 三者共用一份片上
预算**,任何一项加码都在别处扣钱——本仓 GEMM 的 occupancy 17%(regs 170/线程)
却打出 98% 峰值算力(kperf,终端级证据,EXP-T06 §7),就是这笔交易划算的证据:
tensor core kernel 靠**寄存器堆 ILP**藏延迟,比靠"多 warp 轮转"更值钱。
"occupancy 低"只在带宽 % 与算力 % **两个都低**时才是嫌疑人(docs/theory/04 §2)。

### 3.4 num_stages 的语义:CUDA 手写双缓冲在 Triton 里是什么

CUDA 手写版的形状是:两块 shared memory 缓冲 + `cp.async` 预取下一块 +
`__pipeline_wait_prior` 等待 + 计算当前块 + 交换(docs/theory/02 §2 有伪码)。
Triton 版是**同一个朴素循环** `load → dot → 指针步进`,由编译器的 software
pipelining pass 做三件事:①分配 $N$ 份 tile 的 shared memory;②把 `tl.load` 提前
$N-1$ 轮发射(走 cp.async,DMA 直写 shared,不占计算发射槽);③插入等待屏障。

这带来一个本仓反复强调的方法论:**"双缓冲带来多少"从一句口号变成了一个可测数字**
——同一份 kernel 源码、同一组输入,只改一个 launch 参数,四档扫下来就是 §3.2 的表。
能这样做实验,是因为 Triton 把"数据流"和"怎么排流水"分开了;代价是你**控不了**
具体指令与 bank conflict,真出问题时只能靠变体对照而不是读汇编(docs/theory/03)。

### 3.5 FP8 per-block:缩放代数怎么和主循环对齐

fp8 e4m3 只有 3 位尾数、满量程 ±448,per-tensor 一个 scale 会被离群值拖垮。
DeepGEMM 的做法是**细粒度缩放**,本仓原样搬到 sm_89:

- 权重 $B\ (K,N)$:每个 $128\times128$ 块一个 scale,$s^B_{k_g,n_g} = \max|B_{\text{blk}}|/448$;
- 激活 $A\ (M,K)$:每行每 128 长的 K 组一个 scale,$s^A_{m,k_g} = \max|A_{m,k_g}|/448$。

反量化怎么融进累加(这是面试的题眼):
$$C_{mn} = \sum_{k_g} s^A_{m,k_g}\, s^B_{k_g,n_g}
\left(\sum_{k\in k_g} \hat A_{mk}\hat B_{kn}\right)$$
**为什么两个 scale 可以提到内层求和之外**:组内 scale 是常量,标量乘法对求和可分配。
前提是 $\mathrm{BLOCK\_K}$ 与缩放组**硬对齐**(都取 128):一轮主循环恰好覆盖一个
scale 组,内层那个 $\sum_{k\in k_g}$ 就是一次干净的 fp8 `tl.dot`。若错位,scale 要
逐元素进 dot,tensor core 路径直接废掉(src/fp8_gemm.py:73-77 的注释即此)。
$\mathrm{BLOCK\_N}=128$ 与权重块对齐则让 $s^B$ 退化成**一个标量**(否则要向量化)。

代价也要说清:每组结果必须先乘 scale 才能并入总累加器,所以**不能**写
`tl.dot(a, b, acc)`(把累加器直接交给 mma 指令),每组多出一条独立的 FMA 链。
这是 per-block 相对 per-tensor 的固有代价,也正是 Hopper 的 wgmma 用**原生 scale 槽**
替你省掉的那部分——**同一套缩放代数,两代指令**:

| | sm_89(本实现) | sm_90(DeepGEMM 本体) |
|---|---|---|
| 矩阵指令 | mma.sync(同步) | wgmma(异步,warpgroup) |
| 搬运 | cp.async | TMA(张量批搬运) |
| 细粒度缩放 | 累加器侧手乘 | wgmma 原生 scale 槽 |

"DeepGEMM 为什么是 Hopper-only"的答案就在后两行(docs/theory/06 §2)。

## 4. 代码逐段走读:src/gemm_pipelined.py 与 src/fp8_gemm.py

按执行顺序走读(引用为仓内真实代码逐字拷贝,标 文件:起-止行)。

**第 1 段 · grouped 调度:把 L2 复用距离缩到 GROUP_M**(src/gemm_pipelined.py:41-59)

```python
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
```

角色:§3.1 那条腿的全部实现。`group_size_m` 那行 `tl.minimum` 是最容易写错的地方:
M 方向最后一组可能不足 $\mathrm{GROUP\_M}$ 行,组内映射必须按**实际行数**取模,否则
残组的 pid 会算出越界的 `pid_m`。三个连锁选择:①grid 是**一维**的
(src/gemm_pipelined.py:103),重映射完全在 kernel 内做,launcher 不需要知道调度策略;
②`GROUP_M` 是 `tl.constexpr`,除法与取模在编译期折叠成移位/乘法;③指针一次算好、
循环里只做 `+= BLOCK_K * stride`,把地址运算移出热路径。改错会怎样:去掉 grouped
映射(直接 `pid_m = pid // num_pid_n`)不会错,只会在大 N 时让 B 反复走 HBM——
是一个**只在性能上显形**的错误。

**第 2 段 · 主循环:一个朴素循环 + 一个编译器旋钮**(src/gemm_pipelined.py:61-83)

```python
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
```

角色:整篇讲义的核心。这个循环体和"无流水"版本**一个字都不差**——差别全在
`num_stages`(第 3 段的 launch 参数),编译器据此分配 $N$ 份 smem 并把 load 提前
$N-1$ 轮发射(§3.4)。三个细节:①**K 尾块用 mask 兜底**(`offs_k + k0 < K`),
补 0 对 dot 零贡献,于是主循环外不需要单写一个收尾循环——代价是每轮多两个谓词;
②`acc = tl.dot(a, b, acc)` 把累加器作为第三参数,**直接映射 mma 指令的累加寄存器**,
省掉一趟独立的加法(对比第 7 段的 fp8 版:那里因为要先乘 scale,**必须**退回
`acc += ...`);③`IEEE_DOT` 分支关掉 TF32,只在 fp32 校验路线用。
改错会怎样:把 `acc` 从第三参数拆成 `acc += tl.dot(a, b)`,数值一样、性能掉一截,
而且这个损失在 profile 里表现为"更多 FFMA 指令",不看汇编很难归因。

**第 3 段 · 启动器:默认值即实验结论**(src/gemm_pipelined.py:86-98)

```python
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
```

角色:把 EXP-T02 的扫描结论固化成默认值。`num_stages` 默认 3 就是 §3.2 那张表的
最优档;注释里"仅 acc 就占 $128\cdot128$ fp32 / 256 线程 = 64 regs/线程,kperf 实测
总 170"是把 §3.3 的预算账写进代码——**默认参数必须能指回它的来源实验**,否则半年后
没人敢动。fp32 路线降 `block_n` 到 64 的理由同 FA2:tile 字节翻倍,片上装不下。
改错会怎样:把 `num_stages` 默认改成 2("因为教科书说双缓冲"),这个形状上直接丢
20% 算力,而正确性测试全绿。

**第 4 段 · linear:小 M 时 tile 必须缩**(src/gemm_pipelined.py:113-123)

```python
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
```

角色:把 GEMM 接进推理引擎的那一层,也是"训练形状的直觉在 decode 上会翻车"的实例。
$\mathrm{BLOCK\_M}=128$ 在 $M=1$(decode 单 token)时,mma 的行利用率是 1/128,
**127/128 的 tensor core 算力全废**;所以按实际行数降到 32 或 16。这条和讲义 01 §3.6
是同一个病根的两种药:decode 的 M 维天然只有 1,attention 那边靠换并行轴解决,
GEMM 这边靠缩 tile 缓解。改错会怎样:不缩 tile,decode 阶段的 linear 会慢到让整个
kernel 级加速在端到端上归零——**"kernel 快 ≠ 引擎快"的又一个入口**。

**第 5 段 · FP8 权重量化:块级 absmax 怎么算**(src/fp8_gemm.py:38-49)

```python
def quant_fp8_block(w: torch.Tensor):
    """(K,N) fp16/bf16 → fp8 e4m3 + scale (K/128, N/128) fp32。
    权重离线量化一次即可复用——serving 里这一步不进热路径。"""
    K, N = w.shape
    # 整除断言是缩放组代数的前提,也让 GEMM 主循环省掉 K 维 mask
    assert K % GROUP == 0 and N % GROUP == 0
    # 4D view 把 128×128 块折出维度,absmax 一次 amax 完成
    wf = w.float().reshape(K // GROUP, GROUP, N // GROUP, GROUP)
    amax = wf.abs().amax(dim=(1, 3)).clamp(min=1e-8)       # (K/g, N/g)
    scale = amax / FP8_MAX             # absmax 顶到 448 → e4m3 满量程利用
    q = (wf / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX)
    return q.reshape(K, N).to(torch.float8_e4m3fn), scale
```

角色:§3.5 缩放布局的构造端。`reshape(K//G, G, N//G, G)` 把 $128\times128$ 块折成
两个维度,`amax(dim=(1,3))` 一次求出每块的绝对最大值——**用 view 而不是循环**是这段
唯一的性能要点。`scale = amax / 448` 让块内最大值恰好顶到 e4m3 满量程,把有限的 3 位
尾数全用在有效动态范围上;`clamp(min=1e-8)` 防全零块除 0。整除断言不只是防御:它同时
是**缩放组代数的前提**(§3.5)与"GEMM 主循环可以零 K 维 mask"的依据。
改错会怎样:把 absmax 换成 per-tensor 一个 scale,kernel 一行不用改、速度一样,只是
量化误差从 3.6e-2 量级劣化到被离群值主导——**精度回归不会让任何性能测试变红**。

**第 6 段 · BLOCK_K 与缩放组硬对齐**(src/fp8_gemm.py:73-77)

```python
    # 面试点:BLOCK_K 与缩放组硬对齐——一轮主循环恰好覆盖一个 scale 组,
    # 组内 scale 是常量,反量化才能从 dot 里提出来变成秩 1 修正
    # (sa 行向量 × sb 标量);若 BLOCK_K 与组错位,scale 要逐元素进 dot,
    # tensor core 路径直接废掉
    BLOCK_K: tl.constexpr = 128            # 与缩放组硬对齐
```

角色:一行 `constexpr` 承载 §3.5 的全部前提。把 `BLOCK_K` 写死在 kernel 内(而不是
开放成参数)是刻意的:它必须等于缩放组长度,开放出去就等于把一个**正确性前提**降级成
调优旋钮。改错会怎样:BLOCK_K 取 64,一轮主循环只覆盖半个 scale 组,反量化就不能提到
dot 之外——要么逐元素乘 scale(tensor core 路径废掉),要么算错。

**第 7 段 · fp8 主循环:二级累加**(src/fp8_gemm.py:96-115)

```python
    # 主循环按缩放组推进(kg = 第几个 K 组);K 维无 mask——量化函数已断言
    # K % 128 == 0,热循环因此零分支
    for kg in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0.0)
        # 注意不能写 tl.dot(a, b, acc):本组结果要先乘 scale 才能并入总累加,
        # 每组多出一条独立 FMA 链——这是 per-block 缩放相对 per-tensor 的
        # 固有代价,也是 Hopper wgmma 原生 scale 槽替你省掉的那部分
        part = tl.dot(a, b)                                  # fp8→fp32
        sa = tl.load(SA + offs_m * stride_sam + kg * stride_sak,
                     mask=offs_m < M, other=0.0)             # (BLOCK_M,)
        # BLOCK_N=128 恰好整块落在同一权重列块内 → sb 是单标量
        # (pid_n*BLOCK_N//128 即列块号);解耦 BLOCK_N 需 sb 向量化(backlog)
        sb = tl.load(SB + kg * stride_sbk
                     + (pid_n * BLOCK_N // 128) * stride_sbn)  # 标量
        # 二级累加(DeepGEMM 的 accumulator promotion 在 mma 世代的形态):
        # 组内 fp8 dot 出 fp32,组间乘 scale 后并入主累加器
        acc += part * sa[:, None] * sb
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
```

角色:§3.5 公式在 kernel 里的落点,也是与第 2 段最值得对读的一段。差异只有一处但很
致命:这里**不能**写 `tl.dot(a, b, acc)`,因为本组结果要先乘 $s^A s^B$ 才能并入总
累加器,于是每组多一条独立 FMA 链——**这就是"per-block 缩放的固有代价",也是
Hopper wgmma 原生 scale 槽省掉的那部分**(§3.5)。另外两点:①K 维**零 mask**,因为
量化函数已断言 $K \bmod 128 = 0$,热循环因此没有分支;②`sb` 靠 `BLOCK_N=128` 与权重块
对齐退化成单标量,解耦 BLOCK_N 需要 sb 向量化(EXP-T06 §7 的开放项)。
改错会怎样:`sa` 的 `mask=offs_m < M, other=0.0` 若写成 `other=1.0`,越界行会被乘上
无意义的 scale,而这些行本来就要被末尾的 store mask 丢掉——不会错,但会让人误以为
`other` 值无关紧要;真正不能动的是 `sb` 的块号算式 `pid_n * BLOCK_N // 128`。

## 5. 实验数据怎么读

### 5.1 fig2:num_stages 扫描与 cuBLAS 对照

figures/fig2_gemm_stages.png(脚本 scripts/plot_readme_figures.py:94-118)。

- **轴与口径**:横向条形,x = TFLOPS(4096³ fp16,越高越好),y 自上而下 =
  cuBLAS(torch.matmul)/ Triton stages=4/3/2/1;误差条 = **3 轮 std**;源数据
  data/derived/exp-t02_stability_3rounds.csv。标题即结论句(单图单结论)。
- **单变量设计**:四档 stages 用的是**同一个 kernel、同一组输入、同一次进程**
  (scripts/test_ew_gemm.py:90-93 的循环),唯一变量就是 launch 参数。所以四档之间的
  差可以直接归因给流水深度,不需要额外的控制实验——这是"把双缓冲的贡献变成可测数字"
  的实验形态。
- **对照物命名诚实**:cuBLAS 这一行指的是 `torch.matmul`(fp16 走 cuBLASLt),
  脚本里写作 `bench(lambda: a @ b)`(scripts/test_ew_gemm.py:89),记录与 措辞约定
  表都注明 **cuBLAS = torch.matmul dispatch**。它不是直接调 cuBLAS API 的结果,含
  torch 的分发开销——对 0.86 ms 量级的大 GEMM,分发那几微秒可忽略,但口径要写出来。
- **"打平"的准确说法**:square4k 3 轮 **159.4±1.2 vs 160.0±0.7 TFLOPS**,差值落在
  误差条内,所以说"打平(差 0.4% 内,单轮 160.5 vs 159.8)";**不能**说"超过 cuBLAS"。
- **"反超 4.8%"的准确说法**:那是 Qwen3-8B up_proj 形状(2048×4096×12288)的**单轮
  存盘值** 154.4(stages=4)vs 147.3。3 轮口径下是 155.3±0.9 vs 150.4±4.5,幅度收窄到
  +3.3%,且 cuBLAS 侧的轮间 std 达 2.98%(四个数字里波动最大的一个)。**诚实的读法是:
  这一格从"打平"到"小幅反超"都在数据支持范围内,引用 4.8% 时必须带"单轮存盘 raw"**
  (措辞约定:只引存盘 raw 轮)。
- **机理账**:$2MNK/t$。square4k 的 $2\cdot4096^3 = 137.4\ \mathrm{GFLOP}$,
  $/0.8566\,\mathrm{ms} = 160.4\ \mathrm{TFLOPS}$(与存盘的 160.5 对上);对 165 TFLOPS
  峰值是 97%,与 kperf 卡片"算力 98%、occupancy 17%(regs 170)"吻合(终端级证据,
  EXP-T06 §7)。**到顶了**——这也是为什么本仓不再往下扫 BM/BN/BK 全空间:剩余空间
  不足 3%,而 tile 全扫的代价远大于收益(EXP-T02 §7 如实列为未做项)。
- **正确性同批出**:两形状相对误差 ~7e-4(fp16 输入 fp32 累加 vs torch.matmul,
  data/derived/exp-t02_stability_3rounds.csv 的 correctness 行)。性能表和正确性表
  出自**同一次运行**,不存在"快的那版和对的那版不是同一个"。

### 5.2 FP8 的四口径表(EXP-T06)

| 口径 | square4k | 8B up_proj | 说明 |
|---|---|---|---|
| fp8 预量化孤立 GEMM | 228.6±0.1 | **228.1±1.3** | 3 轮;单轮存盘 227.7 / 235.7 |
| fp16 cuBLAS | 156.9±2.4 | 154.9±0.1 | 3 轮;单轮 152.0 / 155.2 |
| fp8 在线量化端到端 | **72.9** | 64.3 | 量化 kernel 是瓶颈,仅示成本 |
| fp16 Triton | 154.3±4.3 | 157.2±1.2 | 同 §5.1 的 kernel |

**怎么读这张表**:第 1 行除以第 2 行就是那个 **1.5×**——3 轮口径算出来是 1.46-1.47×,
单轮存盘口径是 227.7/152.0 = 1.50× 与 235.7/155.2 = 1.52×,同一档,措辞约定统一记 1.5×;
**引用时定语比小数点更要紧:预量化孤立 GEMM vs fp16 cuBLAS,非端到端推理提速**。
第 3 行是同一份代码把量化也计进去的口径,只有 72.9/64.3:**收益被 torch 侧的量化
kernel 吃光了**。
这不是失败,是把"为什么真实 serving 要权重离线量化 + 激活量化融合进上游算子"
证明了一遍。两个精度数字也要分层报:kernel 精确性 1.9e-4(vs 逐块反量化的 fp32 精算,
证明 kernel 没算错)、量化本体误差 3.6e-2 / 3.9e-2 相对(vs 原 fp16,是量化范式自身的
代价,与 kernel 无关)。

**理论 2× 为什么只吃到 1.5×**:缩放乘法与 fp32 累加占 SIMT 算力(§3.5 的额外 FMA
链)、fp8 mma 的峰值配比本身打折;kperf 定界为 compute-bound、约 70% fp8 峰值
(终端级证据,NCU 不可用)。这个缺口是**指令世代的架构税**,不是实现没写好——
换到 Hopper 的 wgmma + TMA 才有另一套上限(§3.5 的表)。

## 6. 误区与边界

1. **"双缓冲是 GEMM 的标配收益"**——本仓实测把它打了折:2 级只 +1%,3 级才 +20%
   (§3.2)。教科书的"双缓冲够用"隐含了 $T_{\text{搬运}}\ll T_{\text{计算}}$ 的前提,
   Ada 上这个形状不满足。**流水深度要按延迟/计算比配,不是按口号配。**
2. **"occupancy 低说明 kernel 没写好"**——本仓 GEMM occupancy 17%(regs 170/线程)
   却打出 98% 峰值算力。tensor core kernel 靠寄存器堆 ILP 藏延迟,占用率只在**带宽 %
   与算力 % 两个都低**时才是嫌疑人(§3.3,docs/theory/04 §2)。
3. **"num_stages 越大越好"**——每级 32 KB smem,3 级已逼近 Ada 每 CTA 上限,4 级名义
   超限;实测最优在 3/4 之间**随形状摇摆**(square4k 为 3、qwen8b 为 4),本仓明确
   不宣称唯一最优深度(§3.3)。
4. **"打平 cuBLAS 可以简写成打平"**——不行。完整口径是**限两测形状 fp16、cuBLAS =
   torch.matmul dispatch(cuBLASLt)、未做全形状扫描**;而且"反超 4.8%"是单轮存盘值,
   3 轮口径收窄到 +3.3% 且对照侧 std 近 3%(§5.1)。
5. **"FP8 给推理提速 1.5×"**——最容易被误引的一句。1.5× 是**预量化孤立 GEMM** 的口径;
   同一份代码把在线量化计进去只有 72.9 TFLOPS(§5.2)。把 1.5× 说成端到端提速,是本仓
   措辞约定明令禁止的措辞。
6. **"这套账换个形状照样成立"**——不一定。§3.1 的实测比 HBM 上界快 2.5 倍,前提是
   4096² 的两个操作数合计 67 MB、基本进得了 L2;权重远大于 L2 的形状要重算访存账
   (推断,本仓未测)。

**适用边界**:全部数字来自单卡 RTX 4090、fp16 输入 fp32 累加、两个测过的形状
(4096³ 与 2048×4096×12288);tile 空间未全扫;stall 归因无性能计数器支持,机制结论
标注为从数据反推;FP8 部分限 e4m3、BLOCK_N 硬绑 128、激活量化未融合进上游算子
(EXP-T06 §7 的开放项)。

## 7. 连环追问

1. **Q:GEMM 为什么一定要分块?**
   算术强度。不分块是 0.5 FLOP/Byte,机器平衡点 164,只能吃到峰值 0.3%(§2)。
   分块把强度抬到 $\mathrm{BM}\cdot\mathrm{BN}/(\mathrm{BM}+\mathrm{BN})$,
   $128\times128$ 给 64 FLOP/Byte。
2. **Q:64 还是小于 164,为什么还能打到 98% 峰值?**
   因为剩下的重读大部分命中 L2 而不是 HBM——tile 模型的 HBM 上界要 2.13 ms,实测
   0.862 ms(§3.1)。所以 tile 的任务只是"抬到够 L2 接手",grouped 调度负责把复用
   距离压到 GROUP_M。
3. **Q:`num_stages=2` 和 CUDA 手写双缓冲是一回事吗?**
   是同一件事的两种写法:两份 smem 缓冲 + 提前一轮的 cp.async + 等待屏障。区别是
   Triton 由编译器生成,你只给一个数字(§3.4);代价是控不了具体指令与 bank conflict。
4. **Q:那为什么你的双缓冲只 +1%?**
   重叠收益上限是 $\min(T_{\text{搬运}}, T_{\text{计算}})$;本仓数据反推出 Ada 上一次
   BLOCK_K 搬运不短于一轮 dot,2 级只藏住一段,3 级才填平气泡(§3.2)。
5. **Q:那为什么不一直加深?**
   smem $\propto N$,每级 32 KB;3 级 96 KB 已逼近 Ada 每 CTA 上限,4 级名义超限
   (§3.3)。深度、tile 大小、occupancy 共用一份片上预算。
6. **Q:occupancy 只有 17%,不该先修这个吗?**
   不该。带宽 % 与算力 % 只要有一个贴顶就说明延迟已经藏住了;本仓算力 98%,
   occupancy 低是**为大 tile 付的钱**,是设计不是缺陷(§3.3)。
7. **Q:FP8 的两个 scale 为什么能提到 dot 外面?**
   组内 scale 是常量,乘法对求和可分配(§3.5 的公式)。前提是 BLOCK_K 与缩放组硬对齐
   (都是 128),BLOCK_N 与权重块对齐使 $s^B$ 退化成标量。
8. **Q:为什么 fp8 版不能用 `tl.dot(a, b, acc)`?**
   因为本组结果要先乘 $s^As^B$ 才能并入总累加器,累加器不能直接交给 mma 指令
   (第 7 段)。每组多一条 FMA 链,这就是 per-block 缩放的固有代价。
9. **Q:DeepGEMM 为什么是 Hopper-only?**
   wgmma(异步 warpgroup 矩阵指令,自带 scale 槽)+ TMA(张量批搬运)是 sm_90 才有的;
   sm_89 只有 mma.sync + cp.async,缩放只能在累加器侧手乘(§3.5 的表)。缩放**代数**
   可以搬,**指令世代**搬不了。
10. **压力问 Q:你说"打平 cuBLAS",是不是挑了对自己有利的形状?**
    诚实答:是**只在两个形状上测过**,而且这两个形状(4096³ 与 8B up_proj)都属于
    "K 维长、操作数进得了 L2"的舒适区,没有做全形状扫描,也没测小 M/瘦长/非对齐形状
    (EXP-T02 §7 明写)。cuBLAS 的优势恰恰在于**形状覆盖面**——它对每个形状族都有
    调好的 kernel,而本仓只有一套 tile。所以正确的说法是"在这两个形状上打平/略胜",
    "打平 cuBLAS"作为一般性结论**不成立**。
11. **压力问 Q:228 TFLOPS 的 FP8,能给推理带来 1.5 倍吗?**
    不能。228 是**预量化孤立 GEMM**;同一份代码在线量化端到端只有 72.9(§5.2)。要在
    真实 serving 里兑现,得满足三个条件:权重离线量化(本仓 `quant_fp8_block` 正是
    "离线一次"的定位)、激活量化融合进上游算子(本仓**未做**,EXP-T06 §7)、以及模型
    本身对 3.6e-2 量级的量化误差可接受(本仓只测了误差,没测下游任务指标)。三条缺
    一条,1.5× 就落不了地。

## 8. 工业对照与延伸

- **cuBLAS / CUTLASS**:同样的 tile + 流水思想,但它们按形状族预置了几十套 kernel 与
  启发式选择器,并做了 split-K、stream-K 等本仓没有的负载均衡策略。本仓是"一套 tile
  打两个形状",差距在**覆盖面**而不是单点峰值(§7 第 10 问)。
- **Hopper 的 wgmma + TMA**:把"预取"从 cp.async 的显式流水升级成硬件描述符驱动的
  批搬运,异步矩阵指令自带流水语义。这是 DeepGEMM 能做到本仓做不到那部分的硬件根源
  (§3.5 的表,docs/theory/06 §2)。
- **推理引擎里的 GEMM**:真实 serving 的瓶颈形状是 decode 的 $M=1$(第 4 段的 linear
  自适应),以及"量化 + GEMM + epilogue"的融合边界——本仓把量化留在 torch 侧,这正是
  §5.2 那个 72.9 的来源;生产实现会把激活量化融进上一个算子的 epilogue。
- **与 vLLM 的 capability 分派互为表里**:上游按 SM 版本分派 fp8 快路径(90/100 跳过
  89),本仓从"为什么 89 走不了那条路"的角度给出同一事实的另一面(docs/theory/06 §2)。

延伸阅读(源码/文档锚):

1. src/gemm_pipelined.py:61-79 —— 主循环与 num_stages 的流水语义注释。
2. src/fp8_gemm.py:96-115 —— 缩放组主循环与二级累加(与上一条对读)。
3. docs/theory/02_double_buffering.md §2 —— CUDA 手写双缓冲伪码与 Triton 版的对照。
4. records/EXP-T02_gemm_pipeline.md §5-§7 —— 存盘轮口径、首轮数字作废的处理,以及
   "最优 stage 随形状摇摆"的结论边界。
