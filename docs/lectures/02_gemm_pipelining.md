# 讲义 02 · GEMM 的流水线:从访存墙推到 num_stages,再推到 FP8 per-block

> 读者:准备校招面试的作者本人,以及第一次给 GEMM 调 tile/流水的工程师。
> 读法:不跳步。每个论断后面跟着它的证据锚(EXP 编号 / 文件:行号 / raw 路径),
> 所有数字与仓内现行口径逐字一致,来源见 records/ 与 data/derived/。
> 引用规范:凡属论文/官方文档的论断,给出标题 + arXiv/DOI 编号 + 章节编号
> (文档给 URL 路径 + 小节名);凡属本讲义补出的推导或折算,行内标注"本讲义推导";
> 无法用检索确认的说法标注"未核实"。

## 1. 这一篇回答什么问题

"GEMM 要做双缓冲"这句话在教科书里是结论,在本仓是一个**被实测打了折的假设**:
同一个 kernel 只改 `num_stages`,2 级只带来 +1%,3 级才 +20%。而这一篇的核心增量是:
**用编译期资源探针把"2 级到底是不是双缓冲"这个前提问题查清楚了**——在本机的
Triton 3.6 上,缓冲份数 = $\max(1, \text{num\_stages}-1)$,所以 `num_stages=2`
**根本没有开出第二份缓冲**。"双缓冲只值 1%"这个读法是错的;正确读法是
"那一档还不是双缓冲,真双缓冲是 stages=3,它值 +20%"(EXP-T08《num_stages 与 shared memory 份数的映射》)。

读完你应当能:

- 手推**访存墙与 tile 复用账**:为什么不分块的 GEMM 只能吃到峰值的千分之几、
  $\mathrm{BM}\times\mathrm{BN}$ 的 tile 把算术强度抬到多少、以及为什么"抬到够用"
  这件事**单靠寄存器和 shared memory 做不到**,必须把 L2 和调度顺序也算进来。
- 说清 `num_stages=N` 在 Triton 里到底生成了什么:文档承诺的是什么、编译产物实测
  出来的是什么、两者为什么差 1、以及这个差 1 如何反过来解释 EXP-T02《流水线 GEMM》那张表。
- 用 shared memory / 寄存器 / 每 SM CTA 数三条预算,算出"深度到几级会撞墙"、
  "为什么加深到 4 级并不额外压占用率"(与本仓旧说法相反,见 §3.4.3)。
- 说清"打平 cuBLAS"的完整口径:**限两测形状 fp16、cuBLAS = torch.matmul dispatch
  (cuBLASLt)**;square4k 3 轮 159.4±1.2 vs 160.0±0.7 TFLOPS(打平),8B up_proj
  形状单轮 154.4 vs 147.3(反超 4.8%)——以及为什么这个"反超"在 3 轮口径下要
  说得更保守。
- 讲清 FP8 per-block 的缩放代数怎么和 GEMM 主循环对齐,**228.1±1.3 TFLOPS 是预量化
  孤立 GEMM 的口径,不是端到端推理提速**(在线量化端到端只有 72.9);并说清
  "Hopper 有原生 scale 槽"这句流行说法**不成立**,硬件块缩放是 Blackwell 才有的
  (§3.6.4)。

### 1.1 本篇要建立的五条能力

1. **三层算术强度**:同一个 GEMM 有三个不同的算术强度(朴素 / 每 CTA tile /
   整问题),它们分别对应"完全不缓存""只有片上缓存""缓存层级全用上"三种世界;
   看到一个 GEMM 数字,先判断它在哪一层。
2. **编译产物阅读**:知道怎么不跑 bench 就把 shared memory 用量、寄存器数、
   spill 数读出来(EXP-T08 的探针),并用它反推流水结构。
3. **资源预算**:能同时算寄存器预算与 shared memory 预算,判断哪一条先卡住,
   并知道"卡住的那条"决定了改哪个参数才有效。
4. **量化代数**:能推导 per-block 缩放为什么可以提到 dot 之外,以及这个"可以"
   依赖哪两条对齐条件;能说清它相对 per-tensor 的固有代价有多大(可算)。
5. **指令世代**:能准确说出 sm_89 / sm_90 / sm_100 在矩阵指令、搬运指令、块缩放
   三件事上的分界,并且**不把 Blackwell 的能力安到 Hopper 头上**。

### 1.2 符号与口径约定

| 符号 | 含义 | 本仓取值 |
|---|---|---|
| M, N, K | GEMM 三维 | square4k:4096³;qwen8b up_proj:2048×4096×12288 |
| BM / BN / BK | tile 三维 | 128 / 128 / 64(fp16 路线) |
| GROUP_M | grouped 调度的组高 | 8 |
| $N_s$ | `num_stages` | 扫 1..4 |
| $\beta$ | 缓冲份数(编译器实际分配) | $\max(1, N_s-1)$(EXP-T08 实测) |
| $\pi_{\text{math}}$ | fp16 tensor core / fp32 累加峰值 | 165.2 TFLOPS(非稀疏) |
| $\pi_{\text{math}}^{\text{fp8}}$ | fp8 tensor core / fp32 累加峰值 | 330.3 TFLOPS(非稀疏) |
| $\pi_{\text{math}}^{\text{fp32}}$ | 非 Tensor 的 FP32 峰值 | 82.6 TFLOPS |
| $\pi_{\text{mem}}$ | HBM 带宽 | 1008 GB/s |
| ops:byte | 机器平衡点 | 165.2e12/1008e9 ≈ 164 FLOP/B |

硬件常数出处:NVIDIA, "NVIDIA Ada GPU Architecture" 白皮书 Appendix A Table 2
(GeForce RTX 4090:SMs 128、GPU Boost Clock 2520 MHz、Memory Bandwidth
1008 GB/sec、L2 Cache Size 73728 KB、Register File Size 32768 KB、
L1 Data Cache/Shared Memory 16384 KB、Peak FP32 TFLOPS (non-Tensor) 82.6、
Peak FP16 Tensor TFLOPS with FP32 Accumulate 165.2/330.4、
Peak FP8 Tensor TFLOPS with FP32 Accumulate 330.3/660.6,脚注 2 注明第二个数是
"Effective TOPS / TFLOPS using the new Sparsity Feature"、脚注 4 注明 FP8/FP32
累加那一格在白皮书 v2.02 里被改成了 "proper number"——**厂商规格表也会被修订,
引用要带版本意识**)。占用率上限出自 Ada Tuning Guide §1.4.1.1:
每 SM 最多 48 warps、64K 个 32 位寄存器、每线程最多 255 个寄存器、每 SM 最多
24 个 thread block、shared memory 每 SM 100 KB、**每 thread block 上限 99 KB**。

### 1.3 本篇引用的一级文献(详细出处见 §8.4)

- Triton 语言与编译器:Tillet, Kung, Cox, "Triton: An Intermediate Language and
  Compiler for Tiled Neural Network Computations", MAPL '19,
  DOI:10.1145/3315508.3329973,§5.1.1 Pre-Fetching、§5.2 Machine-Dependent Passes。
- num_stages 的官方语义:Triton 文档 `triton.Config` 与 `tl.range`(本机 3.6.0
  源码 `runtime/autotuner.py`、`language/core.py`)。
- 异步搬运语义:NVIDIA PTX ISA §9.7.9.26 Asynchronous copy(cp.async /
  commit_group / wait_group)。
- 矩阵指令语义:PTX ISA §9.7.15(mma,warp 级集合)、§9.7.16(wgmma,warpgroup 级
  异步);CUTLASS 的 wgmma 说明。
- FP8 细粒度缩放:DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv:2412.19437,
  §3.3(Fine-Grained Quantization 与 Increasing Accumulation Precision);
  DeepGEMM 仓库 README。
- 性能模型:NVIDIA, "GPU Performance Background User's Guide" §4;
  "Matrix Multiplication Background User's Guide"(Arithmetic Intensity、
  §3.1 Tile Quantization、§3.2 Wave Quantization)。

## 2. 直觉与第一性原理

### 2.1 三层算术强度:同一个 GEMM,三个数

**第一层,朴素**:$C = AB$,每个输出元素读 $A$ 的一行和 $B$ 的一列。每个输出
$2K$ 次 FLOP、读 $2K$ 个元素($4K$ 字节,fp16),算术强度
$$I_{\text{naive}} = \frac{2K}{4K} = 0.5\ \text{FLOP/Byte}$$
4090 的机器平衡点是 $165.2\,\mathrm{TFLOPS} / 1008\,\mathrm{GB/s} \approx 164$
FLOP/Byte(ops:byte 的定义见 NVIDIA GPU Performance Background User's Guide §4:
"the ratio of a processor's math and memory bandwidths")。$0.5 \ll 164$,意味着
算力只能吃到峰值的 $0.5/164 \approx 0.3\%$——**GEMM 的第一堵墙从来不是算力,
是访存。**

**第二层,每 CTA 的 tile**:改成每个 CTA 算一块 $\mathrm{BM}\times\mathrm{BN}$ 的
$C$,沿 K 维流式读入。这块 $C$ 需要 $\mathrm{BM}\times K$ 的 A 与
$K\times\mathrm{BN}$ 的 B,做 $2\cdot\mathrm{BM}\cdot\mathrm{BN}\cdot K$ 次 FLOP:
$$I_{\text{tile}} = \frac{2\,\mathrm{BM}\,\mathrm{BN}\,K}{2(\mathrm{BM}+\mathrm{BN})K}
= \frac{\mathrm{BM}\cdot\mathrm{BN}}{\mathrm{BM}+\mathrm{BN}}$$
本仓默认 $128\times128$ 给出 **64 FLOP/Byte**。注意:64 仍然小于机器平衡点 164。

**第三层,整个问题**:NVIDIA 的矩阵乘指南给的公式是
$M\cdot N\cdot K / (M\cdot K + N\cdot K + M\cdot N)$("Matrix Multiplication
Background User's Guide","Arithmetic Intensity" 小节)。代入 square4k:
$$I_{\text{problem}} = \frac{4096^3}{3\times4096^2} = \frac{4096}{3}
\approx 1365\ \text{FLOP/Byte}$$
(这个公式的分子是 MAC 数、分母是元素数;换算成 FLOP/Byte 时分子乘 2、分母乘 2 B,
比值不变——**这是一个恰好与数据类型无关的量,fp16 与 fp32 给同一个数**,本讲义推导。)

**三层放在一起就是全部故事**:

| 层 | 强度 | 对 164 的关系 | 含义 |
|---|---|---|---|
| 朴素 | 0.5 | ≪ | 完全不可能 |
| 每 CTA tile | 64 | < | 若每个 tile 都真的从 HBM 拿,仍然带宽受限 |
| 整问题 | 1365 | ≫ | 若缓存层级把重读全接住,算力受限 |

实测落在第三层(§3.1 的账):**tile 的任务不是把强度抬到 164 以上,而是抬到
"L2 能接手"的程度**。反推一下"要靠 HBM 单独吃满算力需要多大的方 tile":
$b/2 \ge 164 \Rightarrow b\ge328$,而仅累加器就要
$328^2\times4\,\mathrm{B} \approx 430\,\mathrm{KB}$ 每 CTA——对照 Ada 每 block
99 KB shared、每 SM 64K 寄存器,**寄存器和 shared memory 根本装不下**。所以
"靠 tile 单独解决问题"这条路在硬件上是封死的,不是没人想到。

### 2.2 重叠为什么必须显式安排

主循环每轮做两件事:把一个 $\mathrm{BLOCK\_K}$ 条带从 global 搬到片上,以及用
tensor core 做一轮 `tl.dot`。GPU 不会自动把这两件事重叠——它靠的是**多 warp 轮转**
来藏延迟,而 tensor core kernel 的 occupancy 通常很低(本仓 17%,§3.4.4),
warp 不够多,轮转藏不住。所以必须换一种藏法:**同一个 warp 内,让"搬下一块"
的指令先发出去,不等它回来就去算当前块**——这就是软件流水,硬件基础是 `cp.async`
(SM80+)。

**日常类比与失效点**:tile 像"把仓库里的一批货一次搬到工位旁的小推车上,做完这批
再换"。类比在两处失效:①小推车的容量(shared memory / 寄存器)不是可以无限加大的
自由变量,它和"同时能开工几条产线"(occupancy)是同一份预算;②类比里"搬货"和
"干活"天然可以同时进行,GPU 上却必须**显式**安排(双缓冲 / cp.async / 软件流水),
否则计算单元就是干等。

### 2.3 三条贯穿全篇的公理

- **公理 A(参数的语义要实测,不能只读文档)**:`num_stages=2` 在文档里叫
  "2 stages in flight",在本机编译产物里是"1 份缓冲"。**参数名不是语义,编译产物
  才是**(§3.3)。
- **公理 B(先找出卡住的那条预算)**:寄存器与 shared memory 会给出不同的
  CTA/SM 上限,只有更紧的那条起作用;改另一条参数不会有任何效果(§3.4)。
- **公理 C(代数可以跨代际搬,指令不能)**:DeepGEMM 的缩放代数在 Ada 上一行不改
  就能落地;它依赖的 wgmma/TMA 一条也搬不过来(§3.6)。

## 3. 完整推导与机制

### 3.1 grouped 调度:先把 L2 复用距离压下来

流水线不是第一条腿。先看**同一时刻在跑的那些 CTA 到底在读什么**:朴素的 row-major
线性映射下,相邻 pid 沿 N 方向铺开,于是相邻 CTA 各读各的 B 列块;等到 N 方向绕完
一圈回到同一批 B 列块时,中间已经流过 $\mathrm{num\_pid\_n}$ 个块的数据,L2 早被冲掉。

grouped 调度(src/gemm_pipelined.py:41-53)把线性 pid 重映射成"先在 M 方向排满
$\mathrm{GROUP\_M}$ 行、再换 N 列":同一组内的 CTA 命中同一批 B 列块,**B tile 的
L2 复用距离从 $\mathrm{num\_pid\_n}$ 缩到 $\mathrm{GROUP\_M}$**(本仓 = 8)。
它和流水线正交,一起构成"打平 cuBLAS"的两条腿(docs/theory/02 §2)。

#### 3.1.1 三个流量口径的夹逼(本讲义推导)

用 §2.1 的三层强度各算一遍 square4k 的 HBM 流量,得到一组夹逼:

| 口径 | 字节 | 折算时间(1008 GB/s) | 说明 |
|---|---|---|---|
| **下界**:每个矩阵只读/写一次 | $(MK+NK+MN)\times2\,\mathrm{B} = 3\times4096^2\times2 = 100.7\ \mathrm{MB}$ | 0.100 ms | compulsory traffic |
| **上界**:每个 tile 都从 HBM 拿 | $2KMN(1/\mathrm{BM}+1/\mathrm{BN}) \times 2\,\mathrm{B}$ 折合 **2.15 GB** | 2.13 ms | 即 §2.1 第二层 |
| 算力时间 | $137.4\ \mathrm{GFLOP} / 165.2\ \mathrm{TFLOPS}$ | **0.832 ms** | roofline 的另一条边 |
| **实测**(stages=3,存盘 raw) | — | **0.8566 ms** | 160.5 TFLOPS |

**实测比上界快 2.5 倍,而只比算力下界慢 3%。** 两条结论直接掉出来:

1. 大部分重读根本没走到 HBM。4096² fp16 的 A、B 各 33.55 MB,合计 67.1 MB;
   RTX 4090 的 L2 是 73728 KB(Ada 白皮书 Table 2),**两个操作数整体装得进 L2**。
   这既解释了 grouped 调度为什么值钱,也提醒你:**这个形状的数字自带"操作数进得了
   L2"这个前提**,换成权重远大于 L2 的形状,访存账要重算(推断,本仓未测该形状族)。
2. 实测 0.8566 ms 对算力下界 0.832 ms 的比值是 1.029,即**这个 kernel 已经吃到
   roofline 算力边的 97%**,与 kperf 卡片"算力 98%"逐点吻合(终端级证据,登记于
   EXP-T06《FP8 GEMM》§7)。剩余空间不足 3%,这是本仓不再往下扫 BM/BN/BK 全空间的定量理由
   (EXP-T02 §7 如实列为未做项)。

上界那个 2.15 GB 还可以换一种算法核对(与上表独立):每 CTA 每轮载入
$(\mathrm{BM}+\mathrm{BN})\times\mathrm{BK}\times2\,\mathrm{B} = 256\times64\times2
= 32\ \mathrm{KB}$;CTA 数 $= (4096/128)^2 = 1024$,轮数 $= K/\mathrm{BK} = 64$;
$1024\times64\times32\,\mathrm{KB} = 2.15\ \mathrm{GB}$。在 0.8566 ms 内完成 →
**聚合 2.5 TB/s**,是 HBM 峰值的 2.5 倍。这个 2.5 TB/s 只能由 L2 供给;
本仓**没有测过 4090 的 L2 实际带宽**,所以"2.5 TB/s 在 L2 能力之内"这句标为未核实,
只能说"它超过 HBM 峰值,故必然主要来自片上"。

#### 3.1.2 波量化:两个测试形状都恰好整波(本讲义推导)

NVIDIA 的矩阵乘指南把"一波"定义为同时执行的最大 thread block 数,并指出
tile 总数不是波数整数倍时会出现"tail wave"("Matrix Multiplication Background
User's Guide" §3.2 Wave Quantization)。把本仓两个形状代进去:

| 形状 | CTA 数 | 每 SM 可驻留 CTA(§3.4.4) | 一波 | 波数 |
|---|---|---|---|---|
| square4k | $32\times32=1024$ | 1 | 128 | **8.0** |
| qwen8b up_proj | $16\times96=1536$ | 1 | 128 | **12.0** |

**两个形状都恰好是整数波**——这是"打平 cuBLAS"这个结果的一个未被明说的前提。
换一个 CTA 数不整除 128 的形状(例如 M=4096、N=4224 给 $32\times33=1056$ CTA,
8.25 波),尾波只用 25% 的 SM,理论上要多付近一波的时间。**本仓没有测过非整波形状**,
所以这条是推断;但它是 §7 第 10 问"是不是挑了舒适区形状"的一个具体答案:
**是的,而且现在能说出舒适在哪一条上**。

### 3.2 为什么 2 级"双缓冲"只 +1%:先别急着解释,先查前提

主循环每轮做两件事(§2.2)。串行执行时 tensor core 在等数;重叠的收益上限是
$$\text{省下的时间} \le \min(T_{\text{搬运}},\ T_{\text{计算}})$$
教科书默认 $T_{\text{搬运}} \ll T_{\text{计算}}$,于是"两块缓冲交替"就够把搬运整段
藏进计算——这就是经典双缓冲的适用前提。

**本仓的数据看上去否证了这个默认前提**(EXP-T02,4096³ fp16):

| num_stages | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 存盘单轮 TFLOPS | 131.9 | 133.5 | **160.5** | 157.1 |
| 3 轮 mean±std | 132.0±1.06 | 133.7±0.58 | **159.4±1.23** | 158.0±3.52 |

(3 轮数据:data/derived/exp-t02_stability_3rounds.csv。)

1→2 只 +1%,2→3 跳 +20%。**旧版本的讲义在这里直接给了机制解释**——"Ada 上一次
BLOCK_K 搬运的时长不短于一轮 dot,两级流水只藏住其中一段"。这个解释听起来合理,
数据也不反对它。**但它跳过了一个前提问题:`num_stages=2` 真的开出了两份缓冲吗?**

#### 3.2.1 先做显著性检验,再谈机制(本讲义推导)

用 3 轮 mean/std 做一次最粗的判别(把轮间 std 当作误差,合并标准差
$\sigma = \sqrt{\sigma_1^2+\sigma_2^2}$):

| 对比 | 差值(TFLOPS) | 合并 σ | 差值 / σ | 判读 |
|---|---|---|---|---|
| stages 1→2 | +1.73 | 1.21 | **1.4** | 不显著 |
| stages 2→3 | +25.7 | 1.36 | **18.9** | 极显著 |
| stages 3→4 | −1.37 | 3.73 | **0.37** | 不显著 |

**1→2 那一档在 3 轮口径下根本不构成显著差异**。也就是说,数据本身就在暗示
"stages=1 与 stages=2 是同一种配置",而不是"双缓冲收益很小"。这个暗示需要一个
独立证据来确认——那就是 EXP-T08。

### 3.3 num_stages 的真实语义:文档说什么,编译产物说什么

#### 3.3.1 文档承诺的是"在飞的迭代数"

Triton 的官方描述有两处,本机 3.6.0 源码里可以逐字核对:

- `triton.Config` 的 `num_stages`(runtime/autotuner.py 的 docstring):
  "the number of stages that the compiler should use when software-pipelining
  loops. Mostly useful for matrix multiplication workloads on SM80+ GPUs."
- `tl.range` 的 `num_stages`(language/core.py 的 docstring):
  "pipeline the loop into this many stages (so there are `num_stages` iterations
  of the loop in flight at once)."同一段还提醒两者不同:
  "The kernel argument only pipelines loads that feed into `dot` operations,
  while this attribute tries to pipeline most (though not all) loads in this loop."

**两条都没有说"分配 N 份 shared memory 缓冲"。** 文档承诺的量是"同时在飞的迭代数",
而"缓冲份数"是另一个量。本仓过去的讲义与源码注释(src/gemm_pipelined.py:7-8、
:61-64)把这两个量当成了一个,写成"分配 N 份 smem 缓冲"——**这是本篇要纠正的第一处
自家表述**。

#### 3.3.2 编译期资源探针:不跑 bench 就把份数读出来

EXP-T08 的做法是编译一次、读 `CompiledKernel.metadata.shared`,完全不产生时延数字。
**假设在跑之前锁定**:若 `num_stages=N` 分配 N 份缓冲,则
$\text{smem}(N) = N \times 32\,\mathrm{KB}$;**证伪条件**:若
$\text{smem}(2) = \text{smem}(1)$,假设被证伪。

单份 tile 的字节数先算清楚:$(\mathrm{BM}+\mathrm{BN})\times\mathrm{BK}\times2\,\mathrm{B}
= (128+128)\times64\times2 = 32768\,\mathrm{B} = 32\,\mathrm{KB}$。

实测(EXP-T08 §5,GEMM `_gemm_kernel`,BM128/BN128/BK64/w8):

| num_stages | metadata.shared | 份数 = shared / 32 KB | n_regs | n_spills |
|---|---|---|---|---|
| 1 | 32768 (32 KB) | 1 | 162 | 0 |
| 2 | 32768 (32 KB) | **1** | 136 | 0 |
| 3 | 65536 (64 KB) | 2 | 170 | 0 |
| 4 | 98304 (96 KB) | 3 | 178 | 0 |

**假设被证伪。** 实测映射是
$$\beta = \max(1,\ N_s - 1)$$
同一映射在 FA2 kernel 上也成立(§3.3.4),六格逐格吻合——**两个结构完全不同的
kernel 给出同一条映射,这不是巧合**。

#### 3.3.3 差 1 是怎么来的:流水线的正常结构,不是文档错

把 §3.3.1 的文档语义与 §3.3.2 的实测放在一起,差 1 有一个干净的解释
(本讲义推导,机制层面):

一条 $N_s$ 级软件流水在稳态时,确实有 $N_s$ 个迭代"在飞",但它们处在不同阶段:
**其中一个迭代正在被 `tl.dot` 消费**,其余 $N_s-1$ 个是"已发出 `cp.async`、
尚未被消费"的预取。被消费的那一份数据在寄存器/mma 操作数里,不需要额外占一份
**预取缓冲**。所以

$$\text{在飞迭代数} = N_s,\qquad \text{预取缓冲份数} = N_s - 1$$

代入 $N_s=2$:在飞 2 个迭代,预取缓冲 1 份——**这恰好就是"没有双缓冲"**:
你只有一块 shared memory,`cp.async` 写它、`tl.dot` 读它,两者必须串行。
真正的"两块缓冲交替"要 $\beta=2$,即 $N_s=3$。

$\max(1,\cdot)$ 那个下限也解释得通:$N_s=1$ 表示不做流水,但仍然要有一块地方
放 tile,所以至少 1 份。

PTX 层的对应语义(NVIDIA PTX ISA §9.7.9.26)支持这个图景:
`cp.async.commit_group` "commits all prior uncommitted cp.async instructions into
a cp.async-group";`cp.async.wait_group N` "wait till only N or fewer of the most
recent cp.async-groups are pending and all the prior cp.async-groups committed by
the executing threads are complete"。**等待的粒度是"组"**,每轮把这一轮的预取
commit 成一组,然后 `wait_group(β-1)` 允许 $\beta-1$ 个组继续在飞——组数与缓冲
份数一一对应。**但本仓没有 dump TTGIR/PTX 去数组数**(EXP-T08 §7 明列为开放项),
所以"组数 = 份数"这条标为**未核实**;确定的只有 `metadata.shared` 这一层。

#### 3.3.4 同一映射在 FA2 kernel 上复现,并解释掉那个 160 KB

EXP-T08 同时探了 src/fa2_fwd.py 的 `_fa2_fwd_kernel`(bench 形状
B1·H32/8·S4096·D128 fp16,BM=128)。公式是

$$\text{shared} = \underbrace{\mathrm{BM}\times D\times2\,\mathrm{B}}_{Q\ \text{tile 常驻} = 32\ \mathrm{KB}}
+ \beta \times \underbrace{\mathrm{BN}\times D\times2\,\mathrm{B}\times2}_{K,V\ \text{各一份}}$$

| BLOCK_N | num_stages | 公式值 | 实测 |
|---|---|---|---|
| 32 | 2 | 32 + 1×16 = 48 KB | 49152 |
| 32 | 3 | 32 + 2×16 = 64 KB | 65536 |
| 64 | 2 | 32 + 1×32 = 64 KB | 65536 |
| 64 | 3 | 32 + 2×32 = 96 KB | 98304 |
| 128 | 2 | 32 + 1×64 = 96 KB | 98304 |
| 128 | 3 | 32 + 2×64 = 160 KB | **`OutOfResources: Required 163840, Hardware limit 101376`** |

两条结论:

1. **EXP-T01《Triton FA2 forward》那个"BN=128 撞 shared memory 上限(160 KB)OOM"被逐字复现**:
   163840 B = 160 KB,而 101376 B = 99 KB,正是 Ada Tuning Guide §1.4.1.1 的
   "The maximum shared memory per thread block is 99 KB"。**官方文档给的上限与
   编译器抛出的数字完全对上,这是本仓唯一一条严格闭合的"文档 → 实测"链。**
2. **OOM 只在 stages=3 出现**;BN=128 + stages=2 能编译(96 KB),但寄存器打到
   255/线程(Ada 每线程上限,同节)。EXP-T01 的 tile 扫描是终端级证据、未记录
   stages 列,所以"当年那次 OOM 在哪一档"无法回溯;**能确定的是那个字节数与
   那堵墙**,以及"撞哪堵墙取决于另一个参数"这条更一般的教训。

**还有一条对讲义 01 的回溯影响**:FA2 的 fp16 默认档是 `num_stages=2`
(src/fa2_fwd.py:149),按本映射它只有 **1 份 K/V 缓冲**——也就是说
**本仓的 FA2 达到 SDPA-flash 的 87%,是在完全没有多缓冲的情况下做到的**。
BN=64 + stages=3(96 KB,能编译)是否更快,EXP-T01 的扫描是否覆盖过这一档,
**均无法从记录回溯**,列为开放项。

### 3.4 深度的代价:两条预算,只有更紧的那条起作用

#### 3.4.1 shared memory 预算(按修正后的份数重算)

$\text{smem}(N_s) = \max(1, N_s-1)\times 32\,\mathrm{KB}$:

| $N_s$ | 份数 | smem | 对每 block 99 KB |
|---|---|---|---|
| 1 | 1 | 32 KB | 32% |
| 2 | 1 | 32 KB | 32% |
| 3 | 2 | 64 KB | 65% |
| 4 | 3 | 96 KB | **97%** |
| 5 | 4 | 128 KB | **超限**(推断,本仓未测该档) |

**这张表与本仓旧说法不同,必须显式对照**:旧讲义写的是"$N=2$:64 KB;$N=3$:96 KB,
已经逼近上限;$N=4$:名义 128 KB 已经超限"。按 EXP-T08 的实测映射,整张表**右移
一格**:超限的是 $N_s=5$ 而不是 $N_s=4$;$N_s=4$ 恰好卡在 96 KB(99 KB 的 97%),
是**能编译的最深一档**——而它也确实是 EXP-T02 扫描的最深一档。**扫描到 4 就停,
不是因为选择,是因为再往上编译不过**(推断:本仓没跑 $N_s=5$ 验证)。

#### 3.4.2 寄存器预算

`acc` 是 $\mathrm{BM}\times\mathrm{BN}$ 个 fp32,按 num_warps=8(256 线程)摊:
$128\times128/256 = 64$ 个寄存器/线程,仅累加器一项(src/gemm_pipelined.py:91-92
的注释即此)。kperf 实测总 170/线程(终端级证据,EXP-T06 §7)。于是

$$\text{每 CTA 寄存器} = 170\times256 = 43520,\qquad
\left\lfloor \frac{65536}{43520}\right\rfloor = 1$$

**每 SM 只能驻留 1 个 CTA**(Ada Tuning Guide §1.4.1.1:"The register file size is
64K 32-bit registers per SM")。

#### 3.4.3 两条预算合并:一个与旧说法相反的推论

| $N_s$ | smem 允许的 CTA/SM(100 KB) | 寄存器允许的 CTA/SM(64K) | 实际 |
|---|---|---|---|
| 1 | 3 | 1 | **1** |
| 2 | 3 | 1 | **1** |
| 3 | 1 | 1 | **1** |
| 4 | 1 | 1 | **1** |

**四档的 CTA/SM 全是 1,占用率恒为 $8/48 = 16.7\% \approx 17\%$**,与 kperf 卡片
"occupancy 17%(regs 170)"逐字相同(终端级证据,EXP-T06 §7)。

于是一条本仓旧说法被推翻:源码注释写"stages 过深反过来压 CTA 并发"
(src/gemm_pipelined.py:63-64),**在这个 tile 配置下不成立**——寄存器早已把
CTA/SM 钉死在 1,shared memory 加到 96 KB 也压不出更少的 CTA。**加深流水在这里
不花占用率的钱**;它花的是 99 KB 那堵墙上的余量,以及边际收益递减。

那么"最优深度停在 3/4"的真实理由是什么?按数据说话:

- $N_s=3$ 已经把第二份缓冲开出来,拿到 +19%(§3.2.1 的 18.9σ);
- $N_s=4$ 的第三份缓冲在 square4k 上**没有统计显著的收益**(0.37σ),在
  qwen8b 上有 +2.5%(151.567±1.63 → 155.333±0.907,差值 3.77 / 合并 σ 1.87 = 2.0σ,
  勉强显著);
- $N_s=5$ 编译不过(推断)。

**所以本仓的措辞"最优 stage 在 3/4 间随形状摇摆,不宣称唯一最优深度"(EXP-T02 §6)
在修正后的机制下依然成立**,只是理由从"占用率被压"换成了"第三份缓冲的边际收益
落在噪声附近"。EXP-T08 §7 也把"stages=4 在 qwen8b 上快于 3 的原因(3 份缓冲 vs
2 份 + 调度差异)未隔离"明列为开放项。

#### 3.4.4 占用率 17% 却打出 97% 峰值:这不是矛盾

Ada 白皮书对 SM 的描述给出了发射规则:"the AD10x SM is divided into four
processing blocks (or partitions), with each partition containing a 64 KB register
file, an L0 instruction cache, one warp scheduler, one dispatch unit, 16 CUDA
Cores that are dedicated for processing FP32 operations ... one Ada
Fourth-Generation Tensor Core, four Load/Store units, and a Special Function
Unit (SFU)"。

每 SM 4 个 warp scheduler、每个 1 个 dispatch unit → 每周期最多发 4 条指令。
本仓 8 个 warp(每分区 2 个)对 4 个调度器:**只要每个 warp 都有独立指令可发,
4 个调度器就不会空转**。tensor core kernel 的指令级并行度极高(一条 mma 覆盖
大量 MAC),靠**寄存器堆 ILP**藏延迟,比靠"多 warp 轮转"更值钱。
"occupancy 低"只在**带宽 % 与算力 % 两个都低**时才是嫌疑人(docs/theory/04 §2)。

### 3.5 num_stages 在 Triton 编译器里是什么:从论文到本机

Triton 论文(Tillet, Kung, Cox, MAPL '19, DOI:10.1145/3315508.3329973)把
预取写成一个**机器无关**的 pass(§5.1.1 Pre-Fetching):
"Tile-level memory operation inside loops can be problematic, as they may induce
severe latency that cannot be hidden in the absence of enough independent
instructions. It is however possible to mitigate this problem in Triton-IR
directly by detecting loops and adding adequate prefetching code where necessary"。
论文给的 Listing 7 就是最朴素的一级预取:把循环体里的 `load` 提出去一份、用 phi
节点接力。

机器相关的部分在 §5.2:"the optimizations performed by Triton-JIT consist of
(1) hierarchical tiling, (2) memory coalescing, (3) shared memory allocation and
(4) shared memory synchronization"。其中 §5.2.3 Shared Memory Allocation 的做法是
"first calculating the live range of each variable of interest, and then using the
linear-time storage allocation algorithm";§5.2.4 Shared Memory Synchronization
用 RAW/WAR 数据流分析自动插屏障。

**这三点合起来就解释了本仓能做这个实验的原因**:缓冲份数不是你写在代码里的,
是 pass 根据 live range 算出来的;`num_stages` 只是喂给 pass 的一个参数。
于是"双缓冲带来多少"从一句口号变成了一个可测数字——**同一份 kernel 源码、同一组
输入,只改一个 launch 参数**(scripts/test_ew_gemm.py:90-93 的循环)。

代价也要说清:你**控不了**具体指令与 bank conflict,真出问题时只能靠变体对照
而不是读汇编(docs/theory/03)。EXP-T08 那种"读编译产物元数据"的手段,正是在
"读不了汇编"与"只能看时延"之间找到的中间层——**它便宜(不用 profiler、不用
计数器权限)、确定(编译期量、无噪声)、而且能直接证伪一个机制假设**。

顺带说清 `n_regs` 那一列为什么值得看:EXP-T08 实测 GEMM 四档的寄存器数是
162/136/170/178,**非单调**(stages=2 反而最少)。EXP-T08 §7 把它列为未解释项。
本讲义只能给一个方向性的猜测(**未验证**):$N_s=2$ 时没有第二份缓冲,
编译器不需要维护两套地址/阶段变量,寄存器压力反而最小;$N_s\ge3$ 起每多一份缓冲
就多一套。**猜测就标成猜测**——要验证只需再跑一次探针并 dump TTGIR,是一个
明确可执行的下一步。

### 3.6 FP8 per-block:缩放代数怎么和主循环对齐

#### 3.6.1 为什么要细粒度缩放

fp8 e4m3 只有 3 位尾数、满量程 ±448,per-tensor 一个 scale 会被离群值拖垮。
DeepSeek-V3 技术报告(arXiv:2412.19437 §3.3)把做法写成两句:
"for activations, we group and scale elements on a 1x128 tile basis (i.e., per
token per 128 channels); and (2) for weights, we group and scale elements on a
128x128 block basis";格式选择是 "adopt the E4M3 format on all tensors for higher
precision"。本仓原样搬到 sm_89:

- 权重 $B\ (K,N)$:每个 $128\times128$ 块一个 scale,$s^B_{k_g,n_g} = \max|B_{\text{blk}}|/448$;
- 激活 $A\ (M,K)$:每行每 128 长的 K 组一个 scale,$s^A_{m,k_g} = \max|A_{m,k_g}|/448$。

#### 3.6.2 反量化怎么融进累加(题眼)

$$C_{mn} = \sum_{k_g} s^A_{m,k_g}\, s^B_{k_g,n_g}
\left(\sum_{k\in k_g} \hat A_{mk}\hat B_{kn}\right)$$

**为什么两个 scale 可以提到内层求和之外**,把这一步写严格(本讲义推导):内层求和
的下标 $k$ 跑遍第 $k_g$ 组;在这个组内 $s^A_{m,k_g}$ 与 $s^B_{k_g,n_g}$ 都是常数
(与 $k$ 无关),而有限和满足 $\sum_k (c\cdot x_k) = c\sum_k x_k$。**成立条件就是
"组内 scale 与求和下标无关"**——这要求主循环的 K 步长恰好等于缩放组长度。

两条对齐条件因此都不是调优选择,是**正确性前提**:

1. $\mathrm{BLOCK\_K} = 128 = $ 缩放组长度 → 一轮主循环恰好覆盖一个 scale 组。
   若错位,scale 要逐元素进 dot,tensor core 路径直接废掉
   (src/fp8_gemm.py:73-77 的注释即此)。
2. $\mathrm{BLOCK\_N} = 128 = $ 权重块宽 → $s^B$ 退化成**一个标量**;否则要向量化
   (EXP-T06 §7 的开放项)。

#### 3.6.3 固有代价:能算出来的那一部分

每组结果必须先乘 scale 才能并入总累加器,所以**不能**写 `tl.dot(a, b, acc)`
(把累加器直接交给 mma 指令),每组多出一条独立的 FMA 链。这条代价可以算
(本讲义推导,square4k):

- 每 CTA 每 K 组的 tensor core 工作:$\mathrm{BM}\times\mathrm{BN}\times\mathrm{BK}
  = 128^3 = 2.097\times10^6$ MAC $= 4.194\times10^6$ FLOP;
  在 $\pi_{\text{math}}^{\text{fp8}} = 330.3$ TFLOPS 上是 **12.70 ns**。
- 每 CTA 每 K 组的缩放链:`acc += part * sa[:,None] * sb` 对
  $\mathrm{BM}\times\mathrm{BN} = 16384$ 个 fp32 元素做 2 次乘 + 1 次加
  $= 4.915\times10^4$ FLOP;在 $\pi_{\text{math}}^{\text{fp32}} = 82.6$ TFLOPS 上是
  **0.595 ns**。
- 比值:$0.595/12.70 = \mathbf{4.7\%}$。

**所以缩放乘法本身只值 5% 左右,它解释不了"理论 2× 只吃到 1.5×"的那 30% 缺口。**
这是本篇对旧解释的一次收紧:旧说法"缩放乘法 + fp32 累加占算力"给的是方向,
量级不对。

那缺口在哪?本仓能给出的是**一个可证伪的假设,不是结论**:因为不能用
`tl.dot(a,b,acc)`,`part` 与 `acc` 这两个 $128\times128$ fp32 tile **必须同时活着**,
各占 $128\times128/256 = 64$ 个寄存器/线程,合计 128——而 fp16 版只需要一份 64。
接近 255/线程的上限时,编译器要么减少调度自由度、要么 spill。**验证方法是现成的**:
把 EXP-T08 的探针指向 `_fp8_gemm_kernel`,读 `n_regs` 与 `n_spills` 即可
——本仓**没有做**,列为开放项。目前只有 kperf 的定界(compute-bound、
约 70% fp8 峰值,终端级证据,NCU 不可用)。

顺带核对一下那个"70%":$228.1 / 330.3 = 69.1\%$。**kperf 的口头读数与白皮书
规格值算出来的比值对得上**——这是一条独立的小闭环。

#### 3.6.4 Ada / Hopper / Blackwell 的界线:一处必须纠正的流行说法

先给三代指令的准确对照(每一行都有出处):

| | sm_89(Ada,本实现) | sm_90(Hopper,DeepGEMM 本体) | sm_100(Blackwell) |
|---|---|---|---|
| 矩阵指令 | `mma.sync`(warp 级同步集合,PTX §9.7.15) | `wgmma.mma_async`(warpgroup 级异步,PTX §9.7.16) | `tcgen05.mma` |
| 搬运 | `cp.async`(PTX §9.7.9.26) | TMA(张量批搬运) | TMA |
| **硬件块缩放** | **无** | **无** | **有**(mxf8f6f4 / mxf4 / nvf4 等 kind) |
| 细粒度缩放怎么做 | 累加器侧手乘(本仓) | **也是 CUDA core 侧**(见下) | 硬件:$D = C + (A\cdot SFA)(B\cdot SFB)$ |

**要纠正的说法**:本仓过去写过"Hopper wgmma 原生 scale 槽替你省掉那部分"
(docs/theory/06 §2 的表、src/fp8_gemm.py:24-26 的 docstring)。按 PTX 与
CUTLASS 的语义,**wgmma 的三个 scale 操作数不是任意缩放因子**:
`scale_D` 取 0 或 1,控制累加器是否被清零("scale_D is either 0 or 1, and
controls whether or not the accumulator is zero-initialized");
`scaleA`/`scaleB` 取 1 或 −1,只用于取负("scaleA and scaleB are either 1 or −1
for negating the operand")。**它们是符号位,不是缩放槽。**

真正拥有"每 16/32 个元素一个 scale factor"的硬件块缩放,是 **Blackwell 的
`tcgen05.mma`** 才引入的(CUTLASS 文档:带 `mxf8f6f4`/`mxf4`/`nvf4` kind 的指令
执行 $D = C + (A\times SFA)\times(B\times SFB)$,scale factor 按 K 维每 16 或 32 个
元素一个)。

那么 DeepGEMM 在 Hopper 上是怎么做的?DeepSeek-V3 报告 §3.3 写得很清楚:因为
H800 的 FP8 tensor core 累加精度只有约 14 位,他们
"adopt the strategy of promotion to CUDA Cores for higher precision",在每
$N_C = 128$ 个元素的 MMA 之后把部分和搬到 FP32 寄存器上用 CUDA core 累加。
**这与本仓在 Ada 上做的事是同一件事**(src/fp8_gemm.py:111-113 的注释就叫
"二级累加(DeepGEMM 的 accumulator promotion 在 mma 世代的形态)")。

**所以修正后的结论是**:Hopper 相对 Ada 的优势在**搬运(TMA)与矩阵指令的异步性
(wgmma)**,不在"原生 scale 槽";细粒度缩放两代都得在 CUDA core 侧做。
"DeepGEMM 为什么不能直接跑 4090"的准确答案是**前两行,不是第三行**——
DeepGEMM 的 README 把要求写成 "NVIDIA SM90 or SM100 architecture GPU",
sm_89 不在其中。

**这条纠正不改变本仓任何数字**:1.5× 的口径、69% 的峰值占比、4.7% 的缩放链成本
都不动;改的是对"缺口来自哪"的归因表述(§3.6.3 已经把归因收紧成一个待验证假设)。

#### 3.6.5 Triton 侧的一个旁证:`dot_scaled` 在没有硬件支持时怎么办

本机 Triton 3.6.0 提供了 `tl.dot_scaled`,支持 OCP microscaling 格式,其文档
(language/core.py)明说:"Software emulation enables targeting hardware
architectures without native microscaling operation support. Right now for such
case, microscaled lhs/rhs are upcasted to `bf16` element type beforehand for dot
computation."——**在没有硬件块缩放的架构上,连编译器给的路也是"升到 bf16 再算"**,
等于放弃 fp8 的吞吐。这从另一个方向印证了 §3.6.4:Ada 上没有捷径,手写累加器侧
缩放(本仓的做法)已经是这一代能做的形态。

### 3.7 硬件语义层:这些写法是被什么定死的

#### 3.7.1 cp.async 的三条语义与它们对代码的约束

PTX ISA §9.7.9.26.3.1 规定 `cp.async` 的 cache 修饰符 `ca`(cache at all levels)
与 `cg`(cache in global level only),并限制 **cp-size 只能是 4、8 或 16 字节,
用 `cg` 时必须是 16**。

对本仓的影响是间接但真实的:BLOCK_K=64、fp16 → 一行 tile 是 $64\times2 = 128$ 字节,
是 16 的整数倍,所以每个线程可以搬 16 B 的整块;若 BLOCK_K 取一个使行字节数不是
16 倍数的值,编译器只能退回更小的 cp-size 或普通 load,**带宽利用率直接掉**。
**这是"tile 维度偏好 2 的幂"的一个硬件层理由**,比"对齐比较好"具体得多。
(本仓未做非 16 倍数 BLOCK_K 的对照实验,该推论标为推断。)

`commit_group` / `wait_group` 的组语义见 §3.3.3,它把"流水深度"落成"允许在飞的
组数",进而落成 shared memory 份数。

#### 3.7.2 mma 是 warp 级集合操作:为什么循环体里不能有发散分支

PTX ISA §9.7.15 定义 matrix fragment 时写明 "each thread in a warp holds a
fragment of the matrix",`mma.m16n8k16` 的布局由 `groupID = %laneid >> 2` 与
`threadID_in_group = %laneid % 4` 索引;同章说明 mma 是 "warp-wide collective",
warp 内所有线程必须一起执行。

两条推论:

1. **布局不可自选**:寄存器里哪个 lane 拿矩阵的哪几个元素,是 ISA 规定的置换。
   Triton 的 layout 系统就是在做这件事,而你看不到它——这正是"控不了具体指令"
   这句代价的具体内容之一。
2. **不能在 dot 周围写发散分支**:warp 内发散会让集合操作无法对齐。本仓的
   `IEEE_DOT` 是 `tl.constexpr`,在**编译期**分叉成两个 kernel,而不是运行期分支
   (src/gemm_pipelined.py:72-77);这不是风格选择,是被集合语义逼出来的写法。

#### 3.7.3 mma 的 M 维粒度:一个需要修正的措辞

`mma.m16n8k16` 的 M 维粒度是 16。一个常见说法是"`tl.dot` 要求 M ≥ 16",在本机的
Triton 3.6.0 上**不准确**:NVIDIA 后端的 `min_dot_size` 返回 `(1, 1, 16)`
(8 位输入是 `(1, 1, 32)`),注释原文是 "For small M/N the input we can still use
tensorcores with padding";Python 侧只断言
"Input shapes should have M >= ..., N >= ... and K >= ..."。

**准确说法**:M=1 不会编译失败,tile 会被 pad 到 16 行,15/16 的 tensor core 行
利用率被浪费。src/gemm_pipelined.py:116-117 的注释("BLOCK_M=128 在 M=1 时 mma
行利用率 1/128,127/128 的 tensor core 算力全废")说的是同一件事的更极端版本,
**它的口径是对的**;需要修正的是别处把这件事说成"编译要求"的措辞(讲义 01 §3.7.3
已同步修正)。

#### 3.7.4 L2 与 L1 的容量层级:grouped 调度到底在优化谁

| 层 | 容量 | 出处 |
|---|---|---|
| 寄存器堆 | 64K × 32 bit / SM,整卡 32768 KB | Ada Tuning Guide §1.4.1.1;白皮书 Table 2 |
| L1 / shared(统一) | 128 KB / SM,整卡 16384 KB;shared 最多 100 KB/SM、99 KB/block | Ada Tuning Guide §1.4.2.2 / §1.4.1.1 |
| L2 | **73728 KB(整卡共享)** | 白皮书 Table 2 |
| HBM | 24 GB,1008 GB/s | 白皮书 Table 2 |

grouped 调度优化的是 **L2 那一层**:它不改变每个 CTA 读多少,只改变"同一时刻在跑的
CTA 们读的是不是同一批数据"。因为 L2 是**整卡共享**的,跨 CTA 的复用只能在这一层
兑现——shared memory 是每 block 私有的,帮不上跨 CTA 复用的忙。**这是"为什么
grouped 调度和 tile 大小是两条正交的腿"的准确回答。**

顺带给一个量级感:AD102 完整芯片有 98304 KB L2("AD102 has been outfitted with
98304 KB of L2 cache, an improvement of 16x over the 6144 KB that shipped in
GA102",白皮书 Memory Subsystem 段),4090 上是裁剪后的 73728 KB。
**16 倍于上一代**——L2 变大这件事本身,就是 grouped 调度这类"把复用距离压短"的
优化在 Ada 上格外划算的原因。

### 3.8 每个魔法数的来历(理论上界 / 硬件约束 / 实测扫描)

| 参数 | 值 | 类别 | 依据(可核验) |
|---|---|---|---|
| BLOCK_M / BLOCK_N | 128 / 128 | **硬件约束** | acc 一项就占 64 regs/线程(128·128/256),再大撞 255/线程上限 |
| BLOCK_K | 64 | 实测 + 硬件 | 单份 tile 32 KB;行字节 128 B 是 cp.async 16 B 的整数倍(§3.7.1) |
| GROUP_M | 8 | **未扫参** | src/gemm_pipelined.py:87 默认值;L2 复用距离的经验取值,本仓未扫 |
| num_warps | 8 | 实测 | 256 线程恰好让 acc 摊到 64 regs/线程 |
| num_stages(fp16) | 3 | **实测扫描** | EXP-T02 四档;§3.2.1 给出 18.9σ 的显著性 |
| num_stages(fp32) | 2 | 硬件约束 | fp32 tile 字节翻倍(src/gemm_pipelined.py:95-98) |
| BLOCK_N(fp32) | ≤64 | 硬件约束 | 同上 |
| FP8 `BLOCK_K` | 128 | **正确性前提** | 必须等于缩放组长度(§3.6.2),故写死在 kernel 内 |
| FP8 `BLOCK_N` | 128 | **正确性前提** | 使 $s^B$ 退化成标量;解耦需 sb 向量化(EXP-T06 §7) |
| `FP8_MAX` | 448.0 | 理论 | e4m3 最大正规值 |
| GEMM 的 M 自适应 tile | 128/32/16 | 理论 + 硬件 | M=1 时 mma 行利用率 1/128(§3.7.3) |

**这张表的用法**:被问"这个数为什么是 128",答案必须落在四类之一——理论、硬件、
实测、未扫参。GROUP_M=8 落在"未扫参",说出来不丢人;**说不出属于哪一类才丢人**。

## 4. 代码逐段走读:src/gemm_pipelined.py、src/fp8_gemm.py 与探针脚本

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

补一条这段代码的**适用边界**:重映射的收益依赖"同一组内的 CTA 真的同时在跑"。
本仓 CTA/SM = 1(§3.4.3),一波 128 个 CTA,而 GROUP_M × num_pid_n = 8 × 32 = 256
个 CTA 才凑满一组——**一组横跨两波**。也就是说,组内后半段的 CTA 与前半段并不同时
执行,复用要靠 L2 在两波之间不被冲掉(67 MB 操作数对 72 MB L2,勉强够,§3.1.1)。
**GROUP_M 与波大小的匹配关系本仓未扫参**,这是 §3.8 里 GROUP_M 被标成"未扫参"的
具体含义。

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
`num_stages`(第 3 段的 launch 参数)。三个细节:①**K 尾块用 mask 兜底**
(`offs_k + k0 < K`),补 0 对 dot 零贡献,于是主循环外不需要单写一个收尾循环——
代价是每轮多两个谓词;②`acc = tl.dot(a, b, acc)` 把累加器作为第三参数,**直接映射
mma 指令的累加寄存器**,省掉一趟独立的加法(对比第 7 段的 fp8 版:那里因为要先乘
scale,**必须**退回 `acc += ...`);③`IEEE_DOT` 是 `tl.constexpr`,编译期分叉而非
运行期分支(§3.7.2)。改错会怎样:把 `acc` 从第三参数拆成 `acc += tl.dot(a, b)`,
数值一样、性能掉一截,而且这个损失在 profile 里表现为"更多 FFMA 指令",不看汇编
很难归因。

**这段注释本身有两处需要按 EXP-T08 修正,必须显式说出来**:

- 注释写"分配 N 份 smem 缓冲",实测是 $\max(1,N-1)$ 份(§3.3.2)。
- 注释写"stages 过深反过来压 CTA 并发",在这个 tile 配置下不成立——寄存器已把
  CTA/SM 钉死在 1(§3.4.3)。

**代码不改、数字不改,改的是对它的解释**;这正是 EXP-T08 §6 对自己的定位
("EXP-T02 的数字不变,变的是对它的解释")。**能把"我以前的解释错了"写进讲义,
比讲义里没有错误更有价值。**

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
最优档——按修正后的语义,它也正是**第一个真正开出双缓冲的档**。注释里
"仅 acc 就占 $128\cdot128$ fp32 / 256 线程 = 64 regs/线程,kperf 实测总 170"
是把 §3.4.2 的预算账写进代码——**默认参数必须能指回它的来源实验**,否则半年后
没人敢动。fp32 路线降 `block_n` 到 64 的理由同 FA2:tile 字节翻倍,片上装不下。
改错会怎样:把 `num_stages` 默认改成 2("因为教科书说双缓冲"),这个形状上直接丢
20% 算力,而正确性测试全绿——**现在你还知道更精确的原因:那一档连第二份缓冲都没有。**

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

再看一处这段代码里藏着的成本:`weight.t().contiguous()` **每次调用都会实体化一份
转置后的权重**。在 bench 里它被 warmup 摊掉了,在真实 serving 里这是每步一次的
$O(\text{out}\times\text{in})$ 拷贝——**接口层的隐藏拷贝**,与讲义 01 §8.2 提到的
KV cache `.contiguous()` 是同一类坑。本仓没有把它算进任何数字(bench 只测
`gemm`,不测 `linear`),**这条列为已知的接口层缺陷**。

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

角色:§3.6 缩放布局的构造端。`reshape(K//G, G, N//G, G)` 把 $128\times128$ 块折成
两个维度,`amax(dim=(1,3))` 一次求出每块的绝对最大值——**用 view 而不是循环**是这段
唯一的性能要点。`scale = amax / 448` 让块内最大值恰好顶到 e4m3 满量程,把有限的 3 位
尾数全用在有效动态范围上;`clamp(min=1e-8)` 防全零块除 0。整除断言不只是防御:它同时
是**缩放组代数的前提**(§3.6.2)与"GEMM 主循环可以零 K 维 mask"的依据。
改错会怎样:把 absmax 换成 per-tensor 一个 scale,kernel 一行不用改、速度一样,只是
量化误差从 3.6e-2 量级劣化到被离群值主导——**精度回归不会让任何性能测试变红**。

一个值得算一遍的小账(本讲义推导):scale 张量本身多大?权重 $(K,N)$ 的 scale 是
$(K/128, N/128)$ fp32,即每 $128\times128\times1\,\mathrm{B} = 16384$ B 的 fp8 权重
配 4 B scale,**额外开销 0.024%**;而激活的 per-token-group scale 是 $(M, K/128)$
fp32,每 $128\times1$ B 配 4 B,**开销 3.1%**。两边差 128 倍,原因是激活的分组
只在 K 维、不在 token 维。**"细粒度"的存储代价几乎全在激活侧**——这也是为什么
生产实现会把激活量化融进上游算子的 epilogue,顺手把 scale 写出去,而不是单独跑一趟。

**第 6 段 · BLOCK_K 与缩放组硬对齐**(src/fp8_gemm.py:73-77)

```python
    # 面试点:BLOCK_K 与缩放组硬对齐——一轮主循环恰好覆盖一个 scale 组,
    # 组内 scale 是常量,反量化才能从 dot 里提出来变成秩 1 修正
    # (sa 行向量 × sb 标量);若 BLOCK_K 与组错位,scale 要逐元素进 dot,
    # tensor core 路径直接废掉
    BLOCK_K: tl.constexpr = 128            # 与缩放组硬对齐
```

角色:一行 `constexpr` 承载 §3.6.2 的全部前提。把 `BLOCK_K` 写死在 kernel 内(而不是
开放成参数)是刻意的:它必须等于缩放组长度,开放出去就等于把一个**正确性前提**降级成
调优旋钮。改错会怎样:BLOCK_K 取 64,一轮主循环只覆盖半个 scale 组,反量化就不能提到
dot 之外——要么逐元素乘 scale(tensor core 路径废掉),要么算错。

**顺带说清一个连带后果**:BLOCK_K 被钉死在 128 之后,fp8 kernel 的单份 tile 是
$(128+128)\times128\times1\,\mathrm{B} = 32\,\mathrm{KB}$(fp8 是 1 B/元素),与 fp16
路线的 32 KB **恰好相同**。所以 §3.4.1 那张 shared memory 表可以原样套到 fp8 kernel:
`num_stages=3`(默认,src/fp8_gemm.py:123)= 2 份 = 64 KB。**fp8 省下来的字节被
翻倍的 BLOCK_K 吃回去了**——这是"降精度不一定省片上资源"的一个具体例子
(本讲义推导;本仓未对 fp8 kernel 跑资源探针,数值为按公式推算)。

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

角色:§3.6.2 公式在 kernel 里的落点,也是与第 2 段最值得对读的一段。差异只有一处但很
致命:这里**不能**写 `tl.dot(a, b, acc)`,因为本组结果要先乘 $s^A s^B$ 才能并入总
累加器,于是每组多一条独立 FMA 链。另外两点:①K 维**零 mask**,因为量化函数已断言
$K \bmod 128 = 0$,热循环因此没有分支;②`sb` 靠 `BLOCK_N=128` 与权重块对齐退化成
单标量,解耦 BLOCK_N 需要 sb 向量化(EXP-T06 §7 的开放项)。
改错会怎样:`sa` 的 `mask=offs_m < M, other=0.0` 若写成 `other=1.0`,越界行会被乘上
无意义的 scale,而这些行本来就要被末尾的 store mask 丢掉——不会错,但会让人误以为
`other` 值无关紧要;真正不能动的是 `sb` 的块号算式 `pid_n * BLOCK_N // 128`。

**注释里"Hopper wgmma 原生 scale 槽替你省掉的那部分"这句需要按 §3.6.4 修正**:
wgmma 的 `scale_D` 只有 0/1(是否累加)、`scaleA`/`scaleB` 只有 ±1(取负),
**没有任意缩放槽**;硬件块缩放是 Blackwell 的 `tcgen05.mma` 才有的能力。
Hopper 上的 DeepGEMM 同样把细粒度缩放放在 CUDA core 侧做(DeepSeek-V3 报告
§3.3 的 "promotion to CUDA Cores")。**所以这条 FMA 链不是"Ada 独有的税",
它在 Hopper 上也存在**;Hopper 的优势在 TMA 与 wgmma 的异步性。
数字不变,归因表述修正。

**第 8 段 · fp8 启动器:BLOCK_N 为什么不开放**(src/fp8_gemm.py:122-130)

```python
def fp8_gemm_prequant(a_fp8, sa, b_fp8, sb, out_dtype=torch.float16,
                      block_m=128, group_m=8, num_warps=8, num_stages=3):
    """量化好的输入直接 GEMM(bench 用,不含量化成本;227.7/235.7 TFLOPS
    即此口径)。BLOCK_N 固定 128 与权重块对齐(sb 取标量的前提)。"""
    M, K = a_fp8.shape
    N = b_fp8.shape[1]
    c = torch.empty(M, N, device=a_fp8.device, dtype=out_dtype)
    BLOCK_N = 128
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, BLOCK_N),)
```

角色:把"正确性前提"从签名里拿掉的示范。`block_m` 是参数、`BLOCK_N` 是局部常量——
两者的区别不是风格,是**前者只影响性能、后者影响正确性**。docstring 第一句
把口径写死("量化好的输入直接 GEMM,不含量化成本;227.7/235.7 TFLOPS 即此口径"),
这样任何人从函数签名就能看到那两个数字属于哪个口径,不必翻记录。
改错会怎样:把 `BLOCK_N` 提到参数里并传 64,`sb` 那个"整块落在同一权重列块内"的
前提就破了,`pid_n * BLOCK_N // 128` 会让相邻两个 CTA 取到同一个 scale——
**结果错,而且只在 N 方向的奇数块上错。**

**第 9 段 · num_stages 扫描:单变量实验长什么样**(scripts/test_ew_gemm.py:88-95)

```python
        tf = 2 * M * N * K / 1e12
        row = {"cublas_ms": round(bench(lambda: a @ b), 4)}
        for st in (1, 2, 3, 4):
            ms = bench(lambda: gemm(a, b, num_stages=st))
            row[f"stages{st}_ms"] = round(ms, 4)
            row[f"stages{st}_tflops"] = round(tf / (ms / 1e3), 1)
        row["cublas_tflops"] = round(tf / (row["cublas_ms"] / 1e3), 1)
        out["bench"][f"gemm_{tag}"] = row
```

角色:整个"双缓冲值多少"实验的全部代码。四档 stages 用的是**同一个 kernel、同一组
输入张量 `a`/`b`、同一次进程、同一个 `bench()` 计时函数**,唯一变量就是 launch 参数。
所以四档之间的差可以直接归因给流水配置,不需要额外的控制实验——**这是"把口号变成
可测数字"的实验形态**。三个细节:①`tf` 用 $2MNK$,与 cuBLAS 臂共用同一分子,
保证 TFLOPS 可比;②cuBLAS 臂写作 `bench(lambda: a @ b)`,**对照物命名诚实**:
它是 torch.matmul 分发(cuBLASLt),不是直接调 cuBLAS API;③四档与 cuBLAS 在
同一个 `row` 里出,不存在"快的那版和对的那版不是同一次运行"。
改错会怎样:把四档拆成四次进程跑,时钟/温度漂移会混进档间差,18.9σ 那个结论就
不再干净。

**第 10 段 · 编译期资源探针:不跑 bench 也能证伪机制假设**(scripts/probe_smem_regs.py:26-40)

```python
def drain(jit, tag, extra):
    for k in jit.device_caches[dev][0].values():
        row = {**extra, "shared_bytes": k.metadata.shared, "n_regs": k.n_regs,
               "n_spills": k.n_spills, "num_warps": k.metadata.num_warps}
        res[tag].append(row)
        print(tag, row)


# --- GEMM: BM128/BN128/BK64/w8,单份 tile = (128+128)*64*2B = 32 KB ---
a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
b = torch.randn(512, 512, device="cuda", dtype=torch.float16)
for st in (1, 2, 3, 4):
    _gemm_kernel.device_caches[dev][0].clear()
    gemm(a, b, num_stages=st)
    drain(_gemm_kernel, "gemm_stages", {"num_stages": st})
```

角色:EXP-T08 的全部机制。三个设计点值得学:

1. **读的是编译产物,不是运行时行为**:`metadata.shared` / `n_regs` / `n_spills`
   都是编译期确定的量,**没有噪声、不需要多轮、不需要 profiler 权限**。这解决了
   本容器"无性能计数器权限"(docs/theory/04)这个长期约束下的一大类问题。
2. **每档先 `clear()` 缓存**:Triton 的 JIT 按签名缓存编译结果,不清就会读到上一档
   的产物。**这是最容易写错、且错了看不出来的一行**——数字会全部相同,而"全部相同"
   恰好就是被测假设的证伪信号,你会得到一个假的证伪。
3. **用 512×512 的极小输入**:探针只需要触发一次编译,输入尺寸与结论无关;
   小输入让脚本秒级完成,也避免占用 GPU。脚本头明确写"只做编译 + 一次极小 launch,
   不产生任何 benchmark 时延数字"——**profiler 隔离原则的另一种形态**。

改错会怎样:漏掉 `clear()`,四档读到同一份产物,`smem(2)==smem(1)` 这个"证伪"
就变成了脚本 bug 而不是发现。**一个能证伪假设的实验,必须先能证伪自己的实现。**

## 5. 实验数据怎么读

### 5.1 fig2:num_stages 扫描与 cuBLAS 对照

figures/fig2_gemm_stages.png(脚本 scripts/plot_readme_figures.py:94-118)。

- **轴与口径**:横向条形,x = TFLOPS(4096³ fp16,越高越好),y 自上而下 =
  cuBLAS(torch.matmul)/ Triton stages=4/3/2/1;误差条 = **3 轮 std**;源数据
  data/derived/exp-t02_stability_3rounds.csv。标题即结论句(单图单结论)。
- **单变量设计**:四档 stages 用的是**同一个 kernel、同一组输入、同一次进程**
  (scripts/test_ew_gemm.py:90-93 的循环),唯一变量就是 launch 参数。所以四档之间的
  差可以直接归因给流水配置,不需要额外的控制实验。
- **对照物命名诚实**:cuBLAS 这一行指的是 `torch.matmul`(fp16 走 cuBLASLt),
  脚本里写作 `bench(lambda: a @ b)`(scripts/test_ew_gemm.py:89),记录与措辞约定
  都注明 **cuBLAS = torch.matmul dispatch**。它不是直接调 cuBLAS API 的结果,含
  torch 的分发开销——对 0.86 ms 量级的大 GEMM,分发那几微秒可忽略,但口径要写出来。
- **"打平"的准确说法**:square4k 3 轮 **159.4±1.2 vs 160.0±0.7 TFLOPS**,差值落在
  误差条内,所以说"打平(差 0.4% 内,单轮 160.5 vs 159.8)";**不能**说"超过 cuBLAS"。
- **"反超 4.8%"的准确说法**:那是 Qwen3-8B up_proj 形状(2048×4096×12288)的**单轮
  存盘值** 154.4(stages=4)vs 147.3。3 轮口径下是 155.3±0.9 vs 150.4±4.5,幅度收窄到
  +3.3%,且 cuBLAS 侧的轮间 std 达 2.98%(四个数字里波动最大的一个)。**诚实的读法是:
  这一格从"打平"到"小幅反超"都在数据支持范围内,引用 4.8% 时必须带"单轮存盘 raw"**
  (措辞约定:只引存盘 raw 轮)。
- **机理账**:$2MNK/t$。square4k 的 $2\cdot4096^3 = 137.4\ \mathrm{GFLOP}$,
  $/0.8566\,\mathrm{ms} = 160.4\ \mathrm{TFLOPS}$(与存盘的 160.5 对上);对 165.2
  TFLOPS 峰值是 97%,与 kperf 卡片"算力 98%、occupancy 17%(regs 170)"吻合(终端级
  证据,EXP-T06 §7)。**到顶了**——这也是为什么本仓不再往下扫 BM/BN/BK 全空间:剩余
  空间不足 3%,而 tile 全扫的代价远大于收益(EXP-T02 §7 如实列为未做项)。
- **正确性同批出**:两形状相对误差 ~7e-4(fp16 输入 fp32 累加 vs torch.matmul,
  data/derived/exp-t02_stability_3rounds.csv 的 correctness 行)。性能表和正确性表
  出自**同一次运行**,不存在"快的那版和对的那版不是同一个"。

**这张图现在要配一句新的读法**(EXP-T08 之后):四条 Triton 条的正确标签不是
"1/2/3/4 级流水",而是"**1/1/2/3 份缓冲**"。按份数重画,那张图会变成一条干净的
"份数 1 → 2 有大跳,2 → 3 几乎持平"的曲线,而不是"2 级很弱、3 级突然很强"这种
需要额外解释的形状。**同一批数据,换一个正确的横轴标签,反常就消失了。**

### 5.2 EXP-T02 表的三层读法

| 层 | 内容 | 判据 |
|---|---|---|
| 数字层 | 131.9 / 133.5 / 160.5 / 157.1(单轮);132.0±1.06 / 133.7±0.58 / 159.4±1.23 / 158.0±3.52(3 轮) | 只引存盘轮;3 轮给 mean/std |
| 显著性层 | 1→2 为 1.4σ(不显著);2→3 为 18.9σ;3→4 为 0.37σ | §3.2.1 的合并 σ 计算 |
| 机制层 | 份数 1/1/2/3(EXP-T08 编译期实测) | `metadata.shared` / 32 KB |

**三层必须一起报**:只报数字层会得出"双缓冲只值 1%"的错误结论;报到显著性层能
知道那 1% 不可靠;报到机制层才知道**为什么**不可靠(那两档是同一种配置)。
这是本仓"假设先锁阈值、证伪照登"这套做法能兑现价值的一个完整实例:
**EXP-T02 的数字一个没改,结论的含义变了。**

### 5.3 FP8 的四口径表(EXP-T06)

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

**"理论 2× 只吃到 1.5×"的账,现在能拆得更细**(本讲义推导 + 白皮书规格值):

| 项 | 数值 | 来源 |
|---|---|---|
| fp8 / fp16 峰值比(均 FP32 累加) | 330.3 / 165.2 = **2.00** | Ada 白皮书 Table 2 |
| 本仓 fp8 对 fp8 峰值 | 228.1 / 330.3 = **69%** | EXP-T06 3 轮;与 kperf "~70% fp8 峰值" 一致 |
| cuBLAS fp16 对 fp16 峰值 | 156.9 / 165.2 = **95%** | EXP-T06 3 轮 |
| 实测比 | 228.1 / 154.9 = **1.47** | 两者相除 |
| 缩放链的理论成本 | **4.7%** 的 tensor core 时间 | §3.6.3 |

**关键行是第 2 与第 3 行**:缺口不在"fp8 峰值打折",而在**本仓的 fp8 kernel 只吃到
自己峰值的 69%,而 cuBLAS 吃到 fp16 峰值的 95%**。缩放链只解释其中 4.7 个点。
剩下的 26 个点,本仓给不出实测归因;**唯一列出的假设是"两个 fp32 tile 同时活着导致
寄存器压力翻倍"(§3.6.3),验证手段现成(把 EXP-T08 的探针指向 fp8 kernel),
本仓未做**。把"缺口=架构税"这种笼统说法换成"缺口=69% vs 95%,其中 4.7 点已解释、
其余待测",是本篇对这张表最大的改动。

## 6. 误区与边界

1. **"双缓冲是 GEMM 的标配收益,而你测出只有 1%"**——两句都要修正。EXP-T08 证明
   `num_stages=2` 在本机只有 1 份缓冲,**那一档根本不是双缓冲**;真正的双缓冲是
   `num_stages=3`,它值 +19%(18.9σ)。所以正确说法是:**双缓冲确实值钱,而
   `num_stages` 这个参数名会骗人。**(§3.3)
2. **"num_stages 就是 shared memory 份数"**——不是。文档说的是"在飞的迭代数"
   (`tl.range` docstring),实测份数是 $\max(1, N_s-1)$。差 1 来自"正在被消费的那一
   迭代不额外占预取缓冲"(§3.3.3)。
3. **"stages 越深越压占用率"**——在本仓这个 tile 上不成立。寄存器(170/线程 × 256)
   已经把 CTA/SM 钉死在 1,shared memory 从 32 KB 加到 96 KB 也压不出更少的 CTA
   (§3.4.3)。**先算清哪条预算更紧,再谈代价。**
4. **"occupancy 低说明 kernel 没写好"**——本仓 GEMM occupancy 17% 却打出 97% 峰值
   算力。tensor core kernel 靠寄存器堆 ILP 藏延迟,占用率只在**带宽 % 与算力 %
   两个都低**时才是嫌疑人(§3.4.4,docs/theory/04 §2)。
5. **"打平 cuBLAS 可以简写成打平"**——不行。完整口径是**限两测形状 fp16、cuBLAS =
   torch.matmul dispatch(cuBLASLt)、未做全形状扫描**;而且"反超 4.8%"是单轮存盘值,
   3 轮口径收窄到 +3.3% 且对照侧 std 近 3%(§5.1)。再补一条新发现的舒适条件:
   **两个测试形状的 CTA 数恰好都是 128 的整数倍**(8 波 / 12 波,§3.1.2),
   没有尾波损失。
6. **"FP8 给推理提速 1.5×"**——最容易被误引的一句。1.5× 是**预量化孤立 GEMM** 的口径;
   同一份代码把在线量化计进去只有 72.9 TFLOPS(§5.3)。把 1.5× 说成端到端提速,是本仓
   措辞约定明令禁止的措辞。
7. **"Hopper 的 wgmma 有原生 scale 槽"**——**不成立**。wgmma 的 `scale_D` ∈ {0,1}
   控制是否累加,`scaleA`/`scaleB` ∈ {1,−1} 只用于取负;硬件块缩放是 Blackwell 的
   `tcgen05.mma` 才有(§3.6.4)。DeepGEMM 在 Hopper 上同样把细粒度缩放放在
   CUDA core 侧(DeepSeek-V3 报告 §3.3 的 promotion to CUDA Cores)。
   **本仓过去的表述有误,数字不受影响。**
8. **"这套账换个形状照样成立"**——不一定。§3.1.1 的实测比 HBM 上界快 2.5 倍,前提是
   4096² 的两个操作数合计 67 MB、进得了 72 MB 的 L2;权重远大于 L2 的形状要重算访存账
   (推断,本仓未测)。
9. **"份数公式 $\max(1,N-1)$ 是 Triton 的通用规律"**——**只在本机这一版上验证过**。
   EXP-T08 的环境是 triton 3.6 / torch 2.11 / RTX 4090;换版本、换后端、换到
   `tl.range` 的 `num_stages` 属性(语义与 kernel 参数不同,见 §3.3.1)都可能不一样。
   **它是一条实测事实,不是一条语言规范。**

**适用边界**:全部数字来自单卡 RTX 4090、fp16 输入 fp32 累加、两个测过的形状
(4096³ 与 2048×4096×12288);tile 空间未全扫、GROUP_M 未扫;stall 归因无性能计数器
支持,机制结论标注为从数据反推或从编译产物反推;$N_s=5$ 是否超限为推断、未测;
fp8 kernel 的寄存器数未探;cp.async 的组数与缓冲份数的对应关系未在本仓直接观测
(EXP-T08 §7);FP8 部分限 e4m3、BLOCK_N 硬绑 128、激活量化未融合进上游算子
(EXP-T06 §7 的开放项)。

## 7. 连环追问

1. **Q:GEMM 为什么一定要分块?**
   算术强度。不分块是 0.5 FLOP/Byte,机器平衡点 164,只能吃到峰值 0.3%(§2.1)。
   分块把强度抬到 $\mathrm{BM}\cdot\mathrm{BN}/(\mathrm{BM}+\mathrm{BN})$,
   $128\times128$ 给 64 FLOP/Byte。
2. **Q:64 还是小于 164,为什么还能打到 97% 峰值?**
   因为整问题的算术强度是 $MNK/(MK+NK+MN) = 1365$ FLOP/B(§2.1 第三层),
   剩下的重读大部分命中 L2 而不是 HBM——tile 模型的 HBM 上界要 2.13 ms,
   compulsory 下界只要 0.100 ms,实测 0.857 ms 贴着算力下界 0.832 ms(§3.1.1)。
   tile 的任务只是"抬到够 L2 接手",grouped 调度负责把复用距离压到 GROUP_M。
3. **Q:`num_stages=2` 和 CUDA 手写双缓冲是一回事吗?**
   **不是**。本机 Triton 3.6 下 `num_stages=2` 只分配 1 份缓冲(EXP-T08),
   等价于"没有双缓冲";CUDA 手写双缓冲对应的是 `num_stages=3`。
   这是本篇最反直觉的一条,也是 EXP-T02 那张表被误读多年的原因。
4. **Q:那你怎么知道是 1 份而不是 2 份?**
   编译一次,读 `CompiledKernel.metadata.shared`:四档是
   32768/32768/65536/98304 字节,除以单份 tile 32 KB 就是 1/1/2/3
   (scripts/probe_smem_regs.py:26-40)。**不跑 bench、无噪声、不需要 profiler 权限。**
5. **Q:文档不是说 num_stages 是"在飞的迭代数"吗?那文档错了?**
   文档没错,是两个量。$N_s$ 个迭代在飞时,其中一个正被 `tl.dot` 消费,
   只有 $N_s-1$ 个需要预取缓冲(§3.3.3)。**参数名描述的是流水级数,资源账要按
   份数算**——把这两个量当成一个,就会得到"2 级双缓冲只值 1%"这种错误结论。
6. **Q:那为什么不一直加深?**
   三条:①$N_s=5$ 需要 4 份 = 128 KB > 99 KB,编译不过(推断);②第三份缓冲的
   边际收益在 square4k 上不显著(0.37σ)、在 qwen8b 上 +2.5%(2.0σ);
   ③本仓明确不宣称唯一最优深度(EXP-T02 §6)。**注意"压占用率"不在这三条里**
   (§3.4.3)。
7. **Q:occupancy 只有 17%,不该先修这个吗?**
   不该。带宽 % 与算力 % 只要有一个贴顶就说明延迟已经藏住了;本仓算力 97-98%,
   occupancy 低是**为大 tile 付的钱**,是设计不是缺陷(§3.4.4)。而且它是可推的:
   170 regs × 256 线程 = 43520 > 65536/2 → 1 CTA/SM → 8/48 = 17%。
8. **Q:FP8 的两个 scale 为什么能提到 dot 外面?**
   组内 scale 与求和下标 $k$ 无关,有限和满足 $\sum_k(c x_k) = c\sum_k x_k$
   (§3.6.2)。前提是 BLOCK_K 与缩放组硬对齐(都是 128),BLOCK_N 与权重块对齐使
   $s^B$ 退化成标量。**这两条是正确性前提,不是调优旋钮。**
9. **Q:为什么 fp8 版不能用 `tl.dot(a, b, acc)`?**
   因为本组结果要先乘 $s^As^B$ 才能并入总累加器,累加器不能直接交给 mma 指令
   (第 7 段)。代价可算:缩放链约占 tensor core 时间的 4.7%(§3.6.3);
   更大的嫌疑是 `part` 与 `acc` 两个 fp32 tile 同时活着,寄存器压力翻倍——
   **这条是假设,验证手段现成但本仓未做。**
10. **Q:DeepGEMM 为什么不能在 4090 上跑?**
    因为它要求 SM90 或 SM100(README 的 requirements),依赖 wgmma(异步 warpgroup
    矩阵指令)与 TMA(张量批搬运),sm_89 两样都没有。**但注意:不是因为"Hopper 有
    原生 scale 槽"**——那个说法不成立(§3.6.4),硬件块缩放要到 Blackwell 的
    `tcgen05.mma`。缩放**代数**可以搬(本仓搬了),**指令世代**搬不了。
11. **Q:那 Hopper 上 DeepGEMM 的细粒度缩放是怎么做的?**
    和本仓一样在 CUDA core 侧做:DeepSeek-V3 报告 §3.3 写 H800 的 FP8 tensor core
    累加精度约 14 位,所以每 $N_C=128$ 个元素的 MMA 之后 "promotion to CUDA Cores"
    做高精度累加。**同一件事,两代硬件都得做。**
12. **压力问 Q:你说"打平 cuBLAS",是不是挑了对自己有利的形状?**
    诚实答:是**只在两个形状上测过**,而且现在能说出舒适在哪三条上:
    ①K 维长、操作数进得了 L2(67 MB vs 72 MB);②CTA 数恰好是 128 的整数倍,
    无尾波(§3.1.2);③M、N、K 全部是 tile 尺寸的整数倍,无 tile quantization。
    没有做全形状扫描,也没测小 M / 瘦长 / 非对齐形状(EXP-T02 §7 明写)。
    cuBLAS 的优势恰恰在于**形状覆盖面**——它对每个形状族都有调好的 kernel,
    而本仓只有一套 tile。所以正确的说法是"在这两个形状上打平/略胜",
    "打平 cuBLAS"作为一般性结论**不成立**。
13. **压力问 Q:228 TFLOPS 的 FP8,能给推理带来 1.5 倍吗?**
    不能。228 是**预量化孤立 GEMM**;同一份代码在线量化端到端只有 72.9(§5.3)。要在
    真实 serving 里兑现,得满足三个条件:权重离线量化(本仓 `quant_fp8_block` 正是
    "离线一次"的定位)、激活量化融合进上游算子(本仓**未做**,EXP-T06 §7)、以及模型
    本身对 3.6e-2 量级的量化误差可接受(本仓只测了误差,没测下游任务指标)。三条缺
    一条,1.5× 就落不了地。
14. **压力问 Q:你把自己以前的解释推翻了,那你现在这个解释又能信多久?**
    诚实答:能信到下一个能证伪它的实验为止,而且我可以现在就说出那个实验是什么——
    dump TTGIR/PTX 数 `cp.async` 的 commit/wait 组数(验证 §3.3.3 的机制)、
    把探针指向 fp8 kernel 读 `n_regs`(验证 §3.6.3 的假设)、跑 $N_s=5$
    (验证 §3.4.1 的超限推断)。**一个结论的可信度,等于它附带的证伪路径有多具体。**
    本篇每一条推断都配了这样一条路径,这是它相对"缩放乘法占算力"那种笼统解释的
    全部进步。

## 8. 工业对照与延伸

### 8.1 论文/文档怎么说 vs 本项目实测:逐条对照

| # | 来源与声称 | 本仓实测(EXP 锚) | 差异分析 |
|---|---|---|---|
| 1 | Triton 文档 `tl.range`:"pipeline the loop into this many stages (so there are `num_stages` iterations of the loop in flight at once)" | EXP-T08:缓冲份数 = $\max(1, N_s-1)$;$N_s$=2 时 `metadata.shared` 与 $N_s$=1 完全相同 | **不是文档错,是两个量**:在飞迭代数 vs 预取缓冲份数,差 1 是流水结构的正常结果(§3.3.3)。**本仓过去把两者当成一个,这是被本记录纠正的自家错误** |
| 2 | Triton 文档 `triton.Config`:num_stages "Mostly useful for matrix multiplication workloads on SM80+ GPUs" | 本仓两个 kernel(GEMM 与 FA2)上都生效,且映射一致 | 一致。补充信息:同一文档指出 kernel 参数版本"only pipelines loads that feed into `dot` operations",本仓两个 kernel 的 load 都喂给 dot,所以适用 |
| 3 | Triton 论文(MAPL '19)§5.1.1:Pre-Fetching 是机器无关 pass;§5.2.3:shared memory 按 live range 做线性时间分配 | 本仓只能观测分配结果(`metadata.shared`),看不到 live range 分析过程 | 论文描述的是初代 Triton-C/Triton-IR 架构,本机 3.6.0 已换成 MLIR 架构,**pass 名与实现都已改写**;论文能提供的是设计意图而非当前实现细节。引用时须注明代际 |
| 4 | Triton 论文 §6.1:"Triton and cuBLAS are generally on par with each other, and achieve more than 90% of the device's peak performance on certain tasks" | 本仓 square4k:159.4±1.2 vs cuBLAS 160.0±0.7(打平),对 165.2 峰值 97% | **结论方向一致**,但论文的实验机器是 GTX 1070、对照是 cuBLAS 10.0,与 4090 + 现代 cuBLASLt 完全不可比。论文同时指出 cuBLAS 在浅层 transformer 形状上因 3D(split-K)算法仍占优——**本仓没有 split-K,这是形状覆盖面差距的一个具体名字** |
| 5 | Ada Tuning Guide §1.4.1.1:"The maximum shared memory per thread block is 99 KB" | EXP-T08 逐字复现 `Required 163840, Hardware limit 101376`,101376 = 99×1024 | **文档 → 实测完全闭合**,本篇唯一一条严格的"预言—证实"链 |
| 6 | Ada 白皮书 Table 2:Peak FP8 Tensor TFLOPS with FP32 Accumulate 330.3(非稀疏) | 本仓 fp8 prequant 228.1±1.3 = **69%**;kperf 口头读数"~70% fp8 峰值" | **两条独立来源对上**。注意白皮书脚注 4 说明这一格在 v2.02 里被改成"proper number"——**厂商规格表也会修订,引用要带版本** |
| 7 | DeepSeek-V3 报告 §3.3:激活 1×128 tile、权重 128×128 block 缩放;E4M3 全用 | 本仓原样实现(src/fp8_gemm.py:38-62) | **代数完全一致**。差异只在硬件:报告是 H800 训练场景,本仓是 4090 推理场景、只做前向 GEMM |
| 8 | DeepSeek-V3 报告 §3.3:H800 的 FP8 tensor core 累加精度约 14 位,故 "promotion to CUDA Cores" 每 $N_C=128$ 元素做一次 | 本仓在 Ada 上做同样的二级累加(src/fp8_gemm.py:111-113) | **同一件事**。这条同时否证了"Hopper 有原生 scale 槽所以不用手乘"的说法(§3.6.4) |
| 9 | CUTLASS wgmma 说明:"scale_D is either 0 or 1, and controls whether or not the accumulator is zero-initialized";"scaleA and scaleB are either 1 or −1 for negating the operand" | 本仓无 Hopper 硬件,**无法实测** | **纯文档核实**,用来纠正本仓 docs/theory/06 与 src/fp8_gemm.py docstring 里"wgmma 原生 scale 槽"的表述。标注:未在硬件上验证,依据是 ISA/库文档语义 |
| 10 | CUTLASS Blackwell 说明:`tcgen05.mma` 的 mx/nvf4 kind 执行 $D=C+(A\times SFA)(B\times SFB)$,scale factor 按 K 维每 16 或 32 元素一个 | 不适用(无 Blackwell 硬件) | 用来给出"硬件块缩放在哪一代出现"的准确答案。**本仓不对 Blackwell 做任何性能主张** |
| 11 | DeepGEMM README:要求 "NVIDIA SM90 or SM100 architecture GPU" | 本仓 sm_89,**跑不了** | 直接支持"为什么不能拿 DeepGEMM 跑 4090"。注意 README 已从早期的 Hopper-only 扩到 SM90/SM100,**"Hopper-only"这个说法本身已经过时**,准确说法是"不含 sm_89" |
| 12 | Triton `tl.dot_scaled` 文档:无原生 microscaling 支持的架构上,"microscaled lhs/rhs are upcasted to `bf16` element type beforehand" | 本仓未用 `dot_scaled`,手写累加器侧缩放 | 旁证:**编译器给的替代路也是放弃 fp8 吞吐**,所以手写是这一代的合理形态(§3.6.5) |
| 13 | NVIDIA 矩阵乘指南 §3.2 Wave Quantization:tile 数应是 SM 数的整数倍 | square4k 8.0 波、qwen8b 12.0 波,**都恰好整波** | 这是"打平 cuBLAS"的一个未被明说的舒适条件(§3.1.2)。**本仓未测非整波形状**,推断 |
| 14 | Triton 3.6.0 源码 `min_dot_size` 返回 `(1,1,16)`,注释 "For small M/N the input we can still use tensorcores with padding" | 本仓源码注释表述为"要求 M≥16"(讲义 01 §3.7.3 已修正) | M=1 不会编译失败,只是被 pad。**代价的结论不变,机制的表述以源码为准** |

**总结这张表的读法**:十四条里只有第 5、6 两条是严格闭合的"文档 → 实测";第 1、9、
11、14 是**文档/源码语义纠正了本仓过去的表述**;第 3、4 是"同一件事但代际不可比";
其余是"代数一致、场景不同"。**这一批纠正没有改动任何一个性能数字**——它们改的全是
对数字的解释,而这恰恰是最容易出错、也最少被检查的一层。

### 8.2 与生产实现的差距各在哪一层

- **cuBLAS / CUTLASS**:同样的 tile + 流水思想,但它们按形状族预置了几十套 kernel 与
  启发式选择器,并做了 split-K、stream-K 等本仓没有的负载均衡策略。Triton 论文 §6.1
  就点名过这条("CuBLAS, however, remains faster than Triton on shallow transformer
  neural networks thanks to the use of a 3D algorithm which splits deep reductions
  into independent chunks")。本仓是"一套 tile 打两个形状",差距在**覆盖面**而不是
  单点峰值(§7 第 12 问)。
- **Hopper 的 wgmma + TMA**:把"预取"从 cp.async 的显式流水升级成硬件描述符驱动的
  批搬运,异步矩阵指令自带流水语义与 warpgroup 级同步(`wgmma.commit_group` /
  `wgmma.wait_group`)。这是 DeepGEMM 能做到本仓做不到那部分的硬件根源
  ——**但不包括"原生 scale 槽",见 §3.6.4**。
- **Blackwell 的 tcgen05.mma**:块缩放进硬件($D=C+(A\cdot SFA)(B\cdot SFB)$),
  §3.6.3 那条 4.7% 的缩放链与 §3.6.3 那个"两个 fp32 tile 同时活着"的寄存器压力假设
  在这一代**都会消失**。本仓无该硬件,不做任何性能主张。
- **推理引擎里的 GEMM**:真实 serving 的瓶颈形状是 decode 的 $M=1$(第 4 段的 linear
  自适应),以及"量化 + GEMM + epilogue"的融合边界——本仓把量化留在 torch 侧,这正是
  §5.3 那个 72.9 的来源;生产实现会把激活量化融进上一个算子的 epilogue,顺手把
  per-token scale 写出去(§4 第 5 段的存储账说明了为什么值得这么做)。
- **接口层的隐藏拷贝**:本仓 `linear` 每次都做 `weight.t().contiguous()`(第 4 段),
  生产实现会在加载时就把权重存成需要的布局。**这类成本不在任何 kernel 数字里**,
  只在端到端里显形。

### 8.3 这一篇没做的事(供下一步)

按"验证成本从低到高"排:

1. 跑 $N_s=5$,验证 §3.4.1 的"128 KB 超限"推断(改一个数字,秒级)。
2. 把 EXP-T08 探针指向 `_fp8_gemm_kernel`,读 `n_regs` / `n_spills`,验证 §3.6.3 的
   寄存器压力假设(改三行脚本)。
3. dump TTGIR/PTX,数 `cp.async` 的 commit/wait 组数,验证 §3.3.3 的机制
   (EXP-T08 §7 已列)。
4. 扫 GROUP_M 与 num_warps(§3.8 里唯二标"未扫参"的旋钮)。
5. 测一个非整波、非整除 tile 的形状,验证 §3.1.2 与 §7 第 12 问里那三条舒适条件。
6. 把激活量化融进上游算子的 epilogue,把 §5.3 第 3 行那个 72.9 变成有意义的数
   (EXP-T06 §7 已列)。

### 8.4 延伸阅读(带精确出处,每条一句话说明它能解决什么疑问)

**论文**

1. Tillet, Kung, Cox, "Triton: An Intermediate Language and Compiler for Tiled
   Neural Network Computations", MAPL '19, DOI:10.1145/3315508.3329973,
   §5.1.1 Pre-Fetching、§5.2.3 Shared Memory Allocation、§5.2.4 Shared Memory
   Synchronization、§6.1。——想知道"编译器凭什么能替你排流水""shared memory 份数
   是怎么被算出来的""cuBLAS 在哪类形状上仍然赢",读这四处;同时注意它描述的是初代
   Triton-C/Triton-IR 架构,与本机 3.6.0 的 MLIR 架构已隔一代。
2. DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv:2412.19437,§3.3 FP8 Training
   (Fine-Grained Quantization、Increasing Accumulation Precision)。——想核实
   "1×128 激活 / 128×128 权重"这套缩放布局的原始表述,以及"H800 的 FP8 累加只有
   约 14 位、所以要 promotion to CUDA Cores"这条关键事实,读这一节。
3. Williams, Waterman & Patterson, "Roofline: an insightful visual performance model
   for multicore architectures", CACM 52(4):65-76, DOI:10.1145/1498765.1498785。
   ——想把 §2.1 的三层算术强度放进一个统一框架,读它。

**官方文档**

4. NVIDIA, "NVIDIA Ada GPU Architecture" 白皮书,Appendix A Table 2 与
   "Ada GPU Architecture In-Depth" 的 SM / Memory Subsystem 段。——所有硬件常数
   (128 SM、1008 GB/s、73728 KB L2、165.2 / 330.3 TFLOPS、每分区一个 warp scheduler
   加一个 dispatch unit)的唯一出处;§3.4.4 与 §3.7.4 全靠它。注意脚注 2(稀疏)
   与脚注 4(FP8/FP32 累加那格被修订过)。
5. NVIDIA, "Ada Tuning Guide"(docs.nvidia.com/cuda/ada-tuning-guide),
   §1.4.1.1 Occupancy、§1.4.2.2 Unified Shared Memory/L1/Texture Cache。——
   99 KB / 100 KB / 48 warps / 255 regs / 24 blocks 这五个上限的官方原文,以及
   L1-shared 可配置的 carveout 档位(0/8/16/32/64/100 KB)与 48 KB 以上需 opt-in。
6. NVIDIA PTX ISA,§9.7.9.26 Asynchronous copy。——`cp.async` 的 ca/cg 修饰符与
   cp-size 只能取 4/8/16 字节(cg 必须 16)、`commit_group` 的"成组提交"、
   `wait_group N` 的"只剩 N 组在飞时返回"。§3.3.3 与 §3.7.1 的依据。
7. NVIDIA PTX ISA,§9.7.15(mma,warp 级集合;m16n8k16 的 fragment 布局)与
   §9.7.16(wgmma,warpgroup 级异步)。——想弄清"为什么不能在 dot 周围写发散分支"
   与"wgmma 的三个 scale 操作数到底是什么",读这两章。
8. NVIDIA CUTLASS 文档:Warpgroup MMA Programming Guide(scale_D ∈ {0,1}、
   scaleA/scaleB ∈ {1,−1}、matrix descriptor、commit/wait 模型)与 Blackwell
   functionality(`tcgen05.mma` 的 mxf8f6f4/mxf4/nvf4 kind 与 $D=C+(A\cdot SFA)(B\cdot SFB)$)。
   ——§3.6.4 那条纠正的直接依据;想一次看清三代块缩放的分界,读这两处。
9. NVIDIA, "Matrix Multiplication Background User's Guide":"Arithmetic Intensity"、
   §3.1 Tile Quantization、§3.2 Wave Quantization。——$MNK/(MK+NK+MN)$ 这个公式的
   出处,以及"CTA 数不是 SM 数整数倍会出尾波"的定义(§3.1.2 的依据)。
10. NVIDIA, "GPU Performance Background User's Guide" §4 Understanding Performance。
    ——ops:byte 的官方定义("the ratio of a processor's math and memory bandwidths")
    与三个限制因子的原话。
11. Triton 官方 API 文档与本机 3.6.0 源码:`triton.Config`(runtime/autotuner.py)、
    `tl.range`(language/core.py)、`tl.dot_scaled`(同上)、
    `min_dot_size`(backends/nvidia/compiler.py)、`OutOfResources` 的报错文本
    (runtime/errors.py)。——想核实"num_stages 到底承诺了什么""dot 的真实形状下界
    是多少""那句 `Reducing block sizes or num_stages may help` 从哪来",读这五处。
12. DeepGEMM 仓库 README(github.com/deepseek-ai/DeepGEMM)。——requirements 里
    "NVIDIA SM90 or SM100 architecture GPU" 这一行,是"为什么 4090 跑不了"的
    第一手依据;JIT 设计与 TMA 用法也在同一页。

**源码与本仓证据**

13. src/gemm_pipelined.py:61-79 —— 主循环与 num_stages 的注释(**注意其中两处需按
    §3.3.2 / §3.4.3 修正**,对读能看出"解释比代码更容易过时")。
14. src/fp8_gemm.py:96-115 —— 缩放组主循环与二级累加(与上一条对读)。
15. scripts/probe_smem_regs.py:26-40 —— 编译期资源探针的全部机制,含那行关键的
    `device_caches[dev][0].clear()`。
16. records/EXP-T08_smem_stage_probe.md §1-§7 —— 假设与证伪条件的锁定、六格逐格
    吻合的表、以及三条开放问题(未 dump PTX / 未扫 num_warps / 寄存器非单调未解释)。
17. records/EXP-T02_gemm_pipeline.md §5-§7 —— 存盘轮口径、首轮数字作废的处理,
    以及"最优 stage 随形状摇摆"的结论边界。
18. records/EXP-T06_fp8_gemm.md §5-§7 —— 四口径表、kperf 三卡观测的终端级证据登记,
    以及 BLOCK_N 硬绑 128 这个开放项。
19. docs/theory/02_double_buffering.md §2 —— CUDA 手写双缓冲伪码与 Triton 版的对照
    (**其中"分配 N 份 smem 缓冲"一句按 EXP-T08 修正为 $\max(1,N-1)$ 份**)。
20. docs/theory/06_fp8_gemm_ada.md §2 —— Ada/Hopper 界线表(**其中"wgmma 原生
    scale 槽"一格按 §3.6.4 修正**)。
