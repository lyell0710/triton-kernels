# 讲义 01 · 从 softmax 的减 max 推到 FlashAttention-2,再推到 flash-decoding

> 读者：准备校招面试的作者本人，以及第一次动手写 attention kernel 的工程师。读法：不跳步。每个论断后面都跟着它的证据锚（EXP 编号 / 文件：行号 / raw 路径）；所有数字与仓内现行口径逐字一致，来源见 records/ 与 data/derived/。引用规范：凡属论文/官方文档的论断，给出标题 + arXiv/DOI 编号 + 章节或定理编号（文档给 URL 路径 + 小节名）；凡属本讲义补出的推导或折算，行内标注「本讲义推导」；无法用检索确认的说法标注「未核实」。

## 目录

- [1. 这一篇回答什么问题](#1-这一篇回答什么问题)
  - [1.1 本篇要建立的五条能力](#11-本篇要建立的五条能力)
  - [1.2 符号与口径约定](#12-符号与口径约定)
  - [1.3 本篇引用的一级文献(详细出处见 §8.4)](#13-本篇引用的一级文献详细出处见-84)
- [2. 直觉与第一性原理](#2-直觉与第一性原理)
  - [2.1 没有 FlashAttention 的世界:一笔可以心算的账](#21-没有-flashattention-的世界一笔可以心算的账)
  - [2.2 为什么"不物化"是可能的](#22-为什么不物化是可能的)
  - [2.3 三条贯穿全篇的公理](#23-三条贯穿全篇的公理)
  - [2.4 为什么选"减 max"而不是"换更宽的浮点"](#24-为什么选减-max而不是换更宽的浮点)
- [3. 完整推导与机制](#3-完整推导与机制)
  - [3.1 减 max 的必要性:精确恒等式,不是技巧](#31-减-max-的必要性精确恒等式不是技巧)
  - [3.2 分块可归并性:换基引理与合并算子](#32-分块可归并性换基引理与合并算子)
  - [3.3 IO 复杂度定理:FlashAttention 到底证明了什么](#33-io-复杂度定理flashattention-到底证明了什么)
  - [3.4 FA1 → FA2:循环反转到底省了什么](#34-fa1--fa2循环反转到底省了什么)
  - [3.5 GQA 头映射:一行寻址换来的一半带宽](#35-gqa-头映射一行寻址换来的一半带宽)
  - [3.6 decode 的并行度塌陷:为什么必须换一根并行轴](#36-decode-的并行度塌陷为什么必须换一根并行轴)
  - [3.7 硬件语义层:这些代码写法是被什么约束定死的](#37-硬件语义层这些代码写法是被什么约束定死的)
  - [3.8 每个魔法数的来历(汇总)](#38-每个魔法数的来历汇总)
- [4. 代码逐段走读:src/fa2_fwd.py 全核 + src/flash_decode.py 两段式](#4-代码逐段走读srcfa2_fwdpy-全核--srcflash_decodepy-两段式)
- [5. 实验数据怎么读](#5-实验数据怎么读)
  - [5.1 fig1:FA2 vs SDPA-flash](#51-fig1fa2-vs-sdpa-flash)
  - [5.2 flash-decoding 的三臂表:2.24× 与 5.17× 差在哪](#52-flash-decoding-的三臂表224-与-517-差在哪)
  - [5.3 哪些数字能外推,哪些不能](#53-哪些数字能外推哪些不能)
- [6. 误区与边界](#6-误区与边界)
- [7. 连环追问](#7-连环追问)
- [8. 工业对照与延伸](#8-工业对照与延伸)
  - [8.1 论文/文档怎么说 vs 本项目实测:逐条对照](#81-论文文档怎么说-vs-本项目实测逐条对照)
  - [8.2 与生产实现的差距各在哪一层](#82-与生产实现的差距各在哪一层)
  - [8.3 这一篇没做的事(供下一步)](#83-这一篇没做的事供下一步)
  - [8.4 延伸阅读(带精确出处,每条一句话说明它能解决什么疑问)](#84-延伸阅读带精确出处每条一句话说明它能解决什么疑问)

## 1. 这一篇回答什么问题

一个 attention kernel 从「数值稳定的 softmax」一路长成「不物化 S×S 的 FA2」，再长成「decode 专用的 split-K flash-decoding」，中间每一步都有一个可以手推的理由。读完你应当能：

- 手推三件事：①softmax 为什么必须减 max、减完为什么**精确**等价而不是近似；②分块 softmax 的可归并性代数（换基引理 + 合并公式），以及同一套代数为什么能既做「块内 online」又做「块间 reduce」；③FA1→FA2 的循环反转为什么能把 $(m, l, \mathrm{acc})$ 整段留在寄存器里。
- 说清「87%」的**四要素限定**：简化版、仅 forward、4K 形状（B1·H32/8·D128）、对照 = SDPA flash 后端——缺一不引（本仓措辞约定）；并把差掉的那部分拆到「算力利用率」这一层。
- 答上「flash-decoding 到底快几倍」：**2.24±0.11×（naive repeat 预置口径）/ 5.17±0.24×（含 repeat 实体化口径）**，32K 上下文（EXP-T04《Flash-Decoding》），并主动说出反面——**Skv ≤ 8K 反而只有 0.86-0.88×**，以及这个反亏为什么与算法无关。

### 1.1 本篇要建立的五条能力

1. **代数能力**：能把 online softmax 写成一个**幺半群**（带单位元的结合运算），并当场证明「块内顺扫」与「块间树归」给出同一个数学结果；知道这条性质在哪一步会失效（先除 $l$ 就失效）。
2. **复杂度能力**：能准确复述 FlashAttention 的 IO 复杂度定理(Θ(N²d²M⁻¹)) 与它的下界命题，知道定理里的 $M$ 指什么、上下界的适用区间 $[d, Nd]$ 是什么意思， 以及**定理没有承诺什么**（它不承诺时间，只承诺 HBM 访问量的阶）。
3. **硬件语义能力**：能说出 Ada 每 thread block 的 shared memory 上限是多少、这个上限如何把 BLOCK_N=128 这一档直接判死；能解释 cp.async 的 commit/wait **组**语义与 mma 的 warp 级集合语义如何决定 Triton 的代码形态。
4. **口径能力**：任何速度数字出口都带形状、dtype、对照物与轮数；知道 2.24× 与 5.17× 不是"哪个更真"，而是两个不同的对照物定义。
5. **归因能力**：看到一个"kernel 更快但端到端更慢"或"短上下文反亏"的现象， 能把它拆到设备侧 / 主机侧 / 对照物口径三层里的某一层，而不是笼统说"优化不够"。

### 1.2 符号与口径约定

| 符号 | 含义 | 本仓 bench 取值 |
|---|---|---|
| B | batch | 1（EXP-T01 与 EXP-T04 协议） |
| $H_q$ / $H_{kv}$ | Q 头数 / KV 头数 | T01:32 / 8;T04:16 / 8 |
| G | GQA 组大小 $H_q/H_{kv}$ | T01:4;T04:2 |
| S / $S_{kv}$ | 序列长 / decode 的 KV 长度 | T01:512–4096;T04:512–32768 |
| D | head_dim | 128（部分正确性格用 64） |
| BM / BN | Q 行块 / KV 列块的 tile 大小 | fp16 默认 128 / 64 |
| $m_i, l_i, \mathrm{acc}$ | online softmax 的三个跑动量 | 全 fp32 |
| $\alpha$ | 换基因子 $e^{m_{\text{old}}-m_{\text{new}}}$ | $\in(0,1]$ |
| M（定理里） | 片上 SRAM 容量（元素数） | 见 §3.3.1 |
| $\pi_{\text{mem}}$ | HBM 带宽 | 1008 GB/s（Ada 白皮书 Table 2） |
| $\pi_{\text{math}}$ | fp16 tensor core / fp32 累加峰值 | 165.2 TFLOPS（同上，非稀疏） |

硬件常数的唯一出处：NVIDIA, "NVIDIA Ada GPU Architecture" 白皮书， Appendix A Table 2（GeForce RTX 4090:SMs 128、Memory Bandwidth 1008 GB/sec、 L2 Cache Size 73728 KB、Register File Size 32768 KB、L1 Data Cache/Shared Memory 16384 KB、Peak FP16 Tensor TFLOPS with FP32 Accumulate 165.2/330.4，脚注 2 说明第二个数是"Effective TOPS / TFLOPS using the new Sparsity Feature"，本仓一律用非稀疏那一个）。占用率相关的上限出自 NVIDIA, "NVIDIA Ada GPU Architecture Compatibility Guide / Tuning Guide"(docs.nvidia.com/cuda/ada-tuning-guide), §1.4.1.1 Occupancy:"The maximum number of concurrent warps per SM is 48"、 "The register file size is 64K 32-bit registers per SM"、"The maximum number of registers per thread is 255"、"The maximum number of thread blocks per SM is 24"、 "The shared memory capacity per SM is 100 KB"、"The maximum shared memory per thread block is 99 KB";§1.4.2.2:"The combined L1 cache capacity is 128 KB"。

### 1.3 本篇引用的一级文献(详细出处见 §8.4)

- online softmax 递推式：Milakov & Gimelshein, "Online normalizer calculation for softmax", arXiv:1805.02867,Algorithm 3 与 Theorem 1。
- FlashAttention（IO 复杂度定理）:Dao, Fu, Ermon, Rudra, Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", arXiv:2205.14135, Theorem 1/Theorem 2/Proposition 3、Algorithm 1。
- FlashAttention-2（循环反转与三处改动）:Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", arXiv:2307.08691,§3.1 Algorithm、 §3.2 Parallelism、§3.3 Work Partitioning Between Warps。
- 惰性 softmax 的前身：Rabe & Staats, "Self-attention Does Not Need $O(n^2)$ Memory", arXiv:2112.05682。
- flash-decoding:Dao, Haziza, Massa, Sizov, "Flash-Decoding for long-context inference"（PyTorch 官方博客 pytorch.org/blog/flash-decoding/）。
- 缩放点积注意力：Vaswani et al., "Attention Is All You Need", arXiv:1706.03762,§3.2.1。
- GQA:Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", arXiv:2305.13245。
- 硬件语义：NVIDIA PTX ISA(§9.7.9.26 Asynchronous copy、§9.7.15 Warp Level Matrix Multiply-Accumulate)、Ada Tuning Guide §1.4、Ada 白皮书 Appendix A。
- 性能模型：NVIDIA, "GPU Performance Background User's Guide", §4 Understanding Performance;Williams, Waterman & Patterson, "Roofline: an insightful visual performance model for multicore architectures", CACM 52(4):65-76, DOI:10.1145/1498765.1498785。

## 2. 直觉与第一性原理

### 2.1 没有 FlashAttention 的世界:一笔可以心算的账

$O = \mathrm{softmax}(QK^\top/\sqrt{d})V$ 照字面写，中间必然出现一个 $S\times S$ 分数矩阵。本仓 bench 形状 B=1、$H_q$=32、S=4096、fp16 下它是 $4096^2\times32\times2\,\mathrm{B}=1.07\,\mathrm{GB}$，一次前向至少写一遍、读一遍、写回一遍、再读一遍，约 $3.2\,\mathrm{GB}$ 的 HBM 往返，在 4090 的 1008 GB/s 上就是 $3.2\,\mathrm{ms}$；而算力账只有 $2BH_qS^2D = 137.4\,\mathrm{GFLOP}$（causal 折半后）， 在 165 TFLOPS 的 fp16 tensor core 上是 $0.83\,\mathrm{ms}$。**物化 $S\times S$ 把一个本该 compute-bound 的算子按成 memory-bound，而且按得很深。**

这笔账的每一步都要能自证：

- $S\times S$ 的字节数：$4096^2 = 16{,}777{,}216$ 个元素，×32 头 ×2 B(fp16) $= 1{,}073{,}741{,}824\,\mathrm{B}$，即 1.07 GB。
- "至少四趟"的来历：一个朴素实现是 `s = q@k.T*scale`（写 1.07 GB）→ `p = softmax(s)`（读 1.07 GB、写 1.07 GB；若 softmax 自身分三趟还要更多）→ `o = p@v`（读 1.07 GB）。合计 3–4 份 1.07 GB，取 3 份即 3.2 GB。这是**下界**， 真实框架实现通常更多。
- 为什么"进不了片上"：Ada 每个 thread block 最多能拿 99 KB shared memory (Ada Tuning Guide §1.4.1.1)，整卡 128 个 SM × 100 KB = 12.8 MB；1.07 GB 比它大两个数量级，**没有任何调度能把它留在片上**，只能走 HBM。

仓内量尺（EXP-T01《Triton FA2 forward》，3 轮）：S=2048 上 naive fp32 参考 $6.694\pm0.003\,\mathrm{ms}$，本仓 kernel $0.3296\pm0.0007\,\mathrm{ms}$(data/derived/exp-t01_stability_3rounds.csv)—— 20 倍的差距里算法一个 FLOP 都没少，少的全是那趟 HBM 往返。

### 2.2 为什么"不物化"是可能的

softmax 的分母是求和，天生可分块累加；唯一障碍是分子里的 $e^{x-\max}$ 需要"整行的 max"，而分块时你只有"到目前为止的 max"。FA 的全部魔法就是：**旧 max 算出来的东西乘一个标量就能换到新 max 的基准上**（§3.2 换基引理）。

这不是 FlashAttention 首创。学术脉络上有两个前身，值得各记一句：

1. Milakov & Gimelshein 在 arXiv:1805.02867 里把"求 max"和"求指数和"合成一趟（Algorithm 3），并用 Theorem 1 的归纳法证明它与三趟版给出**同一个** $d_V$。他们的动机是纯访存：摘要写"we propose a way to compute classical Softmax with fewer memory accesses"，实测"Softmax accelerates by up to 1.3x and Softmax+TopK combined and fused by up to 5x"。
2. Rabe & Staats 在 arXiv:2112.05682("Self-attention Does Not Need $O(n^2)$ Memory")把同一招用到 attention 上，维护"an unnormalized running sum $v^*$ and running normalizer $s^*$"，得到单 query $O(1)$、self-attention $O(\log n)$ 的 **内存**结果。

FlashAttention 相对这两者的增量，不在代数而在**目标函数**：前两者优化的是内存占用与访存次数，FlashAttention 把目标明确成"HBM 访问量"，并给出了带下界的复杂度定理（§3.3）。**同一个代数，换一个目标函数，就换出一篇不同的论文**——这是读这条线索时最值得学的一点。

### 2.3 三条贯穿全篇的公理

- **公理 A（可归并性先于性能）**：任何"分块 + 合并"的 kernel，先证明合并算子是结合的、有单位元，再谈 tile 怎么调。本仓 FA2 与 flash-decoding 共用同一个合并算子（§3.2），这不是巧合，是因为先把代数写清楚了。
- **公理 B（数字必须带对照物定义）**："快 N 倍"里的 N 由分母决定。EXP-T04 的 2.24× 与 5.17× 是同一批数据、同一个分子、两个分母（§5.2），二选一就是造假。
- **公理 C（片上资源是硬墙，不是软约束）**：tile 大小、流水深度、occupancy 共用一份 99 KB / 64K 寄存器的预算。撞墙的表现不是"慢一点"，是**编译期直接 OutOfResources**（EXP-T08《num_stages 与 shared memory 份数的映射》逐字复现）。

### 2.4 为什么选"减 max"而不是"换更宽的浮点"

换宽只把溢出点往后推，不改变"存在会炸的输入"这个事实；减 max 把指数参数压到 $\le 0$，**无条件**安全且不花额外带宽。三个溢出线可以当场算给面试官看（本讲义推导）：

| dtype | 最大有限值 | $e^x$ 上溢阈值 $\ln(\text{max})$ |
|---|---|---|
| fp16 | 65504 | 11.09 |
| bf16 | ≈ 3.39e38 | 88.72 |
| fp32 | ≈ 3.40e38 | 88.72 |

bf16 与 fp32 的阈值几乎相同，因为两者指数位都是 8 位；bf16 省的是尾数不是范围。所以"用 bf16 就不会溢出了"这句话在 attention 上**碰巧接近对**，但理由是指数位数， 不是精度——而 fp16 的 11.09 在真实 attention logits 上是随手就能越过的。

优先选把问题消掉的变换，不是推远的变换：这条选择标准在本篇会反复出现（§3.2 的"最后再除"、§3.6 的"两个 kernel 而非原子加"，都是同一条标准的应用）。

**日常类比与它的失效点**：像连续称重时用"目前最重的一件"当基准记录相对重量， 来了更重的就把之前所有记录统一乘一个折算系数。类比在两处失效：

1. 类比里基准换晚了只是记录难看，attention 里换晚了是 $e^x$ **溢出成 inf**、接着 $\mathrm{inf}/\mathrm{inf}=\mathrm{nan}$ 污染整行——"结果作废"而非"精度差一点"。
2. 类比里的合并是普通加法（可交换、可结合、可原子累加），softmax 的合并是**带换基因子的加法**——每项在加之前都要乘一个依赖全局 max 的标量。这正是 flash-decoding 不能用 atomicAdd 归并、必须拆两个 kernel 的根本原因（§3.6）。

## 3. 完整推导与机制

### 3.1 减 max 的必要性:精确恒等式,不是技巧

#### 3.1.1 四步推导,每步写清"凭什么可以这么做"

1. 定义 $\mathrm{softmax}(x)_i = e^{x_i}/\sum_j e^{x_j}$——这是定义，没有自由度。
2. 对任意常数 $c$，分子分母同乘 $e^{-c}$： $$\frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i - c}}{\sum_j e^{x_j - c}}$$——凭的是 $e^{a+b} = e^a e^b$ 与"分子分母同乘非零数不改变商"。这是**恒等式**， 对任何 $c$ 都精确成立，一位有效数字都不损失。 **成立条件**：只需要 $e^{-c}\neq 0$ 且分母非零；在实数域上无条件成立，在浮点上要求 $e^{-c}$ 本身不下溢成 0（取 $c=\max$ 时 $e^{x_i-c}\le 1$，天然安全）。
3. 于是 $c$ 成了自由参数，选它的标准只剩数值范围。取 $c=\max_j x_j$ 时所有 $x_i-c\le0$，故 $e^{x_i-c}\in(0,1]$ **永不上溢**；且至少有一项等于 1，分母 $\ge1$，**永不下溢成 0**。比 max 小的 $c$ 可能上溢，大很多的 $c$ 会让整行下溢—— max 是同时把两侧余量做到最大的那个选择。
4. 代价：要拿到 $\max_j x_j$ 就得先扫一遍整行 → 朴素实现是**三趟**（求 max、求 $\sum e$、归一化）。online softmax 把三趟压成**一趟**，代价是每一步维护"当前基准" 并在基准变化时校正历史。这就是 §3.1.3 与 §3.2。

#### 3.1.2 第 3 步的"最优性"到底强在哪(本讲义推导)

常见的误解是"减 max 只是保险起见"。把它量化一下：设 $c$ 是任意基准，则

- 上溢条件：$\exists i,\ x_i - c > \ln(\text{max\_finite})$；
- 下溢成 0 的条件：$\forall i,\ x_i - c < \ln(\text{min\_subnormal})$。

取 $c = \max_j x_j$ 后，第一个条件恒假（左边 $\le 0$），第二个条件也恒假（至少有一项等于 0，$e^0=1$）。也就是说 **max 是唯一能同时把两个失效条件都变成恒假的选择族里的中心点**：任何 $c < \max - \ln(\text{max\_finite})$ 都会上溢，任何 $c > \max - \ln(\text{min\_subnormal})$ 都会让整行下溢。可用区间宽度对 fp16 是 $11.09 + 16.6 = 27.7$（$\ln$ 最小次正规数 $\approx -16.6$），取中点与取 $\max$ 的差别只是余量分配，而取 $\max$ 让**上溢余量最大**——上溢产生 inf/nan，下溢只损失被指数压得极小的项，两种失效的危害完全不对称。所以取 max 不是折中，是按危害排序后的最优。

#### 3.1.3 三趟 → 两趟 → 一趟:原论文的递推式

Milakov & Gimelshein(arXiv:1805.02867)把"safe softmax"的朴素实现描述为 "three passes over input vector： The first one calculates the maximum value $m_V$， the second one - normalization term $d_V$， and the third one - final values $y_i$"，按他们的计数是每元素 4 次访存；online 版把前两趟合成一趟，降到每元素 3 次。合成的关键就是那条递推（Algorithm 3）：

$$m_j \leftarrow \max(m_{j-1},\, x_j),\qquad d_j \leftarrow d_{j-1}\, e^{m_{j-1}-m_j} + e^{x_j - m_j}$$

初值 $m_0 = -\infty$、$d_0 = 0$。论文的 Theorem 1 用归纳法证明这条递推终点处 $m_V = \max_k x_k$ 且 $d_V = \sum_j e^{x_j - m_V}$，即**与三趟版逐项相等**。

把归纳法补出来（论文只给结论，这里补中间步，本讲义推导）：

- 归纳假设：$d_{j-1} = \sum_{t\le j-1} e^{x_t - m_{j-1}}$。
- 归纳步： $$d_{j-1}e^{m_{j-1}-m_j} = \sum_{t\le j-1} e^{x_t-m_{j-1}}e^{m_{j-1}-m_j} = \sum_{t\le j-1} e^{x_t-m_j}$$ 凭的是**求和号里每一项乘同一个与 $t$ 无关的常数，可以把常数提进去**（有限和的线性性），再用 $e^ae^b=e^{a+b}$ 合并指数。加上新项 $e^{x_j-m_j}$ 即得 $d_j = \sum_{t\le j} e^{x_t-m_j}$。
- 边界：$j=1$ 时 $d_0 e^{-\infty - m_1} = 0$（即使 $m_1$ 也是 $-\infty$，约定 $0\cdot\text{anything}=0$），退化为 $d_1 = e^{x_1-m_1} = 1$。**这就是代码里 $m_i$ 初值取 $-\infty$ 的全部意义：让首块的换基自动退化成赋值，不需要 if 分支。**

仓内落点：非分块版就是 `x = x - tl.max(x, axis=0)`（src/elementwise_kernels.py：75， 讲义 03 第 1 段走读）；FA2 里它变成跑动版 `m_new = tl.maximum(m_i, tl.max(qk, 1))`(src/fa2_fwd.py：110)。同一条数学，区别只在 "扫一遍就知道 max"还是"边扫边修正 max"。

#### 3.1.4 一个必须分清的边界:online 更省的是访存不是算力

online 版把访存从每元素 4 次降到 3 次，但**多做了指数运算**：每步都要算 $e^{m_{j-1}-m_j}$。在向量 softmax 上这笔交易是否划算取决于形状（论文实测 1.3×）； 在 attention 里它无条件划算，因为省掉的不是"一趟读向量"，而是"一趟读 $S\times S$ 矩阵"——数量级不同。**同一个算法在不同上下文里的收益比可以差两个数量级，这是 "论文数字不能直接搬"的一个干净例子**（§8.1 第 1 条会再引一次）。

### 3.2 分块可归并性:换基引理与合并算子

#### 3.2.1 部分统计量与换基引理

**部分统计量**（第 $p$ 块 K/V，分数 $s_j = q\cdot k_j\cdot\text{scale}$）：

$$m_p = \max_{j\in p} s_j,\qquad l_p = \sum_{j\in p} e^{s_j - m_p},\qquad \mathrm{acc}_p = \sum_{j\in p} e^{s_j - m_p}\, v_j$$

注意 $\mathrm{acc}_p$ **没有除以** $l_p$——这是全部推导的枢纽。

**换基引理**：$e^{s-m'} = e^{s-m}\cdot e^{m-m'}$，凭的还是 $e^{a+b}=e^ae^b$；关键在于因子 $e^{m-m'}$ **与 $j$ 无关**，可以从求和号里提出来——一个标量乘法就把整块历史换到新基准。

引理的**成立条件**要说全：①$m,m'$ 都是有限实数，或 $m=-\infty$ 且约定 $e^{-\infty}=0$；②浮点上要求 $m-m'\le 0$ 才保证 $e^{m-m'}\le 1$ 不上溢——而 $m' = \max(m, \cdot)\ge m$ 由构造保证。**"不会上溢是被结构保证的，不是运气"**， 这句话在 §4 第 5 段还会以代码形态出现一次。

#### 3.2.2 合并算子与三条代数性质

定义 $(m_A,l_A,\mathrm{acc}_A) \oplus (m_B,l_B,\mathrm{acc}_B)$ 为

$$m = \max(m_A, m_B),\quad l = l_A e^{m_A-m} + l_B e^{m_B-m},\quad \mathrm{acc} = \mathrm{acc}_A e^{m_A-m} + \mathrm{acc}_B e^{m_B-m}$$

**正确性验证**（把"代入定义即可验证"这一步真的写出来，本讲义推导）：设 A、B 是不相交的下标集，$m=\max(m_A,m_B)=\max_{j\in A\cup B}s_j$。由换基引理， $l_A e^{m_A-m} = \sum_{j\in A}e^{s_j-m_A}e^{m_A-m} = \sum_{j\in A}e^{s_j-m}$； 对 B 同理。两者相加得 $\sum_{j\in A\cup B}e^{s_j-m}$，这正是"把 $A\cup B$ 当一块直接算"的 $l$。$\mathrm{acc}$ 逐分量重复同一步。**注意这一步只用了有限和的可交换与结合律，没有用任何近似**——所以合并是精确的。

三条性质，每条都在实现里被用到：

- **结合律**：$(x\oplus y)\oplus z = x\oplus(y\oplus z)$。证明要点：两边的 $m$ 都是三者的 max，而三个 acc 项各自被乘上 $e^{m_\bullet - m}$；由 $e^ae^b=e^{a+b}$， 分两步换基与一步换基给出同一个因子。**结论**：分块方式、归并顺序、树形还是线性， 数学结果同一个（浮点舍入除外）——这就是"块内 online 顺扫"与"块间一次性 reduce" 能共用一套代数的原因。
- **交换律**：$\oplus$ 只依赖 $\max$ 与加法，两者都可交换。
- **单位元** $(-\infty, 0, \mathbf{0})$：$e^{-\infty-m}=0$ 使空块贡献自动湮灭， 不需要任何 if 分支——src/flash_decode.py：121-123 直接吃这条性质。

三条合起来：$(\{(m,l,\mathrm{acc})\},\oplus)$ 是一个**交换幺半群**。这是本篇最值钱的一句抽象：任何"可以并行归约"的算子都必须是幺半群，而 attention 的 softmax 能被分块，正因为有人把它写成了幺半群。

#### 3.2.3 "除法必须最后做"不是省除法的小优化

若每块先算 $\mathrm{acc}_p/l_p$ 就丢了 $l_p$ 这个权重，合并得写成加权平均 $\big(\sum_p l_p e^{m_p-m}\cdot\tfrac{\mathrm{acc}_p}{l_p}\big)/\sum_p l_p e^{m_p-m}$， $l_p$ 还是得带着走。换句话说：**"先除"并没有减少需要传递的状态，只是把状态藏进了一个更难看的公式里**；而且它引入了一次额外的除法与一次额外的乘法（先除后乘回来）， 数值上还多一次舍入。所以"最后再除"是**可归并性的自然形式**，不是省除法的小技巧。

FlashAttention-2 论文把这件事说成算法层的改动（arXiv:2307.08691 §3.1）： "We do not have to rescale both terms of the output update by $\mathrm{diag}(\ell^{(2)})^{-1}$"，而是 "maintain an 'un-scaled' version of $O^{(2)}$ and keep around the statistics $\ell^{(2)}$ ... Only at the every end of the loop do we scale the final $\tilde O^{(\text{last})}$ by $\mathrm{diag}(\ell^{(\text{last})})^{-1}$ to get the right output"。同一节还提到反向只需存 logsumexp： "We only need to store the logsumexp $L^{(j)}=m^{(j)}+\log(\ell^{(j)})$"——本仓只做 forward，不存 $L$，这一条列为**未实现**而非"不需要"（§6 适用边界）。

#### 3.2.4 精确恒等 vs 浮点舍入:误差到底有多大

代数上精确，不等于位级相同。本仓的实测把这条界给死了（EXP-T04,3 轮协议 B1·H16/8·D128·bf16，参考 = fp32 精算）：

| $S_{kv}$ | flash_decode vs fp32 精算 max abs err |
|---|---|
| 512 | **0.0** |
| 2048 | **0.0** |
| 8192 | 4.9e-4 量级（单轮记录值） |
| 32768 | 6.1e-5 |

（512/2048 两格的 0.0 见 data/raw/EXP-T04/20260825T152434_flash_decode_stability_r1.json； 8192 那格的量级取自 EXP-T04 §5 的原表，该表其余速度行已作废、误差行保留。）

怎么读这张表：**误差不随 $S_{kv}$ 单调增长**，32K 的 6.1e-5 比 8K 还小。原因是分母 $l$ 越大、单项相对权重越小，而输出是加权平均——**加权平均的舍入误差不随项数线性累积**。这与"求和项数越多误差越大"的直觉相反，值得记住：online softmax 的归一化结构本身就是一个方差抑制器。

**同一代数用两次**（白板卡 docs/talk/whiteboard_card_fa2_algebra.md 的主线）：块内 online 是"（已累积的历史） ⊕（新的 K/V 块）"(src/fa2_fwd.py：107-119)，块间 reduce 是 "（段 0） ⊕（段 1） ⊕… ⊕（段 p）"(src/flash_decode.py：115-123)。

### 3.3 IO 复杂度定理:FlashAttention 到底证明了什么

#### 3.3.1 三条结论的准确陈述

FlashAttention(arXiv:2205.14135)给的不是"更快"，而是三条可证的结论。逐条抄原文并解释：

1. **Theorem 1**（正确性与内存）："Algorithm 1 returns $\mathbf{O}=\mathrm{softmax}(\mathbf{QK}^\top)\mathbf{V}$ with $O(N^2d)$ FLOPs and requires $O(N)$ additional memory beyond inputs and output."——读法：**FLOPs 一个没省**（$O(N^2d)$ 与朴素同阶），省的是额外内存（$O(N)$ 而不是 $O(N^2)$）。任何"FlashAttention 减少了计算量"的说法都与 Theorem 1 矛盾。
2. **Theorem 2**（IO 复杂度）："Standard attention requires $\Theta(Nd+N^2)$ HBM accesses， while FlashAttention requires $\Theta(N^2d^2M^{-1})$ HBM accesses." 前提写在定理的设定里：$d \le M \le Nd$，$M$ 是片上 SRAM 的容量。
3. **Proposition 3**（下界）："There does not exist an algorithm to compute exact attention with $o(N^2d^2M^{-1})$ HBM accesses for all $M$ in the range $[d, Nd]$."——读法：Theorem 2 的上界在这个 $M$ 区间上**是紧的**，不存在渐进更优的精确算法。这条命题才是这篇论文与"又一个 attention 优化"的分界线。

论文给的硬件量纲（A100）:"The A100 GPU has 40-80GB of high bandwidth memory with bandwidth 1.5-2.0TB/s and 192KB of on-chip SRAM per streaming multiprocessor with bandwidth estimated around 19TB/s."并注明"For typical values of $d$ (64-128) and $M$ (around 100KB), $d^2$ is many times smaller than $M$"。

#### 3.3.2 Theorem 2 的计数论证(补出中间步)

论文的证明骨架只有两句："Given the SRAM size of $M$, we can load blocks of $\mathbf{K}, \mathbf{V}$ of size $\Theta(M)$ each. For each block of $\mathbf{K}$ and $\mathbf{V}$, we iterate over all blocks of $\mathbf{Q}$ to compute the intermediate values, resulting in $\Theta(NdM^{-1})$ passes over $\mathbf{Q}$." "Each pass loads $\Theta(Nd)$ elements, which amounts to $\Theta(N^2d^2M^{-1})$ HBM accesses."

把中间步补齐（本讲义推导）：

- K 的元素总数是 $Nd$；每个 K/V 块占 $\Theta(M)$ 个元素；于是块数 $T_c = \Theta(Nd/M)$。**这一步的合法性**：要求一个块连同 Q 块与中间量能同时装进 $M$，即 Algorithm 1 里的 $B_c=\lceil M/4d\rceil$——分母里那个 4 就是"同时驻留 Q、K、V、O 四份"的常数，被 $\Theta$ 吸收掉了。
- 外层每换一个 K/V 块，内层要把 Q（和 O）整条过一遍，即 $\Theta(Nd)$ 个元素。
- 相乘：$\Theta(Nd/M)\times\Theta(Nd)=\Theta(N^2d^2/M)$。
- **上界区间 $M\le Nd$ 的意义**：若 $M > Nd$，整个 K/V 一次装下，$T_c=1$，访问量退化成 $\Theta(Nd)$，定理的形式不再有意义；**下界区间 $M\ge d$ 的意义**：至少要能装下一行，否则连一次点积都做不完。

Algorithm 1 的块大小设定原文是："Set block sizes $B_c=\lceil M/4d\rceil$， $B_r=\min(\lceil M/4d\rceil, d)$"。注意 $B_r$ 那个 $\min(\cdot, d)$：它保证 $B_r\times d$ 的 Q 块不会比一个 K/V 块还大，是为了让片上预算平摊——**论文没解释这个 min 的来历，补出来就是"Q 块的两个维度乘积受同一份 $M$ 约束"**。

#### 3.3.3 把定理代进本仓形状:被消掉的与没被消掉的

一个常见的半对说法是"FA 把 HBM 流量从 $O(S^2)$ 降到 $O(S\cdot D)$"。按 Theorem 2 的准确形式，降到的是 $\Theta(S^2D^2/M)$，不是 $\Theta(SD)$。精确一点：

- **被消掉的**：$S\times S$ 分数矩阵的**写回 + 读取**——1.07 GB 对 99 KB 级的 per-block shared memory，无论如何进不了片上，是纯粹不可缓存的 HBM 往返。
- **没有被消掉的**：K/V 的**重读**。每个 Q 行块都要把可见的那段 K/V 整条流过一遍， S=4096、BLOCK_M=128 时有 $S/\mathrm{BM}=32$ 个 Q 行块，causal 下平均各读一半： $$0.5 \times 32 \times \underbrace{(8 \times 4096 \times 128 \times 2\,\mathrm{B}) \times 2}_{K+V=16.8\,\mathrm{MB}} \approx 268\ \mathrm{MB}$$ 在 1008 GB/s 上约 $0.266\,\mathrm{ms}$，占实测 $1.1184\,\mathrm{ms}$ 的 24%；同形状的算力时间是 $137.4/165 = 0.83\,\mathrm{ms}$，占 74%。**compute-bound 成立，但不是因为"访存量降到 $O(S\cdot D)$"，而是因为剩下的访存量正好被算力盖住。**

**这个 268 MB 用的是什么假设，必须挑明**（本讲义推导）。它按"每个 Q 行块索引 $m$ 把整份 KV（8 个 kv 头）读一遍"计数，隐含假设是：同一个 $m$ 上的 4 个共享同一 kv 头的 q head **彼此复用到了缓存**。把假设两端放松，得到一对夹逼：

| 口径 | 字节数 | 折算时间 | 占实测 1.1184 ms |
|---|---|---|---|
| 下界：每个张量只读/写一次（compulsory） | Q 33.55 + K 8.39 + V 8.39 + O 33.55 = **83.9 MB** | 0.083 ms | 7.4% |
| 中：上式（GQA 组内完美复用） | **268 MB** | 0.266 ms | 24% |
| 上界：GQA 组内零复用（$H_q\times S/\mathrm{BM}$ 次各读一半） | **1.07 GB** | 1.065 ms | 95% |

三个数差 13 倍，而实测算力占比 74%（与 kperf 卡片一致），**说明真实情况明显靠近下界那一侧**。原因可以直接指出来：整份 K+V 只有 16.8 MB，而 RTX 4090 的 L2 是 73728 KB（Ada 白皮书 Table 2），**KV 整体装得进 L2**，所谓"重读"绝大部分是 L2 命中而不是 HBM 往返。这条也解释了为什么讲义 02 §3.1 的 GEMM 会出现同一个现象—— **"tile 模型算出的 HBM 流量"永远是上界，L2 是那个把上界打下来的东西。**

#### 3.3.4 tile 尺寸的三个来源:理论上界 / 硬件约束 / 实测扫描

这条账还解释了 tile 扫描的主要结果：BLOCK_M 从 64 提到 128,4K 上 $1.362\to1.126\,\mathrm{ms}$（+17%，EXP-T01 §5，tile 扫描为终端级证据）。BM 翻倍同时做了两件事：K/V 的重读**次数减半**，以及每块 softmax 簿记被更多 Q 行摊薄。反方向的硬约束同样实测过：BLOCK_N 提到 128 直接 OOM（需求 160 KB）。

把本仓每个 tile 相关的魔法数按"谁决定的"分类（这是 §3.8 的预告）：

| 魔法数 | 值 | 决定者 | 依据 |
|---|---|---|---|
| BLOCK_M(fp16) | 128 | **实测扫描** | 4K 上比 BM64 快 17%(EXP-T01 §5) |
| BLOCK_N(fp16) | 64 | **硬件约束** | 128 那一档 shared memory 需求 160 KB > 99 KB（EXP-T08 逐字复现 `Required: 163840, Hardware limit: 101376`） |
| num_warps(fp16) | 8 | 实测扫描 | EXP-T01 优配；寄存器 213/线程使每 SM 仅 1 CTA(§3.7.4) |
| num_stages(fp16) | 2 | 实测扫描 + 硬件约束 | 见讲义 02：Triton 3.6 下这一档只有 1 份缓冲（EXP-T08），再深会撞 99 KB |
| BLOCK_M(fp32) | 32 | 硬件约束 | fp32 tile 字节翻倍，BM128 超上限（src/fa2_fwd.py：139-142 的 docstring） |
| $\mathrm{scale}=D^{-1/2}$ | 1/√128 | **理论** | Vaswani et al. arXiv:1706.03762 §3.2.1 的方差论证 |

**记住这张表的形状，而不是表里的数**：任何一个 tile 参数，若你说不出它属于哪一类， 就是没调完。

### 3.4 FA1 → FA2:循环反转到底省了什么

#### 3.4.1 论文的三处改动

FlashAttention-2(arXiv:2307.08691)摘要把改动列成三条： "tweak the algorithm to reduce the number of non-matmul FLOPs"(§3.1)、 "parallelize the attention computation, even for a single head, across different thread blocks to increase occupancy"(§3.2)、 "within each thread block, distribute the work between warps to reduce communication through shared memory"(§3.3)。整体效果是 "around 2× speedup compared to FlashAttention, reaching 50-73% of the theoretical maximum FLOPs/s on A100"。

三条在本仓的落点各不相同，必须分开说：

| 论文改动 | 本仓状态 | 落点 / 原因 |
|---|---|---|
| §3.1 减少非 matmul FLOPs | **已做** | 循环外只除一次（src/fa2_fwd.py：125） |
| §3.2 seq 维并行 | **已做** | grid 第一维 = `cdiv(S, block_m)`(src/fa2_fwd.py：155) |
| §3.3 warp 间划分（split-Q） | **未做** | 交给 Triton 编译器；这正是 §5.1 那段"抽象税"的一部分 |

论文对 §3.3 的原话是：FA1 用 "split-K"（把 K/V 切给不同 warp），导致 "all warps need to write their intermediate results out to shared memory， synchronize， then add up"；FA2 改成 "split Q across 4 warps while keeping K and V accessible by all warps"，消掉 warp 间通信。**本仓写的是 Triton，warp 级划分不在可控范围内**——这是"用 Triton 换开发效率"的代价，如实记在 §8.2。

#### 3.4.2 循环反转的中间量账(本讲义推算,非实测)

FA1 的循环是**外层 K/V、内层 Q**，而 $(m,l,\mathrm{acc})$ 是**按 Q 行**维护的，所以每处理一个 K/V 块就要把所有 Q 行块的三元组从 HBM 读出、更新、写回。按本仓形状粗算（BM=128、BN=64、S=4096、每行 $2+D=130$ 个 fp32）：

$$\underbrace{\frac{S}{\mathrm{BN}}}_{64\ \text{个 K 块}} \times \underbrace{\frac{S}{\mathrm{BM}}}_{32\ \text{个 Q 块}} \times \underbrace{128 \times 130 \times 4\,\mathrm{B}}_{\text{一个 Q 块的三元组}} \approx 136\ \mathrm{MB}\ (\text{每 } (b,h)\ \text{读写各一遍})$$

乘上 $B\cdot H_q=32$ 就是 GB 级——**与被消掉的 $S^2$ 物化同一个量级**。所以循环反转不是锦上添花：FA1 消掉了 $S\times S$ 物化却在中间量上还回去一大半，FA2 把外层换成 Q、内层换成 K/V，三元组从头到尾**只活在寄存器里**，这一半才真正消掉。（以上为按本仓形状的推断算式，非实测；本仓无 FA1 实现。）

这个账与 Theorem 2 并不矛盾：定理是渐进阶，FA1 与 FA2 的 HBM 访问量**同阶** ($\Theta(N^2d^2M^{-1})$)，差的是常数因子与"哪个张量在被重读"。**渐进同阶而工程上差一倍，是复杂度理论与 kernel 工程之间最常见的落差**，也是为什么 FA2 是一篇独立论文而不是一个 commit。

#### 3.4.3 非 matmul FLOPs 为什么值钱:那个 16 倍

FA2 论文把理由写得很直白：A100 有 "max theoretical throughput of 312 TFLOPs/s of FP16/BF16 matmul， but only 19.5 TFLOPs/s of non-matmul FP32"，比值 16×。也就是说**一次非 matmul 的 FP32 运算， 按机器时间折算，值 16 次 matmul FLOP**。

本仓的 4090 上这个比值是多少？按 Ada 白皮书 Table 2：FP16 Tensor / FP32 累加 165.2 TFLOPS，非 Tensor 的 Peak FP32 82.6 TFLOPS，比值只有 **2.0×**（本讲义推导）。差别的来源要说清楚：A100 的 19.5 TFLOPs/s 是 FP32 **CUDA core** 的经典口径（每 SM 64 个 FP32 单元），而 Ada 每个 SM 有 128 个 FP32 通道（白皮书：每个 SM 分区含"16 CUDA Cores that are dedicated for processing FP32 operations"加 "16 CUDA Cores that can process FP32 or INT32"），同时 tensor core 的峰值相对较低。 **所以"省非 matmul FLOPs"在 Ada 上的边际收益本来就比 A100 小一半以上**——这条是本讲义自己算出来的推论，本仓没有做"关掉这项优化"的对照实验，标为推断。

但要注意：`tl.exp` 走的是 SFU（每 SM 分区一个 Special Function Unit，白皮书 Ada SM 结构段），不是 FP32 通道，吞吐比 FP32 FMA 更低；所以真实的"非 matmul 税" 比 2.0× 这个下界更重。本仓无计数器权限（docs/theory/04），不给具体倍数。

#### 3.4.4 并行度:seq 维进 grid

`grid = (triton.cdiv(S, block_m), B * Hq)`(src/fa2_fwd.py：155)，4K 形状下 $32\times32=1024$ 个 program，对 128 个 SM 绰绰有余。NVIDIA 的性能指南把这条要求写成"the number of thread blocks to be several times higher than the number of SMs"(GPU Performance Background User's Guide，§4)；1024/128 = 8 倍，满足。记住这个数字——§3.6 里它会塌成 16，只有 SM 数的 1/8。

### 3.5 GQA 头映射:一行寻址换来的一半带宽

GQA 让多个 q head 共享一个 kv head。kernel 里改的只有一行： `hkv = hq // GQA_GROUP`(src/fa2_fwd.py：55)。

- **为什么可以这么写**：GQA 的定义就是**连续分组**共享（第 $g$ 组 q head 是 $[gG,(g+1)G)$），整数除法正是那个映射。论文（arXiv:2305.13245）的定位是 "an interpolation between multi-query and multi-head attention"，目标是在 MQA 的带宽收益与 MHA 的质量之间取中间点。
- **收益在哪**：KV 的 HBM 读取量按 $H_{kv}/H_q$ 缩小（本仓 8/32 = 1/4），而且 **不物化 repeat**——参考实现要 `repeat_interleave` 造一份 4 倍大的 KV (scripts/test_fa2.py：25-26)，kernel 只换个索引读原张量。这份"免掉的拷贝"在 §5 的三臂口径里被单独标价。
- **改错会怎样**：写成 `hq % Hkv` 就把连续分组改成轮转分组——不 nan、不越界、性能一模一样，只是**悄悄算错**。正确性 gate 里的 `(1, 16, 4, 777, 128, True)` (scripts/test_fa2.py：41)就是为抓这类"安静的错误"设的格。

**一个必须承认的口径问题**：本仓从未做过"关掉 GQA 走 MHA"的对照臂，所以 "GQA 省了多少"在 EXP-T01 里**测不出来**；它只在 EXP-T04 的三臂设计里以 "repeat 内计 vs 预置"的形式被标价（§5.2）。**没有对照臂的收益不能报数字**， 这是本仓的一条通用纪律。

### 3.6 decode 的并行度塌陷:为什么必须换一根并行轴

#### 3.6.1 塌陷的算式

把 §3.4.4 的 grid 代入 decode：$S_q=1$ 时第一维 $=\lceil 1/\mathrm{BM}\rceil=1$，program 总数塌成 $B\cdot H_q$。本仓 EXP-T04 协议 B=1、$H_q$=16(scripts/test_flash_decode.py：37)， 就是 **16 个 program 面对 128 个 SM**，87.5% 的 SM 全程闲置——这不是调参能救的。

PyTorch 官方 flash-decoding 博客把同一件事写在 A100 上： "During inference， the query length is typically 1： this means that if the batch size is smaller than the number of streaming multiprocessors (SMs) on the GPU (108 for an A100)， the operation will only use a small part of the GPU！ ... With a batch size of 1， FlashAttention will use less than 1% of the GPU！"——本仓的 16/128 = 12.5% 比"less than 1%"好，因为本仓 $H_q=16$ 而博客的语境是 per-head 已经算进去之后的 batch 维。**同一个机制，不同的分子分母，数字差一个量级； 引用时必须把分母说清楚**。

#### 3.6.2 split-K 的构造与 splits 启发式的三个上界

**构造**：把 KV 切成 `num_splits` 段，grid 变成 `(num_splits, B*Hq)`。每段独立跑与 FA2 完全相同的 online softmax，只是**不做最后那次除法**，而把 $(m_p,l_p,\mathrm{acc}_p)$ 写出去；第二个 kernel 用 §3.2 的合并算子一次归并所有段。官方博客把它写成三步："First, we split the keys/values in smaller chunks."、 "We compute the attention of the query with each of these splits in parallel using FlashAttention. We also write 1 extra scalar per row and per split: the log-sum-exp of the attention values."、"Finally, we compute the actual output by reducing over all the splits, using the log-sum-exp to scale the contribution of each split." 并明确"we have 2 separate kernels to perform respectively (2) and (3)"。

**本仓的 splits 怎么选**(src/flash_decode.py：139-147)，三个约束依次收紧：

1. **想要多少**（填满 SM 的启发式）：$\text{want} = 2\times128/(B H_q)$——4090 有 128 个 SM，每 SM 至少 2 个 CTA 才有延迟切换余地。这一条来自 NVIDIA 的通用建议（"GPUs hide dependent instruction latency by switching to the execution of other threads"，GPU Performance Background User's Guide §4），**不是本仓测出来的**。
2. **最多能切多少**（算法上界）：$\lceil S_{kv}/\mathrm{BLOCK\_N}\rceil$——段长不小于一个 BLOCK_N，再细切只剩空转。
3. **编译期约束**：`next_power_of_2`，因为 combine kernel 里 `tl.arange(0, NUM_SPLITS)` 要求编译期 2 的幂。**这是语言约束，不是硬件约束**， 代价是可能造出空段——由 §3.2.2 的单位元性质自动兜住。

代进 32K 那格：want = 256/16 = 16，上界 256，故 splits = 16、split_size = 2048。 **启发式未扫参**(EXP-T04 §7)——这是本仓明确留着的开放项，不要把它讲成调优结论。

#### 3.6.3 为什么必须两个 kernel

归并每一项都要乘 $e^{m_p-m_g}$，而 $m_g=\max_p m_p$ 要等**所有段**算完才知道； atomicAdd 只能做无状态的可交换加法，做不了"先等全局 max 再逐项换基"的非线性归约。

把这条论证收紧一点（本讲义推导）：设想用 atomicAdd 累加 $\mathrm{acc}_p$。要让结果正确，每个 $\mathrm{acc}_p$ 必须在加之前乘上 $e^{m_p-m_g}$；但 $m_g$ 依赖所有 $p$， 所以第一个到达的线程无法知道自己的系数。**唯一的绕法**是改用"固定基准"：令 $m_g$ 取一个先验安全上界 $\hat m \ge \max_p m_p$，则每段可独立乘 $e^{m_p-\hat m}$ 后原子累加。代价是：$\hat m$ 取大了整行下溢、取小了上溢——**回到 §2.4 的老问题**， 而且是无法自适应的版本。生产实现里确实有这条路（固定 scale 的 FP8 attention 是它的远亲），但它牺牲的是无条件的数值安全，本仓不走。

两个 kernel 的边界就是那次跨 CTA 同步（CUDA 里 grid 级同步最便宜的写法）。代价是 **多一次 launch**，这笔账会在 §5.2 的短上下文那格原样出现。

### 3.7 硬件语义层:这些代码写法是被什么约束定死的

这一节把"为什么代码非得这么写"落到指令与资源的准确语义上。本仓写的是 Triton， 不直接写 PTX，但**Triton 生成什么、能不能编译通过，完全由这些语义决定**。

#### 3.7.1 shared memory:每 thread block 99 KB 是一堵会报错的墙

Ada Tuning Guide §1.4.1.1 的两句话是本篇所有"装不下"的唯一依据： "The shared memory capacity per SM is 100 KB."与 "The maximum shared memory per thread block is 99 KB."（§1.4.2.2 另说明整块 "combined L1 cache capacity is 128 KB"，在 L1 与 shared 之间可配置， 支持 "shared memory capacity of 0, 8, 16, 32, 64 or 100 KB per SM"; "Static shared memory allocations remain limited to 48 KB, and an explicit opt-in is also required to enable dynamic allocations above this limit"。）

$99\times1024 = 101376$。这个数字在本仓被**逐字撞到过**：EXP-T08 用编译期资源探针复现 BLOCK_N=128 那一档的报错原文——`OutOfResources: Required 163840, Hardware limit 101376`。163840 B = 160 KB，正是 EXP-T01 §5 记的那个数；101376 B 正是 99 KB。**论文/文档给的上限与编译器抛出的数字对得上，这是本篇唯一一条 "文档 → 实测"完全闭合的链**（§8.1 会再列一次）。

FA2 kernel 的 shared memory 怎么算（EXP-T08 §6 的公式，逐格吻合）：

$$\text{shared} = \underbrace{\mathrm{BM}\times D\times 2\,\mathrm{B}}_{Q\ \text{tile 常驻}}
+ \max(1, N_{\text{stages}}-1)\times\underbrace{\mathrm{BN}\times D\times 2\,\mathrm{B}\times 2}_{K,V\ \text{各一份}}$$

代入 BM=128、D=128 得 Q tile = 32 KB；各档验算（EXP-T08 §5 实测表）：

| BLOCK_N | num_stages | 公式值 | 实测 metadata.shared |
|---|---|---|---|
| 32 | 2 | 32 + 1×16 = 48 KB | 49152 |
| 32 | 3 | 32 + 2×16 = 64 KB | 65536 |
| 64 | 2 | 32 + 1×32 = 64 KB | 65536 |
| 64 | 3 | 32 + 2×32 = 96 KB | 98304 |
| 128 | 2 | 32 + 1×64 = 96 KB | 98304 |
| 128 | 3 | 32 + 2×64 = **160 KB** | **OOM(163840 > 101376)** |

**注意那个 $\max(1, N-1)$**——它不是笔误，是 Triton 3.6 在本机的实测语义，讲义 02 §3.3 专门讲透。这里只需要它的一个推论：**BLOCK_N=128 在 stages=2 那一档是能编译的（96 KB），OOM 只发生在 stages=3**；而 stages=2 那一档的寄存器打到 255/线程（Ada 每线程上限，Tuning Guide §1.4.1.1），等于把压力从 shared memory 换到了寄存器。 EXP-T01 的 tile 扫描是终端级证据、未记录 stages 列，所以"当年那次 OOM 具体在哪一档" 无法回溯；**能确定的是那个字节数与那堵墙**。

#### 3.7.2 cp.async 的 commit-wait 组语义

Triton 的软件流水在 SM80+ 上会把 `tl.load` 降成 `cp.async`。这条指令的语义决定了 "预取"在硬件上是什么形状，PTX ISA 的准确说法（§9.7.9.26 Asynchronous copy）：

- `cp.async.ca.shared.global` / `cp.async.cg.shared.global`(§9.7.9.26.3.1)：把数据从 global **直接**搬进 shared，不经过寄存器。`ca` 是 cache at all levels、 `cg` 是 cache in global level only；cp-size 只能是 4、8 或 16 字节，且用 `cg` 时 cp-size **必须**是 16。
- `cp.async.commit_group`(§9.7.9.26.3.2)："commits all prior uncommitted cp.async instructions into a cp.async-group"。**注意"组"这个词**：等待的粒度不是单条指令，是一批。
- `cp.async.wait_group N`(§9.7.9.26.3.3):"wait till only N or fewer of the most recent cp.async-groups are pending and all the prior cp.async-groups committed by the executing threads are complete"。

这三条合起来解释了软件流水的形状：**每一轮循环把这一轮的所有预取 commit 成一个组， 然后 `wait_group(N-1)` 等到只剩 $N-1$ 个组在飞**。于是"流水深度"在硬件上就是 "允许同时在飞的组数"，而"每个组需要一份独立的 shared memory 缓冲"——这就是 shared memory 占用随深度线性增长的机制根源。

**本仓做到哪一层、没做到哪一层要说清**：EXP-T08 只读了编译产物的 `metadata.shared`，**没有 dump TTGIR/PTX 去数 commit/wait 的分组数量**（EXP-T08 §7 明列为开放项）。所以上面这段是"PTX 语义 + 缓冲份数实测"的合成解释，**组数与份数一一对应这条尚未在本仓被直接观测**，标为未核实。

#### 3.7.3 mma 的 fragment 布局与 M 维粒度

`tl.dot` 在 Ada 上降成 `mma.sync`。PTX ISA §9.7.15(Warp Level Matrix Multiply-Accumulate Instructions)的两条语义决定了写法：

1. **fragment 是分布式的**：§9.7.15.4.1 定义 matrix fragment 时明确 "each thread in a warp holds a fragment of the matrix"；`mma.m16n8k16` 的布局由 `groupID = %laneid >> 2`、`threadID_in_group = %laneid % 4` 索引出来（§9.7.15.5.8 的 Matrix Fragments 表）。也就是说**矩阵的行列与线程 lane 之间有一个固定的、由 ISA 规定的置换**，你不能自选布局。
2. **mma 是 warp 级集合操作**：同节说明它是 "warp-wide collective"，warp 内所有线程必须一起执行。**这条直接解释了为什么 Triton 不让你在 `tl.dot` 周围写发散分支**——warp 内一旦发散，集合操作就没法保证对齐。

对本仓最有用的推论是 **M 维粒度**。一个常见说法是"`tl.dot` 要求 M ≥ 16"，这句话在本机的 Triton 3.6.0 上**不准确**：NVIDIA 后端的 `min_dot_size` 返回 `(1, 1, 16)`（8 位输入时是 `(1, 1, 32)`），即 M 与 N 的下界是 1、K 的下界才是 16（triton/backends/nvidia/compiler.py 的 `check_dot_compatibility`，注释原文： "For small M/N the input we can still use tensorcores with padding"）；Python 侧的断言只检查 `M >= min_dot_size[0]` 等三条（triton/language/semantic.py 的 `dot`，报错文本 "Input shapes should have M >= ...， N >= ... and K >= ..."）。

**所以准确的说法是**：M=1 不会编译失败，但 `mma.m16n8k16` 的 M 维粒度是 16， tile 会被**填充**到 16 行，15/16 的 tensor core 行利用率被浪费。代价是真的， "编译报错"这个机制不是。src/flash_decode.py：53-54 的注释（"tl.dot 要求 M≥16 得 pad"）与 src/gemm_pipelined.py：116-117 的注释（"mma 行利用率 1/128"）表达的是同一件事的两种口径，前者的措辞略松——**代价的结论不变，机制的表述以此处为准**。

#### 3.7.4 warp 调度与"occupancy 17%"的算术

Ada 白皮书对 SM 的描述给了发射规则："the AD10x SM is divided into four processing blocks (or partitions), with each partition containing a 64 KB register file, an L0 instruction cache, one warp scheduler, one dispatch unit, 16 CUDA Cores that are dedicated for processing FP32 operations ... one Ada Fourth-Generation Tensor Core, four Load/Store units, and a Special Function Unit (SFU)"。

四条推论，都是本篇要用的：

1. **每 SM 4 个 warp scheduler、每个 1 个 dispatch unit** → 每周期最多发射 4 条指令，一个分区一条。所以"warp 多"只在**有指令可发**时才有用。
2. **每分区 64 KB 寄存器 = 16384 个 32 位寄存器**，4 个分区合计 64K（与 Tuning Guide §1.4.1.1 的 "64K 32-bit registers per SM" 一致）。
3. **每 SM 最多 48 warps**（同上）。
4. 每分区一个 tensor core、一个 SFU：**`tl.exp` 与 `tl.dot` 抢的不是同一个单元**， 这是 online softmax 能与 matmul 部分重叠的物理基础。

把本仓 FA2 的数字代进去（本讲义推导，与 EXP-T08 §6 的交叉验证一致）：

- num_warps=8 → 每 CTA 256 线程 = 8 warps;
- kperf 观测 regs = 213/线程（终端级证据，登记于 EXP-T06《FP8 GEMM》§7）；
- 每 CTA 需要 $213\times256 = 54528$ 个寄存器；$65536/54528 = 1.2 < 2$ → **每 SM 只能驻留 1 个 CTA**；
- 占用率 $= 8\ \text{warps} / 48 = 16.7\% \approx 17\%$，与 kperf 卡片逐字相同。

**同一个结论还有第二条独立路径**：FA2 默认档（BM128/BN64/stages2）的 shared memory 是 64 KB（EXP-T08 实测），而每 SM 只有 100 KB → 同样只能放下 1 个 CTA。 **寄存器与 shared memory 两条路给出同一个答案，占用率 17% 就不再需要"观测"， 它是可以推出来的。** 这也把"occupancy 低 = 没写好"这条直觉直接证伪： 本仓 FA2 的算力利用率是 74%(§5.1)，occupancy 17% 是**为大 tile 付的价钱**。

### 3.8 每个魔法数的来历(汇总)

| 参数 | 值 | 类别 | 依据（可核验） |
|---|---|---|---|
| `sm_scale` | $D^{-1/2}$ | 理论 | arXiv:1706.03762 §3.2.1:$q\cdot k$ 方差为 $d_k$，故除 $\sqrt{d_k}$ |
| `BLOCK_M` fp16 | 128 | 实测 | EXP-T01 §5,4K 上比 64 快 17% |
| `BLOCK_N` fp16 | 64 | 硬件 | 128 档需 160 KB > 99 KB（EXP-T08 复现报错原文） |
| `num_warps` fp16 | 8 | 实测 | EXP-T01 扫描；占用率后果见 §3.7.4 |
| `num_stages` fp16 | 2 | 实测 | EXP-T01 扫描；语义见讲义 02 §3.3(份数 = max(1，N−1)) |
| `BLOCK_M`/`num_warps`/`num_stages` fp32 | 32/4/1 | 硬件 | tile 字节翻倍（src/fa2_fwd.py:139-142） |
| `IEEE_DOT` | dtype 决定 | 语义 | fp32 校验路线要关 TF32（10 位尾数） |
| fd `block_n` | 128 | 未扫参 | src/flash_decode.py：131 默认值，EXP-T04 §7 列为开放项 |
| fd `num_splits` | 见 §3.6.2 | 启发式 | 三个上界依次收紧；**未扫参** |
| fd `num_warps` | 4 | 未扫参 | src/flash_decode.py:165,170 |
| 正确性阈值 | 2e-2 | 经验 | scripts/test_fa2.py：5 的 docstring 明写"经验界" |

**这张表的用法**：面试被问"这个数为什么是 128"，答案必须落在四类之一——理论、硬件、实测、未扫参。落在"未扫参"不丢人，**说不出属于哪一类才丢人**。

## 4. 代码逐段走读:src/fa2_fwd.py 全核 + src/flash_decode.py 两段式

按执行顺序走读（引用为仓内真实代码逐字拷贝，标 文件：起-止行）。

**第 1 段 · 启动器：形状契约、tile 默认值与 grid**(src/fa2_fwd.py：143-155)

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

角色：kernel 的全部外部约定都在这 13 行里定死。①**tile 按 dtype 分叉**：fp16 走 BM128/w8/s2（EXP-T01 扫描最优），fp32 降 BM32/w4/s1——fp32 的 tile 字节翻倍、BM128 撞 Ada 的 shared memory 上限；fp32 只是"校验路线"，与 `IEEE_DOT` 配套（第 5 段）。 ②**`assert Hq % Hkv == 0`** 是 GQA 连续分组映射（§3.5）的前提，不满足不是"算慢了"而是 "算错了"，所以用 assert 而非 fallback。③改错会怎样：grid 只写 `(B*Hq,)` 就退化成 §3.6 的 decode 惨状。

补两点这一段隐含的设计决策：

- **`D in (64, 128)` 是白名单而非区间检查**。原因是 HEAD_DIM 进 `tl.constexpr` 后要参与 `tl.arange(0, HEAD_DIM)`，必须是编译期 2 的幂；写成 `D <= 128` 会让 D=96 这类值编译期报错，报错点离用户很远。**把语言约束前移成接口断言**，是 Triton 项目里反复出现的模式（讲义 03 第 2 段的 `next_power_of_2` 是同一模式）。
- **`sm_scale` 允许外部覆盖**。默认 $D^{-1/2}$ 来自 Vaswani et al. §3.2.1，但 QK-Norm 之类的变体会改这个值；把它留成参数而不是写死，是"数学常数也可能是契约的一部分"的体现。

**第 2 段 · program 定位与 GQA 一行映射**(src/fa2_fwd.py：51-56)

```python
    pid_m = tl.program_id(0)          # 第几个 Q 行块
    pid_bh = tl.program_id(1)         # batch*q_head 扁平索引(两维并行压一维,免 3D grid)
    b = pid_bh // NUM_Q_HEADS
    hq = pid_bh % NUM_Q_HEADS
    hkv = hq // GQA_GROUP             # GQA:多个 q head 共享一个 kv head,直接换算
                                      # 索引读原 KV,不物化 repeat(省 HBM 与显存)
```

角色：把线性 program id 翻译成 $(b,h_q,\text{M 块})$ 三元坐标，顺手完成 GQA 头映射。 **这里没有任何原子操作与跨 program 通信**——每个 program 独占一个输出行块，这是循环反转（§3.4）的结构性好处，也是三元组能常驻寄存器的前提。改错会怎样：`GQA_GROUP` 若不是 `tl.constexpr`，整数除法从编译期常量折叠变成运行期指令，寻址开销进热路径。

**为什么 `pid_bh` 放在 grid 的第二维而不是第一维**（值得单独想一遍）：CUDA 的 blockIdx 按 x 维最快变化，同一时刻在跑的 CTA 其 `program_id(0)` 相邻。把 M 块放在第 0 维意味着**相邻 CTA 处理同一个（b， hq） 的相邻 Q 行块**，它们读的是同一段 K/V → L2 命中率高。若对调两维，相邻 CTA 会分属不同 head，读的 K/V 完全不同，L2 复用被打散。这与讲义 02 §3.1 的 grouped 调度是同一条原理的两种实现（本讲义推导，本仓未做对调两维的对照实验）。

**第 3 段 · Q tile 常驻 + 三个跑动量的不变量**(src/fa2_fwd.py：62-77)

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

角色：建立主循环的**不变量**。注释里那三行不是装饰，是循环正确性的证明骨架——任意时刻 $\mathrm{acc}/l_i$ 就是"只看前 $n$ 列"的精确输出，所以循环在任何一块之后停下都自洽。三个决策：①**Q tile 只载一次**，它在整条 K/V 流里被复用 $S/\mathrm{BN}$ 次，是算术强度的分子；②**全 fp32**，$l_i$ 是几千项求和，fp16 丢的位会直接乘进输出； ③**$m_i$ 初值 $-\infty$** 使首块 $\alpha=0$、"换基"退化成"直接赋值"，正是 §3.1.3 里归纳法边界条件在代码里的样子。改错会怎样：初值写 0 则首块分数全为负时 $l_i$ 偏小、输出整体错。

**"全 fp32"这个决定值多少钱**（本讲义推导）：$\mathrm{acc}$ 是 $\mathrm{BM}\times D = 128\times128$ 个 fp32，按 256 线程摊 = **64 个寄存器/线程**， 加上 $m_i$、$l_i$ 与地址寄存器，就把 213 regs/线程这个观测值解释掉了大半（§3.7.4）。也就是说：**"用 fp32 累加"这个数值决定，直接决定了 occupancy 只有 17%**。数值精度与占用率在同一份预算里争抢——这是 kernel 设计里最典型的一对张力。

**第 4 段 · causal 上界与循环体：三种 mask 各司其职**(src/fa2_fwd.py：79-105)

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

角色：这一段同时处理**算力**与**正确性**，且刻意用了三种不同的越界处理。`hi` 是算力那件事：本块行号 $\in[\mathrm{pid}_m\mathrm{BM},(\mathrm{pid}_m+1)\mathrm{BM})$， causal 下行 $m$ 只可见列 $n\le m$，故可见列的上确界是 $(\mathrm{pid}_m+1)\mathrm{BM}$——对角线以下的整块**根本不进循环**，FLOPs 直接减半。这是"mask 不只是填 $-\infty$， 更是根本不算"的实现，与 FA2 论文 §3.2 的说法一致： "For any blocks where all the column indices are more than the row indices (approximately half of the blocks for large sequence length)， we can skip the computation of that block."

三种越界处理的分工必须分清：①K/V 的 `tl.load(..., other=0.0)` 补 0 只为让访存合法； ②`qk = tl.where(curr_n < n_ctx, qk, -inf)` 才是真正的剔除，**必须是 $-\infty$ 不是 0**——$e^{0-m}$ 会给每个越界列贡献一份假质量进 $l_i$ 与 $\mathrm{acc}$；③causal 的逐元素 mask 无条件套用，不判"是不是对角块"，谓词开销远小于一个控制流分支。改错会怎样：② 若省掉，S=777 这类非整除形状会**安静地算错**（多算 7 列假质量），正确性 gate 里 S=777 那格（scripts/test_fa2.py：41）就是为它设的。

**为什么"无条件套用谓词"比"判断是不是对角块"便宜**，这里要给硬件理由（§3.7.4）： 一个控制流分支若在 warp 内发散，硬件要串行执行两条路径；而 `tl.where` 降下来是 `selp`/谓词化指令，**不产生发散**。在 tensor core kernel 里，发散的代价还要加上 "mma 是 warp 级集合操作"(PTX §9.7.15)这条约束——发散会让集合操作无法对齐。 **所以"多算一点谓词"是在买"绝不发散"这个保证。**

**第 5 段 · online 更新的五行数学**(src/fa2_fwd.py：107-119)

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

角色：§3.2 的合并算子在 kernel 里的样子，整篇讲义的核心五行。逐行对应：`m_new` = $\max(m_A,m_B)$、`alpha` = 换基因子 $e^{m_A-m}$、`p` = 新块在新基准下的指数、`l_i` 与 `acc` 就是合并公式的两条。三个细节：①`alpha` 恒 $\le1$ 且 $m$ 单调不减，保证所有 `tl.exp` 的参数 $\le0$——**不会上溢是被结构保证的，不是运气**；②`p.to(v.dtype)` 把概率降回 fp16 走 tensor core，精度损失由 gate 兜底，是明确接受的交易；③`IEEE_DOT` 关掉 TF32 走真 fp32，只在校验路线用。改错会怎样：`l_i` 与 `acc` 少乘一个 `alpha`， 输出不会 nan 只会偏，只有与 fp32 精算逐元素比对才抓得住。

**②这个交易到底赔多少**（本讲义推导）：$p\in(0,1]$，fp16 在 $(0,1]$ 上的相对精度是 $2^{-11}\approx 4.9\times10^{-4}$；$\mathrm{acc}$ 是 $p$ 的加权和，相对误差不会超过单项相对误差（加权平均不放大相对误差），所以这一步引入的相对误差在 5e-4 量级。 gate 阈值 2e-2 比它宽 40 倍，实测 6 形状全部 ≤ 2e-3(EXP-T01 §5)。**换来的是 `tl.dot` 走 tensor core 而不是 SIMT**——按 §3.4.3 的比值，至少是 2 倍的算力差。 **这就是"明确接受的交易"应该长的样子：代价可算、收益可算、判据先锁。**

**第 6 段 · 归一化与写回：nan 为什么不落地**(src/fa2_fwd.py：121-129)

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

角色：FA2 相对 FA1 的"非 matmul FLOPs 削减"就落在这一次除法上（§3.4）。注释里那条 nan 推理链要能背下来：越界填充行全部列被 mask 成 $-\infty$ → $l_i=0$ → $0/0=\mathrm{nan}$， 但 store 的行 mask 保证这些行永不写出。**"内部允许出现 nan 但保证它出不去"是需要显式论证的设计。** 改错会怎样：mask 漏掉则输出尾部出现 nan，且只在非整除形状上出现，常规 shape 的测试全绿。

补一条**为什么"有效行 $l_i>0$"是可证的**（注释一带而过，这里补出）：causal 下第 $m$ 行至少可见列 $n=m$（对角元自身），该列的 $qk$ 值有限，故 $e^{qk-m_i}>0$，于是 $l_i \ge$ 某个正数。**非 causal 时更强**：整行都可见。唯一让 $l_i=0$ 的情形是 "所有列都被 mask 掉"，而这只发生在越界填充行——恰好被 store 的行 mask 拦住。 **边界情况被穷举完了，这才叫论证。**

**第 7 段 · flash-decoding partial kernel：同一套代数，标量版**(src/flash_decode.py：57-83)

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

角色：split-K 的"分"。与第 4/5 段逐行对照即知：**这是同一套 online softmax，只是行数从 BLOCK_M 变成 1**。三处刻意差异：①`hi = min(lo + split_size, n_ctx)` 把段边界与序列边界一起兜住，`kmask = curr < hi` 一个条件管两件事；②用广播乘 + 行内规约替代 `tl.dot`——$S_q=1$ 时 tensor core 的 M 维粒度是 16、要 pad，15/16 的算力全废（§3.7.3）， 而 decode 本就是带宽瓶颈，SIMT 反而干净；③全程 fp32，部分量要参与跨段换基。改错会怎样：`hi` 换回 `n_ctx` 则每段都扫完整条 KV，split-K 变成 splits 倍的重复劳动， **而结果依然正确**——最难查的一类性能 bug。

**②的量化依据**（本讲义推导，EXP-T04 协议 32K）：整条 KV 只读一次， FLOPs $= 2\times2\times H_q S_{kv} D = 2\times2\times16\times32768\times128 \approx 2.68\times10^8$，字节 $= 2\times(8\times32768\times128)\times2\,\mathrm{B} = 134.2\,\mathrm{MB}$，算术强度 $\approx 2.0$ FLOP/B。而 4090 的 ops：byte 是 $165.2\times10^{12}/1008\times10^9 \approx 164$ FLOP/B（NVIDIA GPU Performance Background User's Guide §4 的定义）。**2.0 对 164，差 82 倍——tensor core 在这里一点用都没有，放弃它不是妥协是正解。**

**第 8 段 · combine kernel：块间归并与空段湮灭**(src/flash_decode.py：115-125)

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

角色：split-K 的"合"，§3.2 代数的第二次使用。注释里那段推导要能当场写出来： $\sum_p l_p e^{m_p-m_g} = \sum_i e^{s_i-m_g}$ 是**全局行和**， $\sum_p \mathrm{acc}_p e^{m_p-m_g} = \sum_i e^{s_i-m_g}v_i$ 是**全局未归一化输出**， 相除即精确结果，与单 pass 同源而非近似。两个工程细节：①`tl.exp(m - m_g)` 对空段自动给 0，`next_power_of_2` 造出的空段不需特判（单位元性质在此兑现）；②每个 $(b,h)$ 用 **一个 program 串行收全部段**而非树规约或原子——splits 至多几百，直读更快且**求和顺序确定**、数值可复现。改错会怎样：改成 atomicAdd 则 $m_g$ 未知无法换基（§3.6.3）； `NUM_SPLITS` 传非 2 的幂则 `tl.arange` 编译期报错，launcher 的 `next_power_of_2` (src/flash_decode.py：147)就是这条约束的兑付。

②这条"求和顺序确定"的价值，在长上下文上是可以量化的：combine 的输入是 `(B, Hq, num_splits, D)` 的 fp32,32K 那格 splits=16，即每个输出元素是 16 项的和。 **16 项定序求和 vs 原子无序求和，前者跨运行逐位可复现，后者不可**。本仓选可复现， 理由与 flash_decode.py 文件头"fp32 是硬要求"同源：**部分量参与非线性换基，任何额外噪声都会被 $e^{\cdot}$ 放大。**

**第 9 段 · splits 启发式：三个上界的代码形态**(src/flash_decode.py：139-148)

```python
    if num_splits is None:
        # 填满 SM 的启发式:4090 有 128 个 SM,目标 B*Hq*splits ≳ 2×128
        # (每 SM 至少 2 个 CTA 才有延迟切换余地);上限 cdiv(Skv, block_n)
        # 保证段长不小于一个 BLOCK_N(再细切只剩空转)。未扫参(EXP-T04 §7)
        want = max(1, (2 * 128) // max(B * Hq, 1))
        num_splits = min(max(want, 1), triton.cdiv(Skv, block_n))
    # combine 里 NUM_SPLITS 喂 tl.arange,必须 2 的幂;向上取整可能造出
    # 空段,由 combine 的 scale=0 兜底
    num_splits = triton.next_power_of_2(num_splits)
    split_size = triton.cdiv(Skv, num_splits)
```

角色：§3.6.2 三个约束的代码落点，一行一条。值得指出的是**这段代码把"硬编码 128" 写进了源码**——4090 的 SM 数。换卡就错，而且是**性能错不是正确性错**，测试抓不到。生产实现会读 `torch.cuda.get_device_properties().multi_processor_count`；本仓为可读性保留字面量并在注释里写明来源。**把硬件常数写死并注明，好过写活但没人知道它从哪来**——但两者都不如从设备属性读。改错会怎样：把 `next_power_of_2` 去掉， `tl.arange(0, NUM_SPLITS)` 在编译期直接报错，报错信息指向 combine kernel 而不是这里，排查距离被拉长。

## 5. 实验数据怎么读

### 5.1 fig1:FA2 vs SDPA-flash

figures/fig1_fa2_vs_sdpa.png（脚本 scripts/plot_readme_figures.py:59-90）。

- **轴与口径**：横向条形图，x = 时延（ms，越短越好），y = 序列长 $S\in\{512,1024,2048,4096\}$；两条 = 本仓 FA2 简化版 / torch SDPA（flash 后端）； 误差条 = **3 轮 std**（非轮内 100 次迭代的分布）。源数据 exp-t01_stability_3rounds.csv。
- **百分比标签的定义**：`eff = sdpa_ms / ours_ms * 100`(plot_readme_figures.py：64)， 是**效率**不是加速比；读反了会把"达到官方的 87%"说成"比官方慢 87%"。四格效率 88/75/86/87% 非单调，不要挑数字讲（整表见 docs/theory/01_flashattention.md §3； 512 那格为何不可当真见 §6 第 2 条）。
- **87% 的四要素**（本仓措辞约定，缺一不引）：**简化版 / 仅 forward / 4K 形状（B1·H32/8·D128，fp16）/ 对照 = SDPA flash 后端**。3 轮口径 $1.1184\pm0.0015$ vs $0.9749\pm0.0024$ ms → **87.2%**；存盘单轮 1.119 vs 0.979 → 87.45%，**不进位**记 87%（EXP-T01 §5）。
- **机理账（可以心算的那种）**： $$\mathrm{FLOPs} = \frac{4\,B H_q S^2 D}{2}\Big|_{\text{causal}} = 2\times1\times32\times4096^2\times128 = 137.4\ \mathrm{GFLOP}$$（口径就写在 scripts/test_fa2.py：71-73）。除以 1.1184 ms 得 **122.9 TFLOPS**（与 derived 的 122.867 对上）；对 4090 fp16 tensor core 峰值 165 TFLOPS（docs/theory/04 §2 口径）= **74%**，与 kperf 卡片的 "算力 74%、occupancy 17%(regs 213)"逐点吻合（终端级证据，登记于 EXP-T06 §7）。 SDPA 侧 141 TFLOPS = 85%。**所以"差 13%"的准确说法是算力利用率 74% vs 85%**， 不来自算法差异（两边都是 FA2），而来自 tensor-core 布局微调、cp.async 双缓冲、 warp 专业化这三层抽象税（docs/theory/01 §4）。
- **这个设计防了哪些坑**：①**正确性 gate 先于性能**——6 形状（MHA、GQA 2:1/4:1、 S=777 非整除、双 head_dim、非 causal）max abs err ≤ 2e-3 全过，参考是 fp32 精算（scripts/test_fa2.py：22-31）；②**对照物命名诚实**——显式 `sdpa_kernel(SDPBackend.FLASH_ATTENTION)`(scripts/test_fa2.py：85-89)把后端钉死， 否则 SDPA 可能落到 math 后端，那就成了和另一个算法比；③**naive fp32 臂**给出量尺， S>2048 时如实记 NaN 不补估计值；④**3 轮 std** 除 S=512 那格外全部 ≤0.3%，所以 87.2% 与 87.45% 的差异是轮次噪声。

**把四格效率的非单调性讲清楚**（本讲义推导，给出可检验的解释而不是含糊带过）：

| S | ours (ms) | sdpa (ms) | 效率 | ours TFLOPS | 主要限制 |
|---|---|---|---|---|---|
| 512 | 0.0390±0.0009 | 0.0334±0.0001 | 88% | 55.1 | 两边都被 launch 地板盖住（§6 第 2 条） |
| 1024 | 0.1226±0.0003 | 0.0926±0.0003 | 75% | 70.0 | 并行度 8×32=256 个 program,SM 尾波明显 |
| 2048 | 0.3296±0.0007 | 0.2818±0.0003 | 86% | 104.3 | 开始进入 compute-bound |
| 4096 | 1.1184±0.0015 | 0.9749±0.0024 | 87% | 122.9 | compute-bound（算力 74%） |

1024 那格最低，用**波量化**(wave quantization)可以解释：program 数 = $S/\mathrm{BM} \times BH_q = 8\times32 = 256$，而 SM 数 128、每 SM 只能驻留 1 个 CTA(§3.7.4)， 所以恰好 2 个满波。但 causal 下各 program 的工作量**不等**（第 $m$ 块要扫 $m+1$ 个 K 块），两波之间负载严重不均，尾波拖尾。NVIDIA 的 "Matrix Multiplication Background User's Guide" §3.2 Wave Quantization 描述的正是这个现象（"a wave represents the maximum number of thread blocks that execute simultaneously"）。4096 那格 program 数 1024 = 8 个波，不均被摊平，效率回升。 **这是推断：本仓未做逐 program 计时，无法直接验证尾波假设**（无计数器权限， docs/theory/04）。

### 5.2 flash-decoding 的三臂表:2.24× 与 5.17× 差在哪

data/derived/exp-t04_stability_3rounds.csv（3 轮，协议 B1·Hq16/Hkv8·D128·bf16, scripts/test_flash_decode.py:37）:

| Skv | flash_decode (ms) | naive[repeat 预置] | 提速 | naive[含 repeat] | 提速 |
|---|---|---|---|---|---|
| 512 | 0.0907±0.0022 | 0.0779 | **0.86±0.02×** | 0.1243 | 1.37±0.01× |
| 2048 | 0.0941±0.0024 | 0.0816 | **0.87±0.01×** | 0.1271 | 1.35±0.02× |
| 8192 | 0.0918±0.0026 | 0.0804 | **0.88±0.01×** | 0.1726 | 1.88±0.05× |
| 32768 | 0.1515±0.0073 | 0.3393 | **2.24±0.11×** | 0.7822 | **5.17±0.24×** |

**为什么要拆三臂**：GQA 原生 kernel 免掉的正是 `repeat_interleave` 那份实体化拷贝（scripts/test_flash_decode.py：16-20）；算进对照物得 5.17×，不算进得 2.24×。**两个都对，但报数字必须带口径**——这就是拆三臂而不是选一个好看的报的原因。

**机理账（把加速比算回字节比）**：三臂在 32K 那格全部带宽受限，加速比应当约等于搬运字节比。逐臂列式（bf16,2 B/元素）：

- flash_decode：只读未 repeat 的 KV，$2\times(8\times32768\times128)\times2\,\mathrm{B} = 134.2\,\mathrm{MB}$（中间量 Accp 仅 131 KB，可忽略）；$/0.1515\,\mathrm{ms} = 886\ \mathrm{GB/s}$ = 1008 峰值的 **88%**。
- naive[repeat 预置]：读 16 头 KV = 268.4 MB + 物化分数矩阵往返约 4 MB $\approx 272.6\,\mathrm{MB}$；$/0.3393\,\mathrm{ms} = 803\ \mathrm{GB/s}$ = **80%**。
- naive[含 repeat]：再加 repeat 的写 268.4 MB + 读 134.2 MB，共 $\approx675\,\mathrm{MB}$; $/0.7822\,\mathrm{ms} = 863\ \mathrm{GB/s}$ = **86%**。

字节比 $272.6/134.2=2.03$（实测 2.24）、$675/134.2=5.03$（实测 5.17）：**两个口径的差就是 repeat 那份拷贝的读写**；拆开看，2× 来自 GQA 不 repeat($H_q/H_{kv}=2$)，剩下的 2.5× 来自 repeat 本身。（字节数按协议推算、时间为 3 轮实测；算式与实测 10% 内的偏差来自小张量 kernel 的效率差异，未单独隔离。）

**Accp 那 131 KB 是怎么算的**（补出中间步）：`accp` 形状 $(B, H_q, \text{splits}, D) = (1,16,16,128)$ fp32 $= 131072\,\mathrm{B}$，加上 `mp`/`lp` 各 $(1,16,16)$ fp32 = 1 KB。写一遍读一遍共约 262 KB，占 134.2 MB 的 **0.2%**——**split-K 的额外流量在长上下文上完全可以忽略，这正是它在长上下文划算的机制原因**。反过来在 512 那格：KV 只有 2.1 MB，262 KB 占 12%，再加上多一次 launch，划不来就是必然。

**短上下文反亏 0.86-0.88× 的机理账**：fd 在 512/2048/8192 三格几乎不变（0.0907 / 0.0941 / 0.0918 ms），naive 同段也平（0.0779 / 0.0816 / 0.0804）。**两条线都平，说明这一段的时间根本不在设备上**，都是主机侧地板：fd 每次要发 2 次 Triton launch (partial + combine)，naive 是 3 个 torch 算子但走 C++ 分发，而同机实测的分发成本是 Triton eager $36.2\pm0.1\,\mu s$ vs torch $8.03\pm0.74\,\mu s$（EXP-T05《CUDA Graph 消 launch 开销实测》3 轮）。所以 0.86-0.88 是**两条地板之比，与 KV 长度无关**——不是算法输了，是这一档 launch 口径输了。正解在讲义 03：CUDA Graph 把 launch 塌缩 11.6×。

**把这条推断量化到可证伪的程度**（本讲义推导）：若时间全在主机侧，则 fd/naive $\approx (2\times T_{\text{triton}})/(3\times T_{\text{torch}}) = (2\times36.2)/(3\times8.03) = 72.4/24.1 = 3.0$——**这与实测的 1.16(= 1/0.86) 差得远**。所以"纯主机侧地板"这个解释是**过强的**：真实情况是 fd 的两次 launch 与设备侧执行有重叠（§2 的 $\max$ 而非 $\sum$，讲义 03 §2），而 naive 的三个 torch 算子各自的设备时间也不为零。诚实的结论只能是：**这一段两条线都平、都不随 KV 长度变化， 说明设备侧不是瓶颈；但精确的归因需要设备侧计时，本仓没做。** 把一个方向正确的推断写成"约等于两条地板之比"是过度自信，这里改成"同一层的两个常数之比，具体配比未隔离"。

**另一组形状变体**（Qwen3-8B 形状 H32/fp16,3 轮，raw 在 data/raw/EXP-T04/）：32K 上 $8.96\pm0.13\times$（记 9.0×），口径是 **repeat 内计**。头数 16→32、dtype 换 fp16， 数字就换一套——"数字必须带形状定语"的又一个实例。为什么头数翻倍收益变大： $H_q/H_{kv}$ 从 2 变成 4，repeat 那份拷贝也从 2 倍大变成 4 倍大，免掉的字节更多。 **方向可以预测，倍数不能**：实测从 5.17× 涨到 8.96×（比值 1.73），而单看 GQA 组从 2 变 4，repeat 拷贝的字节比只从 2 变 4、免掉的相对量对应的理想倍数并不等于 1.73； 差额里还混着 bf16→fp16 这一处改动。**两个变量同时改，不能做单变量归因**——如实记为 "两处同时变化，未做单变量对照"。

### 5.3 哪些数字能外推,哪些不能

| 数字 | 能外推的部分 | 不能外推的部分 |
|---|---|---|
| 87%(EXP-T01) | "算力利用率 74% vs 85%"这个**归因结构** | 87 这个值，只属于 4K/fp16/B1·H32/8·D128 |
| 2.24× / 5.17×(EXP-T04) | "两个口径差 = repeat 的读写"这条**账** | 倍数值，只属于 32K/bf16/H16/8 |
| 0.86-0.88×(EXP-T04) | "短上下文 split-K 不划算"这个**方向** | 具体比值，依赖两侧的 launch 成本 |
| 6.1e-5（误差） | "online softmax 是精确算法，误差只来自舍入"这条**性质** | 具体误差值，依赖 dtype 与 $S_{kv}$ |
| 17% occupancy | 可以从 regs 与 smem **推出来**(§3.7.4) | 换 tile 就换一套 |
| 160 KB / 99 KB | 硬件常数，**换卡才变** | 无 |

## 6. 误区与边界

至少踩过一次才写得出来的错误直觉（第 3 条是本仓自己被证伪的假设）：

1. **"FA 把 HBM 流量降到 $O(S\cdot D)$"**——半对，而且连"半"都要修正。Theorem 2 给的是 $\Theta(N^2d^2M^{-1})$，不是 $\Theta(Nd)$。被消掉的是不可缓存的 $S\times S$ 写回；K/V 仍被 $S/\mathrm{BM}$ 个 Q 行块各读一遍（§3.3.3 的三口径夹逼：83.9 MB / 268 MB / 1.07 GB）。正确的调优方向是"BM 越大重读越少"，这才和实测的 BM 64→128 +17% 对得上。
2. **"S=512 那格 88%，说明小序列上我们也接近官方"**——错。0.0390 ms 恰是同机实测的 Triton 每调用 launch 地板（$36.2\pm0.1\,\mu s$，EXP-T05），kernel 本体被盖住， **该点的 kernel 级差距在本仓协议下不可测**；要测就得先上 CUDA Graph 或改用设备侧计时。
3. **「flash-decoding 总比 naive 快」**——**被本仓自己的复测证伪**。EXP-T04 §5 原表里 Skv ≤ 8192 的各行已全部作废（未存脚本的混合口径，不可复现），3 轮实测是 **0.86-0.88× 的反亏**，它的正确定位是**长上下文武器**。方法论提炼：**旧数字不可复现时，作废它比解释它更诚实**；拆成三臂口径后，同一批数据同时给出了「反亏」与「5.17×」两个真相。
4. **"online softmax 是近似算法"**——不是。换基是精确恒等式（§3.2），误差只来自浮点舍入：flash_decode 在 Skv=512/2048 两格与 fp32 精算的 max abs err **恰为 0.0** (data/raw/EXP-T04/20260825T152434_flash_decode_stability_r1.json)，32K 也只有 6.1e-5。
5. **"decode 用同一个 FA2 kernel 就行"**——不行。$S_q=1$ 时 M 维 tile 全废，grid 塌成 $B\cdot H_q$(§3.6)，且 `mma` 的 M 维粒度是 16、pad 之后 15/16 的 tensor core 算力空转。换并行轴不是优化，是换算法结构。
6. **"`tl.dot` 会因为 M<16 编译报错"**——本机 Triton 3.6.0 上不会（§3.7.3： `min_dot_size` 返回 `(1,1,16)`，注释明写小 M/N 靠 padding 走 tensor core）。 **代价存在，报错不存在**；把代价说成报错，会让人去写没必要的 pad 逻辑。
7. **"占用率 17% 说明 kernel 没写好"**——本仓 FA2 算力利用率 74%，占用率 17% 是 fp32 累加器（64 regs/线程）与大 tile 换来的（§3.7.4）。占用率只在**带宽 % 与算力 % 两个都低**时才是嫌疑人（docs/theory/04 §2）。
8. **"BLOCK_N=128 一定 OOM"**——不一定。EXP-T08 实测：stages=2 那一档能编译（96 KB）， 只是寄存器打到 255/线程；OOM 出现在 stages=3(160 KB)。**"撞哪堵墙"取决于另一个参数**，这正是片上预算是**一份**而不是几份的证据。

**适用边界**：全部数字来自单卡 RTX 4090、fp16/bf16、合成随机输入、**仅 forward**； 87% 不含 backward/dropout/alibi/paged，且只在 B1·H32/8·D128·S=4096 一个形状上成立； flash-decoding 的 2.24×/5.17× 限 B1·H16/8·D128·bf16 协议的 32K 那一点，H32/fp16 变体另有一套数字；splits 启发式未扫参（EXP-T04 §7）；FA2 的 warp 级划分（论文 §3.3） 本仓未实现、交给编译器；logsumexp 未存，故本 kernel 无法直接支撑 backward； kperf 卡片是终端级证据（本容器无性能计数器权限，docs/theory/04）； cp.async 的 commit/wait 分组数与缓冲份数的对应关系未在本仓直接观测（EXP-T08 §7）。

## 7. 连环追问

1. **Q：softmax 减 max 是为了精度还是为了不溢出？** 为了不溢出——减 max 在实数域上是**精确恒等式**（§3.1.1 第 2 步），顺带保证分母 $\ge1$、不出现 $0/0$。fp16 的溢出线只有 $x>11.09$，真实模型的 attention logits 轻易越过它，这不是理论洁癖。追问一层：为什么不换 bf16？bf16 阈值是 88.72， 确实安全得多，但那是指数位数带来的，不是精度；而且换 dtype 不能解决"存在会炸的输入"这个一般问题（§2.4）。
2. **Q：分块之后为什么还能算对？** 因为 $e^{s-m'}=e^{s-m}e^{m-m'}$，换基因子与求和下标无关、可提到求和号外；于是 "旧部分量乘一个标量"就换到新基准，合并退化为普通加法（§3.2）。更严格的说法： $(m,l,\mathrm{acc})$ 在 $\oplus$ 下构成**交换幺半群**，所以任意分块与任意归并顺序给出同一结果。
3. **Q：为什么最后才除 $l$？** 先除就丢了 $l_p$ 权重，合并要写成加权平均、$l_p$ 还得带着走（§3.2.3）——"最后再除" 是可归并性的自然形式。代码落点 src/fa2_fwd.py：125。FA2 论文 §3.1 把它写成 "maintain an 'un-scaled' version of $O^{(2)}$"。
4. **Q：FlashAttention 的定理到底证明了什么？** 三条：Theorem 1 说算法正确且额外内存 $O(N)$、FLOPs 仍是 $O(N^2d)$（**没省算力**）； Theorem 2 说 HBM 访问从 $\Theta(Nd+N^2)$ 降到 $\Theta(N^2d^2M^{-1})$； Proposition 3 说在 $M\in[d,Nd]$ 上不存在渐进更优的精确算法。**下界那条才是这篇论文与普通优化工作的分界。**
5. **Q：FA2 相对 FA1 改了什么？** 论文列三条（§3.1/§3.2/§3.3）：减少非 matmul FLOPs、seq 维并行、warp 间改用 split-Q。本仓实现了前两条，第三条交给 Triton 编译器——这正是 §5.1 那段抽象税的一部分（§3.4.1 的表）。补一句机制：循环反转让三元组常驻寄存器，省掉 FA1 那笔与 $S^2$ 同量级的中间量往返（§3.4.2 的 136 MB 算式）。
6. **Q：GQA 在 kernel 里改了几行？收益从哪来？** 一行 `hkv = hq // GQA_GROUP`(src/fa2_fwd.py：55)。收益是 KV 读取量按 $H_{kv}/H_q$ 缩小且**不物化 repeat**——这份"免掉的拷贝"在 EXP-T04 里被单独标价（2.24× 与 5.17× 的差就是它）。诚实补充：EXP-T01 里没有 MHA 对照臂，所以 "GQA 省了多少"在那个实验里**测不出来**。
7. **Q：你的 87% 具体差在哪一层？** 算力利用率 74% vs 85%（§5.1 机理账），不是算法差异；缺口在 tensor-core 布局微调、 cp.async 双缓冲、warp 专业化（论文 §3.3 的 split-Q）。本仓立场：讲得清的 87% 好过讲不清的 100%。
8. **Q：decode 为什么不能复用 FA2 kernel？** $S_q=1$ → grid 第一维 = 1 → program 数 $=B\cdot H_q$，本仓协议下 16 个对 128 SM (§3.6.1)；且 `mma` 的 M 维粒度 16 使 tile 被 pad。必须换并行轴到 KV 维。
9. **Q：split-K 为什么要两个 kernel？能不能用 atomicAdd 省一次 launch？** 不能——除非放弃自适应基准。归并每项要乘 $e^{m_p-m_g}$，而 $m_g$ 要等所有段算完； atomicAdd 只能做无状态的可交换加法（§3.6.3）。唯一绕法是取一个先验安全上界 $\hat m$ 当固定基准，代价是回到"要么上溢要么整行下溢"的老问题。
10. **Q：那 splits 是不是越多越好？** 不是，三个上界依次收紧（§3.6.2）：填满 SM 的下需求、$\lceil S_{kv}/BN\rceil$ 的算法上界、`next_power_of_2` 的编译期约束。而且本仓的启发式**未扫参**(EXP-T04 §7)，把它当调优结论引用是不诚实的。
11. **Q：BLOCK_N 为什么不能取 128？** Ada 每 thread block 的 shared memory 上限是 99 KB = 101376 B(Ada Tuning Guide §1.4.1.1)；BN=128 + stages=3 需要 163840 B，编译期直接 `OutOfResources: Required 163840, Hardware limit 101376`（EXP-T08 逐字复现）。补一层：BN=128 + stages=2 其实能编译（96 KB），但寄存器打到 255/线程（Ada 每线程上限）——**换了一堵墙撞而已**。
12. **Q：occupancy 只有 17%，你不慌吗？** 不慌，而且它是**可推的**不是观测的：regs 213 × 256 线程 = 54528 > 65536/2， 所以每 SM 只能放 1 个 CTA；8 warps / 48 = 17%(§3.7.4)。shared memory 那条路（64 KB / 100 KB）给出同一答案。算力利用率 74% 说明延迟已经藏住了。
13. **压力问 Q：2.24× 和 5.17× 你到底该报哪个？会不会是挑了个好看的？** 诚实答：两个都要报，并说清它们的差就是 `repeat_interleave` 的写+读（§5.2 的字节账把这点算死了）。若上游引擎的 attention 契约已把 repeat 好的 KV 传进来（本仓 llm-engine 接线当时正是如此，EXP-T04 §6），GQA 原生那部分收益**根本兑现不了**， 该报 2.24×；能控制契约时 5.17× 才真实可得。所以正确说法是"口径 A 下 2.24×、口径 B 下 5.17×"，而不是二选一。
14. **压力问 Q：87% 换个形状还成立吗？换成 backward 呢？** 不能承诺。同一份代码在 S=1024 上只有 75%，S=512 那格的 88% 甚至不可测（§6 第 2 条）——**87% 是 4K 那一个点的值**，四要素限定就是为此存在。backward 更是另一件事：要重算 P（存不下），需要存 logsumexp（本仓没存，FA2 论文 §3.1 明说反向只需要 $L=m+\log \ell$），且 dQ 与 dK/dV 的归约方向不同（行向 vs 列向）。本仓只做 forward 并如实声明，不外推。
15. **压力问 Q：你说 IO 复杂度定理是核心，但你的 kernel 里哪一行体现了它？** 诚实答：**没有哪一行直接体现，定理体现在"没有哪一行去写 $S\times S$"这件事上**。定理的作用是**告诉你哪一类实现不可能更优**(Proposition 3)，从而让你停止寻找 "更聪明的精确 attention"，转而把力气花在常数因子上——本仓的 BM 扫描、tile 与 shared memory 预算的博弈，全是常数因子的工作。**理论负责划定边界，工程负责在边界内取常数**，这是本篇最重要的一句方法论。

## 8. 工业对照与延伸

### 8.1 论文/文档怎么说 vs 本项目实测:逐条对照

本节把"论文或官方文档说了什么"与"本仓在单卡 RTX 4090 上测到什么"并排放，并诚实分析差异来源。差异不粉饰：多数来自硬件代际、规模与口径，少数来自本仓的实现简化。

| # | 来源与声称 | 本仓实测（EXP 锚） | 差异分析 |
|---|---|---|---|
| 1 | online softmax（arXiv:1805.02867 摘要）："Softmax accelerates by up to 1.3x and Softmax+TopK combined and fused by up to 5x" | 本仓无"向量 softmax 三趟 vs 一趟"对照臂，**无法验证** | 论文测的是独立的 softmax 算子；本仓的 online softmax 嵌在 attention 里，省掉的是 $S\times S$ 而不是一趟向量读。**同一算法在两个上下文里收益差两个数量级**(§3.1.4)，1.3× 这个数不能引到本仓 |
| 2 | FlashAttention Theorem 2：HBM 访问 $\Theta(N^2d^2M^{-1})$ | 本仓未测 HBM 计数器（无权限）；按 tile 模型推得三口径 83.9 MB / 268 MB / 1.07 GB(§3.3.3) | 定理是**渐进阶**，常数被 $\Theta$ 吸收，无法用来预测具体字节数；本仓的三口径夹逼与"算力占 74%"一致。**定理不可证伪于单点实测，这是理论与实验的正常关系，不是矛盾** |
| 3 | FlashAttention 论文：A100 SRAM 192 KB/SM、带宽约 19 TB/s | RTX 4090：每 SM 100 KB shared（每 block 上限 99 KB）、L1/shared 合计 128 KB/SM(Ada Tuning Guide §1.4.1.1/§1.4.2.2) | **代际差**：Ada 的片上预算比 A100 小近一半，所以论文按 $M\approx$ 192 KB 推的块大小在 4090 上要缩。本仓 BN=128 的 OOM 正是这个代际差的直接后果 |
| 4 | FlashAttention-2 摘要："around 2× speedup compared to FlashAttention， reaching 50-73% of the theoretical maximum FLOPs/s on A100" | 本仓 FA2 简化版在 4090 上达 **74%** 峰值（122.9/165.2，EXP-T01 3 轮） | **数字接近但不可直接比**：论文的 50-73% 是 A100 官方 CUDA 实现在多形状上的区间，本仓是 4090 单形状的 Triton 简化版。落在同一区间是巧合级的一致，不能当作"追平官方"的证据 |
| 5 | FlashAttention-2 §3.3：FA2 改用 split-Q，"split Q across 4 warps while keeping K and V accessible by all warps" | 本仓**未实现**，warp 级划分交给 Triton | 这是本仓与官方实现差距的一个已知来源（§5.1 的"抽象税"）。**能指出自己缺哪一条，好过笼统说"实现没那么优化"** |
| 6 | FlashAttention-2：A100 matmul 312 vs 非 matmul 19.5 TFLOPs/s，比值 16× | 4090 上按白皮书 Table 2 算是 165.2 / 82.6 = **2.0×**（本讲义推导） | **代际差**：Ada 每 SM 128 条 FP32 通道，非 matmul 相对不那么贵。推论是"省非 matmul FLOPs"在 Ada 上边际收益更小——**但本仓没做关掉该优化的对照，标为推断** |
| 7 | flash-decoding 官方博客："up to 8x faster generation for very long sequences"，微基准 B=1/seqlen=65536 时 FA2 2300.6 µs vs Flash-Decoding 64.4 µs | 本仓 32K 上 **2.24×（repeat 预置）/ 5.17×（含 repeat）**；$S_{kv}\le$ 8K 反而 **0.86-0.88×** | **对照物完全不同**：博客的分母是 FlashAttention v2 的 decode 路径，本仓的分母是 naive torch attention；博客的机器是 A100、序列到 128K。**"8×"与"5.17×"不是同一个量的两次测量**。短上下文反亏在博客里没有出现，因为它没测那一段 |
| 8 | flash-decoding 官方博客：batch size 1 时 FlashAttention "will use less than 1% of the GPU" | 本仓 decode grid = $B H_q$ = 16 个 program 对 128 SM = **12.5%** | **分母定义不同**：博客说的是 batch 维塌陷后的整体占用；本仓把 head 维也算进 program 数。机制同一个，数字差一个量级——**引用时必须说清分子分母** |
| 9 | Triton 文档（`triton.Config`）："num_stages： the number of stages that the compiler should use when software-pipelining loops. Mostly useful for matrix multiplication workloads on SM80+ GPUs"；`tl.range` 文档："pipeline the loop into this many stages (so there are num_stages iterations of the loop in flight at once)" | EXP-T08 编译期探针：缓冲份数 = **max(1， num_stages − 1)**，故 stages=2 只有 1 份缓冲 | **文档描述的是"在飞的迭代数"，不是"缓冲份数"**；两者相差 1 是流水线的正常结构（消费中的那一级不额外占一份预取缓冲）。本仓的探针把这条差异量化了，讲义 02 §3.3 讲透。这不是文档错，是**读者容易把两个量当成一个** |
| 10 | Ada Tuning Guide §1.4.1.1："The maximum shared memory per thread block is 99 KB" | EXP-T08 逐字复现编译器报错 `Required 163840, Hardware limit 101376`，101376 = 99×1024 | **文档 → 实测完全闭合**，本篇唯一一条严格的"预言—证实"链。文档只给上限，本仓把撞墙那一刻的字节数也钉死了 |
| 11 | PTX ISA §9.7.9.26.3.3：`cp.async.wait_group` "wait till only N or fewer of the most recent cp.async-groups are pending" | 本仓**未 dump PTX**，组数与缓冲份数的对应关系未观测（EXP-T08 §7） | 标为**未核实**。本仓只到"份数"这一层，再往下需要 TTGIR/PTX 级证据 |
| 12 | Triton 3.6.0 源码 `min_dot_size` 返回 `(1,1,16)`，注释 "For small M/N the input we can still use tensorcores with padding" | 本仓源码注释写"tl.dot 要求 M≥16 得 pad"(src/flash_decode.py：53-54) | **本仓注释措辞偏松**：M=1 不会编译失败，只是被 pad。代价的结论不变，机制的表述以 §3.7.3 为准。**这是本篇主动指出的自家表述问题** |

**总结这张表的读法**：十二条里只有第 10 条是严格的"预言—证实"闭环；第 9、12 两条是 "文档/源码语义比本仓旧表述更准确"；其余多为"不可比"或"代际/口径不同"。**论文的数字几乎从不能直接搬到你的机器上，能搬的是机制与判据。**

### 8.2 与生产实现的差距各在哪一层

- **官方 FlashAttention(CUDA)**：算法同构，差在实现层——tensor-core 的 swizzle/布局微调、cp.async 双缓冲、warp 专业化（论文 §3.3 的 split-Q，producer/consumer 分工）。本仓把这三层交给 Triton 编译器，代价就是 §5.1 里那段算力利用率差。
- **FlashAttention-3(arXiv:2407.08608)**：Hopper 专属的下一代，靠 "asynchrony of the Tensor Cores and TMA"做 warp 专业化、把 matmul 与 softmax 交错、并加 FP8 的 block quantization；报的是 H100 上 FP16 740 TFLOPs/s（75% 利用率）。 **本仓的 Ada(sm_89)没有 TMA、没有 wgmma，这条路整条走不了**——不是没做，是硬件代际不支持（界线见讲义 02 §3.5）。
- **vLLM paged attention**：本 kernel 的数学 + block table 间接寻址。本仓 K/V 指针是连续 stride 寻址（src/fa2_fwd.py：87-90），paged 版换成"先查页表再算页内偏移"， 数学一行不改。
- **vLLM / SGLang 的 decode kernel**：paged 读 + split 归并的合体，本仓 flash_decode 只做后半。两者正交：paged 解决"KV 在哪"，flash-decoding 解决"并行度从哪来"。
- **splits 的选择**：官方 flash-decoding 博客说 "the number of splits determined by a heuristic at launch"；本仓的启发式（§3.6.2）形式相近但**未扫参**，而且把 SM 数硬编码成 128。生产实现会读设备属性并按 batch/seqlen 查表。
- **引擎接线这一层的坑**：kernel 快不等于引擎快。本仓接进 llm-engine 时首版对 KV cache 切片做了 `.contiguous()`，每层每步整拷一遍 KV，TPOT 反而变差（EXP-T04 §6）—— **隐藏拷贝藏在调用约定里。**

### 8.3 这一篇没做的事(供下一步)

- backward：需要存 logsumexp(FA2 §3.1)、重算 P、处理 dQ 与 dK/dV 归约方向不同； 本仓 forward-only。
- warp 级划分（split-Q）：Triton 不暴露，需要 CUDA/CUTLASS 才能做。
- PTX 级验证：cp.async 的 commit/wait 组数、mma fragment 的实际布局（EXP-T08 §7）。
- splits 扫参与设备侧计时：前者是 EXP-T04 §7 的开放项，后者是让 S=512 那格变得可测的前提（§6 第 2 条）。
- paged / 变长 batch：本仓的 stride 寻址假设 KV 连续。

### 8.4 延伸阅读(带精确出处,每条一句话说明它能解决什么疑问)

**论文**

1. Milakov & Gimelshein， "Online normalizer calculation for softmax"， arXiv:1805.02867，Algorithm 3 与 Theorem 1。——想看"三趟压成一趟"的原始递推式与它的归纳法证明（本篇 §3.1.3 补出的中间步就是照着这里补的），读这两处。
2. Dao， Fu， Ermon， Rudra， Ré， "FlashAttention： Fast and Memory-Efficient Exact Attention with IO-Awareness"， arXiv:2205.14135，Theorem 1、Theorem 2、 Proposition 3、Algorithm 1（块大小 $B_c=\lceil M/4d\rceil$）。——想弄清 "FlashAttention 到底证明了什么、下界为什么重要、块大小里那个 4 从哪来"，读这四处。
3. Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", arXiv:2307.08691,§3.1 Algorithm、§3.2 Parallelism、 §3.3 Work Partitioning Between Warps。——想知道"循环反转具体改了哪三处""为什么非 matmul FLOPs 值 16 倍""split-K 与 split-Q 差在哪"，读这三节。
4. Rabe & Staats， "Self-attention Does Not Need $O(n^2)$ Memory"， arXiv:2112.05682。——想知道 online softmax 用到 attention 上的最早形态，以及 "内存 $O(\log n)$"与"HBM 访问 $\Theta(N^2d^2M^{-1})$"是两个不同的目标函数，读它。
5. Shah， Bikshandi， Zhang， Thakkar， Ramani， Dao， "FlashAttention-3： Fast and Accurate Attention with Asynchrony and Low-precision"， arXiv:2407.08608。——想知道 "为什么 Ada 上这条路走不了"以及 warp 专业化 + TMA 能再拿多少（H100 FP16 740 TFLOPs/s、75% 利用率），读它。
6. Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai, "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", arXiv:2305.13245。——想回答"GQA 是带宽优化还是质量折中""从 MHA checkpoint 转过来要多少算力"，读它。
7. Vaswani et al.， "Attention Is All You Need"， arXiv:1706.03762，§3.2.1 及其脚注。——$1/\sqrt{d_k}$ 的方差论证，本仓 `sm_scale` 默认值的唯一理论出处。
8. Williams, Waterman & Patterson, "Roofline: an insightful visual performance model for multicore architectures", CACM 52(4):65-76, DOI:10.1145/1498765.1498785。——想把"算术强度 2.0 对 ops:byte 164"这类判断放进一个统一框架，读它。

**官方文档**

9. NVIDIA， "NVIDIA Ada GPU Architecture" 白皮书，Appendix A Table 2 与 SM 结构段。——所有硬件常数（128 SM、1008 GB/s、73728 KB L2、165.2 TFLOPS、每分区一个 warp scheduler + 一个 dispatch unit）的唯一出处；§3.7.4 的占用率算术全靠它。
10. NVIDIA, "Ada Tuning Guide"(docs.nvidia.com/cuda/ada-tuning-guide), §1.4.1.1 Occupancy、§1.4.2.2 Unified Shared Memory/L1/Texture Cache。——想知道 99 KB / 100 KB / 48 warps / 255 regs 这四个上限的官方原文，以及 48 KB 以上的动态 shared memory 需要显式 opt-in，读这两节。
11. NVIDIA PTX ISA，§9.7.9.26 Asynchronous copy(cp.async / commit_group / wait_group)与 §9.7.15 Warp Level Matrix Multiply-Accumulate（matrix fragments、 warp-wide collective、m16n8k16 的 `groupID = %laneid >> 2` 布局）。——想弄清 "预取到底是什么指令""为什么等待的粒度是组""为什么不能在 dot 周围写发散分支"， 读这两章。
12. NVIDIA， "GPU Performance Background User's Guide"，§4 Understanding Performance。——算术强度、ops：byte、三个限制因子（memory bandwidth / math bandwidth / latency）的官方定义，以及"thread blocks 要几倍于 SM 数"的原话。
13. NVIDIA， "Matrix Multiplication Background User's Guide"，§3.2 Wave Quantization。——§5.1 里"1024 那格效率最低"的尾波解释就是照这一节的定义写的。
14. PyTorch 官方博客， "Flash-Decoding for long-context inference" (pytorch.org/blog/flash-decoding/)。——split-K 三步法的原始表述、 "batch size 1 时用不到 1% 的 GPU"、以及 CodeLlama-34B 与微基准表； §8.1 第 7/8 条的对照原件。
15. Triton 官方 API 文档与本机 3.6.0 源码：`triton.Config` 的 num_stages 说明、 `tl.range` 的 "num_stages iterations of the loop in flight at once"、 `triton/backends/nvidia/compiler.py` 的 `min_dot_size`。——想核实 "num_stages 到底承诺了什么"与"`tl.dot` 的真实形状下界"，读这三处。

**源码与本仓证据**

16. src/fa2_fwd.py：107-125—— online 更新五行 + 唯一一次归一化，整篇讲义的核心。
17. src/flash_decode.py：115-123—— 块间归并代数与空段湮灭的注释推导。
18. src/flash_decode.py：139-148—— splits 启发式的三个上界（含硬编码 128 的来历）。
19. records/EXP-T08_smem_stage_probe.md §5-§7—— 编译期资源探针的原始表格： 份数 = max(1， num_stages−1)、160 KB 的逐字复现、以及"未 dump PTX"的开放项。
20. records/EXP-T04_flash_decoding.md §5-§7—— 三臂口径拆分、小 Skv 行作废的完整过程， 以及引擎接线时那笔 `.contiguous()` 隐藏拷贝。
21. records/EXP-T01_fa2_forward.md §5-§7—— 6 形状正确性 gate、tile 扫描结论与 "终端级证据"的限定。
22. docs/theory/01_flashattention.md §2 第 3 步（FA1→FA2 三处改动）与 §3（完整效率表）。
23. docs/talk/whiteboard_card_fa2_algebra.md—— 白板推导卡，含"为什么不能每块先除"。
