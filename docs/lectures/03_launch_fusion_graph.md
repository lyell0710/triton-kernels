# 讲义 03 · "Triton 比 CUDA 慢"的四层拆解:设备侧、launch、融合、CUDA Graph

> 读者：准备校招面试的作者本人，以及被"小 kernel 慢了 4 倍"卡住的工程师。读法：不跳步。每个论断后面跟着它的证据锚（EXP 编号 / 文件：行号 / raw 路径）， 所有数字与仓内现行口径逐字一致，来源见 records/ 与 data/derived/。引用规范：凡属论文/官方文档的论断，给出标题 + arXiv/DOI 编号 + 章节编号（文档给 URL 路径 + 小节名）；凡属本讲义补出的推导或折算，行内标注"本讲义推导"； 无法用检索确认的说法标注"未核实"。

## 1. 这一篇回答什么问题

"Triton 比 CUDA 慢吗"是本仓明令**禁止裸答**的问题（本仓措辞约定）。原因不是政治正确，是这句话在本仓的数据里同时有三个互相矛盾的答案。读完你应当能：

- 手推**三点法**：为什么用"极小尺寸 + 目标尺寸 + 带宽主导尺寸"三个点，就能把 "时间到底在不在 kernel 里"这件事**证明**出来，而不是猜出来；并且知道本仓的数据里还藏着**第四个点**，它在事后独立地支持了同一结论（§3.2.3）。
- 说清四层因果链：设备侧同速（8192² softmax **917 / 921 GB/s**，双双贴 roofline 91%） → 主机侧分发差（Triton ~30 µs > torch ~8 µs > 裸 CUDA ~5 µs）→ 端到端被**融合数** 反转（1 次 launch 的 Triton 融合赢 4 次 launch 的 ext-CUDA）→ CUDA Graph 把 launch 塌缩 **11.6×（36.2 → 3.11 µs/调用）**。
- 答上"int8 那三个数字到底哪个是哪个"：**5.9 µs（裸 CUDA v4，scale 预置）/ 65.1 µs（ext 绑定端到端，4 次 launch）/ 51.7 µs（Triton 单 kernel 融合，跨会话 41.6~52 µs 波动）**——三口径不得混引，且 binding 端到端两数**仍为单轮**。
- 算出**这条主机侧地板在什么尺寸上会被设备侧盖过去**（本讲义推导：约 36.5 MB 的读写量，对 fp32 方阵约 $2136^2$），从而知道"该不该上 Graph"是一个**可以先算再测** 的问题，不是拍脑袋。
- 拿出一棵能当场画的决策树：什么时候用 Triton、什么时候写 CUDA、什么时候上 Graph。

### 1.1 本篇要建立的五条能力

1. **口径拆分能力**：遇到任何"A 比 B 慢 N 倍"，先问"你测的是 kernel、是一次调用、还是一条调用链"，再问"三者中哪一个是你的负载真正关心的"。
2. **实验设计能力**：能设计一组"如果不是 X 会怎样"的对照，把机制**证明**出来而不是猜出来；并且知道一个被证伪的假设为什么值得留在仓里。
3. **两条时间线的模型**：知道主机侧与设备侧在不同步的循环里是**流水并行**的，所以测到的是 $\max$ 而不是 $\sum$；知道哪些写法（每次 `synchronize()`）会把 $\max$ 破坏成 $\sum$。
4. **融合的双重收益**：能分别算出"少一次 launch 省多少"与"少一份中间张量往返省多少"， 并知道后者在带宽主导的算子上通常更大。
5. **Graph 的边界**：知道捕获需要什么前提（地址稳定、无 CPU 同步、无动态控制流）、知道 3.11 µs 这个数里哪一半可以外推、哪一半不能。

### 1.2 符号与口径约定

| 符号 | 含义 | 本仓取值 |
|---|---|---|
| $T_{\text{host}}$ | 一次调用的主机侧成本 | Triton ~36 µs、torch ~8 µs（1024² 协议） |
| $T_{\text{dev}}$ | 一次调用的设备侧执行时间 | 随形状变，1024² fp32 softmax 约 3 µs 量级 |
| N | 一次 bench 里连续发射的调用数 | 100(EXP-T05) |
| $\pi_{\text{mem}}$ | HBM 带宽 | 1008 GB/s |
| roofline % | 实测带宽 / 1008 GB/s | 8192² softmax 91% |
| 三口径（int8） | 裸 / ext 端到端 / Triton 融合 | 5.9 / 65.1 / 51.7 µs |

硬件常数出处：NVIDIA, "NVIDIA Ada GPU Architecture" 白皮书，Appendix A Table 2 (GeForce RTX 4090:SMs 128、Memory Bandwidth 1008 GB/sec、**L2 Cache Size 73728 KB**、L1 Data Cache/Shared Memory 16384 KB)。L2 那一行在 §3.4.3 会被用来解释 3.11 µs 这个数的性质。

### 1.3 本篇引用的一级文献(详细出处见 §8.4)

- CUDA Graph 的三阶段模型与约束：NVIDIA CUDA C++ Programming Guide，"CUDA Graphs" 一章（定义/实例化/执行、stream capture 的限制）。
- CUDA Graph 的量化收益：NVIDIA Technical Blog, "Getting Started with CUDA Graphs"。
- PyTorch 侧的捕获契约：PyTorch 文档 "CUDA semantics" 的 CUDA Graphs 一节。
- 融合的收益结构：Ivanov, Dryden, Ben-Nun, Li, Hoefler, "Data Movement Is All You Need: A Case Study on Optimizing Transformers", arXiv:2007.00072。
- 性能模型：NVIDIA, "GPU Performance Background User's Guide" §4 Understanding Performance;Williams, Waterman & Patterson, "Roofline", CACM 52(4):65-76, DOI:10.1145/1498765.1498785。
- Triton 侧语义：Triton 官方文档与本机 3.6.0 源码（`triton.Config`、`tl.range`）。

## 2. 直觉与第一性原理

### 2.1 一次 kernel 调用到底花在哪

成本分两段——**主机侧**（Python/C++ 分发、参数处理、JIT 缓存查找、grid 计算、 wrapper 里的临时张量分配）与**设备侧**（kernel 本体执行）。在一个不做同步的循环里连续发射时，两段是**流水并行**的：主机在为第 $i+1$ 次调用做准备，设备还在跑第 $i$ 次。于是 $$T_{\text{每调用}} \approx \max（T_{\text{主机}}，\ T_{\text{设备}}）$$ 不是相加。这个 $\max$ 就是全篇的第一性原理：**你测到的数字，只反映两者中更大的那个。**

**这个 $\max$ 凭什么成立**，要能说出硬件与软件两层理由（本讲义推导）：

1. **软件层**：CUDA 的 kernel launch 是**异步**的——host 把工作提交到 stream 就返回， 不等设备执行完。所以在没有同步点的循环里，host 可以一直往前跑。
2. **硬件层**：同一 stream 上的 kernel 按提交顺序串行执行，但**提交本身**与执行在不同的处理器上（CPU vs GPU），两者天然并行。
3. **失效条件**（必须记住）：循环体里只要出现一次 `torch.cuda.synchronize()`、 `.item()`、`.cpu()` 或任何读回设备数据的操作，流水就被打断，$\max$ 退化成 $\sum$。 §4 第 6 段会指出这一点在计时函数里的具体形态。

### 2.2 没有这层区分会怎样

在 1024² 上测出 Triton 37.6 µs vs torch 8.1 µs，得出"Triton 慢 4.4 倍"的结论， 然后去优化 kernel——tile、向量化、规约树改一遍，数字纹丝不动。这正是本仓真实发生过的事（§3.2.4 的证伪案例），而它浪费的不是算力，是人的时间。

**日常类比与它的失效点**：像快递的"下单 + 分拣 + 派送"。类比在三处失效：①快递的三段是串行相加的，GPU 的主机段与设备段是流水并行的（所以是 $\max$ 不是 $\sum$）； ②"合并订单"在快递里省的是运费，在 GPU 上**同时**省两笔——少一趟 launch，以及少一份中间张量的 HBM 往返（§3.3 的融合就吃这两笔）；③快递没有"把整条派送路线录下来重放" 这种操作，而 CUDA Graph 恰恰是这个（§3.4）。

### 2.3 稳态、首次调用与"为什么 N=100 就够"

$\max$ 模型是**稳态**下的结论，而一个长度为 N 的循环并不是从稳态开始的。把它写细（本讲义推导）：

- 第 1 次调用：主机做完准备后提交，设备开始执行。这一刻设备是空的，所以这一次的时间是 $T_{\text{host}} + T_{\text{dev}}$（相加，不是取大）。
- 第 2 次起：主机在准备第 $i+1$ 次时，设备正在跑第 $i$ 次，进入稳态，每次 $\max(T_{\text{host}}， T_{\text{dev}})$。
- 最后一次：主机已经提交完，但要等设备排空，末尾补一个 $T_{\text{dev}}$。

于是 $$T_{\text{total}} \approx N\cdot\max(T_{\text{host}}, T_{\text{dev}})
+ \min(T_{\text{host}}， T_{\text{dev}})$$ 除以 N 得到的每调用值与稳态值的相对偏差是 $\min/(N\cdot\max) \le 1/N$。 **N=100 时这个偏差 ≤ 1%，小于本仓最不稳那一格的轮间 std(9.17%)**——所以 N=100 是够的，而 N=1 完全不够（那时测的是 $T_{\text{host}}+T_{\text{dev}}$，是另一个量）。

这条推导还顺手解释了 `wall()` 里 `wu=10` 次 warmup 为什么必要（§4 第 6 段）： 它不只是"让缓存热起来"，更是**把首次调用那笔 $T_{\text{host}}+T_{\text{dev}}$ 以及 JIT 编译、分配器首次分配全部挪到计时区间之外**。

### 2.4 三条贯穿全篇的公理

- **公理 A（先分口径，再改代码）**：任何优化动作之前，先确定时间在主机侧还是设备侧。这个判断只需要多测两个形状，成本比改一次 kernel 低一个数量级。
- **公理 B（猜测必须先设计对照）**："我觉得瓶颈是 X"要先能回答"如果不是 X，实验会长什么样"。否则改完看不出差别时，你连"是没效果"还是"改错了"都分不清（§3.2.4）。
- **公理 C（倍数与绝对值的外推性不同）**：一个比值里，主机侧那一半通常可以外推（与张量内容无关），设备侧那一半通常不能（依赖缓存命中）。11.6× 与 3.11 µs 是这条公理最好的例子（§3.4.3）。

## 3. 完整推导与机制

### 3.1 第一层:带宽主导尺寸下,设备侧同速

先把设备侧单独看清楚。8192×8192 fp32 的行 softmax，一次读一次写： $$2 \times 8192^2 \times 4\，\mathrm{B} = 536.9\ \mathrm{MB}$$ 存盘 raw(data/raw/EXP-T02/ew_gemm_bench.json)里 Triton 0.58497 ms、torch 0.58252 ms， 换算即 **0.92 TB/s 量级**；仓内现行口径记 **917 / 921 GB/s**（EXP-T03《三件套移植 + torch 绑定》§5），对 4090 的 1008 GB/s roofline 是 **91%**，与 kperf 卡片"带宽 91%、occ 67%（regs 限）"一致（终端级证据，登记于 EXP-T06《FP8 GEMM》§7）。

3 轮口径同样贴合：0.584433±0.000843 ms(Triton)与 0.582353±0.000179 ms(torch)， 折算 918.7 / 921.9 GB/s（data/derived/exp-t02_stability_3rounds.csv，本讲义按同一字节数折算）。**两个口径给同一结论，轮间 std 都在 0.15% 以内——设备侧的数字是稳的。** 记住这句话，§5.1 会用它的反面（主机侧数字很飘）来做一次归因。

**为什么"两边都贴 roofline"就能推出"kernel 没差距"**：两边都被同一堵 HBM 带宽墙卡住， 任何写法差异（向量化、bank conflict、规约树形状）都只能在墙内做文章，不可能反映到墙外的总时长上。反过来说，**这个结论只在带宽主导尺寸成立**——它不是"Triton 和 CUDA 一样快"的普遍证明，而是"在这个尺寸上差距不可测"的诚实陈述。

**那 9% 的缺口在哪**（本讲义推导，给个方向而非结论）：行 softmax 每行要做两次规约（max 与 sum）再做一次逐元素写，规约期间不产生访存，所以带宽利用必然低于 100%； kperf 记的 occupancy 67% 且注明"regs 限"说明每 SM 能驻留的 warp 不够多到把规约段完全遮住。**本仓没有做 occupancy 扫描来验证这条**，标为推断。

### 3.2 第二层:三点法把时间从 kernel 里赶出来

#### 3.2.1 三点法的构造

同一个 softmax kernel，只换形状（EXP-T03）：

| 点 | 设备工作量 | Triton | 对照 | 这个点证明什么 |
|---|---|---|---|---|
| 8×8 | ≈ 0 | 37.4 µs | torch 8.0 | 纯主机侧开销的**读数** |
| 1024² | 小 | 37.6 µs | torch 8.1 / CUDA v4 7.8 | 37.6 ≈ 37.4 → 时间**不在 kernel 里** |
| 8192² | 大（带宽主导） | 917 GB/s | torch 921 GB/s | 设备侧**同速**(§3.1) |

推理链一步一步：①8×8 的设备工作量近似 0，所以那 37.4 µs **只能**是主机侧； ②1024² 的 37.6 µs 与 37.4 µs 相差不到 1%，说明这个尺寸上设备侧仍被主机侧盖住； ③8192² 让设备侧显形，两边打平。三个点连起来，结论是**唯一的**：小尺寸的 4.4× 差距是一个**与形状无关的主机侧常数**，不是 kernel 的差距。

**这个论证的逻辑形式值得单独记住**（本讲义推导）：它是一个**极限夹逼**。点 1 把 $T_{\text{dev}}\to 0$，于是测到的就是 $T_{\text{host}}$；点 3 把 $T_{\text{dev}}$ 拉到远大于 $T_{\text{host}}$，于是测到的就是 $T_{\text{dev}}$； 点 2 是待解释的那个点，它落在哪一侧由前两个点的读数直接判定。 **只用了 §2.1 那一个 $\max$ 模型，没有引入任何额外假设**——这是三点法比 "profile 一下看看"更可靠的地方：后者依赖工具的归因是否正确，前者只依赖一个能写在纸上的模型。

于是有了三口径的分发成本：**Triton 的 Python 分发 ~30 µs > torch 的 C++ 分发 ~8 µs > 裸 CUDA ~5 µs**。Triton 贵在每次调用都要过 Python 包装（参数处理、JIT 缓存查找、 grid 计算）以及 wrapper 里的临时分配。

**数字分层要说清楚**（诚实度要求）：三点法用的是排障会话的**终端级证据**（EXP-T03 §5 登记）；同尺寸的存盘 raw 值是 36.83 / 7.84 µs，3 轮 stability 是 36.33±0.44 / 8.24±0.36 µs(data/derived/exp-t02_stability_3rounds.csv)。三组数字同量级，**结论不依赖小数点**——但引用时要说清是哪一组。

#### 3.2.2 交叉点在哪:一个可以先算再测的量(本讲义推导)

既然 $T_{\text{每调用}} = \max(T_{\text{host}}， T_{\text{dev}})$，那么"该不该上 Graph" 就等价于问"我的 $T_{\text{dev}}$ 有没有超过那条地板"。对带宽主导的行核： $$T_{\text{dev}} \approx \frac{\text{读写字节}}{\pi_{\text{mem}}}$$ 令它等于 Triton 的地板 $36.2\，\mu s$： $$\text{读写字节} = 36.2\times10^{-6}\times1008\times10^{9} \approx 36.5\ \mathrm{MB}$$ 对 fp32 一读一写的方阵，$8n^2 = 36.5\，\mathrm{MB} \Rightarrow n \approx 2136$。

**结论**：本仓这类行核在约 $2048^2$ fp32 以下，时间由主机侧地板决定；以上才由设备侧决定。对照实测：$1024^2$(8.39 MB)确实是主机侧主导，$8192^2$(536.9 MB)确实是设备侧主导，而 $2048^2$ 这个交叉点附近**本仓没有测过**。这条预测是可证伪的，验证成本是往 scripts/test_ew_gemm.py 的形状列表里加一行（§8.3 列为待办）。

**这个算式的用法比数值更重要**：换一张卡、换一个 Triton 版本，地板会变，交叉点跟着变； 但"先算交叉点、再决定优化方向"这个动作不变。

#### 3.2.3 藏在数据里的第四个点

三点法是当年为排障设计的。回头看 3 轮 stability 表，里面还有一个**当时没被当成证据、事后独立支持同一结论**的点：

| 形状 | 元素数比 | Triton（3 轮 µs） | torch（3 轮 µs） |
|---|---|---|---|
| 1024×1024 | 1.00 | 36.33±0.44 | 8.24±0.36 |
| 1024×1500 | **1.46** | 36.11±0.35 | 8.45±0.30 |

**数据量多了 46%，时间一点没变**（Triton 侧差值 −0.22 µs，合并 σ = 0.56，即 0.4σ， 不显著）。按 §3.2.2 的算式，1024×1500 fp32 一读一写是 12.29 MB，设备侧下限 $12.2\，\mu s$，仍远低于 36 µs 的地板——**所以时间不变正是模型的预言**。

这个点还顺手证伪了另一件事：1024×1024 走的是 `EXACT` 无 mask 快路径，1024×1500 走的是带 mask 的慢路径（§4 第 1 段），**两条路径的时间在 3 轮口径下不可区分**。这就是 §3.2.4 那个假设的独立复核。

#### 3.2.4 一个被证伪的假设,完整复盘

本仓最有教学价值的一次失败：

- **跑前假设**：小尺寸慢是因为掩码 load 阻断了 128-bit 向量化访存。
- **动作**：给 softmax 与 int8 quantize 各加一条"整除时走无 mask 快路径"的分支（`EXACT`，src/elementwise_kernels.py：63-69 与：100-102）。
- **实测**：数字**纹丝不动**。假设作废。
- **事后复核**（本讲义补）：3 轮 stability 给出 0.4σ 的不显著差（§3.2.3）， 把当年的"纹丝不动"从定性升成定量。
- **处理**：快路径语义无害，**代码留在仓里**作为"猜测必须交给对照实验"的物证； 记录里保留全过程（EXP-T03 §7）。随后改测三点法，才坐实了主机侧假设。

方法论提炼：**"我觉得瓶颈是 X"必须先设计出一个"如果不是 X 会怎样"的对照，再动手改代码**；否则改完看不出差别时，你连"是没效果还是改错了"都分不清。

**为什么这个假设当时听起来很合理，值得说一句**：mask load 确实会影响向量化——这在访存受限的大尺寸上是真问题。假设错的不是机制，是**适用区间**：它被用在了一个设备侧根本不显形的尺寸上。**很多"合理但无效"的优化，错在区间而不是错在机制。**

### 3.3 第三层:端到端反转,融合数比单核快慢更重要

#### 3.3.1 两条路径的对照

第三层的对象换成 int8 per-channel quantize，因为它有**两种实现路径**可比：

- **ext-CUDA 路径**：复用 Kernel_Optimazation 的 `quantize_v4` CUDA kernel（零改动）， 用 torch extension 绑进来。但 v4 的签名要求 **scale 预置**，所以 scale 只能在绑定层用 torch 算——`(std::get<0>(x.abs().max(1)) / 127.0f).clamp_min(1e-8f)` 这**一行** 展开就是 abs → max → div 三次独立 kernel launch，加上 v4 本身共 4 次（src/torch_ext/int8_binding.cpp：26-32）。端到端 **65.1 µs**。
- **Triton 融合路径**：absmax 规约、缩放、舍入、写回 scale **一趟做完**， **1 次 launch**(src/elementwise_kernels.py：91-113)。端到端 **51.7 µs**。
- **裸 kernel 口径**：同一个 v4 kernel 单独 bench 只有 **5.9 µs**（EXP-K01《四 kernel 4090 重基准》，口径 = scale 预置，不含 torch 封装）。

**"更快的 kernel 输掉端到端"**：v4 本体比 Triton 版快一个量级，端到端却输 13.4 µs。账要这么算（不能简单地"3 × 8 µs = 24 µs"）：Triton 路径 ≈ 1 次贵分发 + 1 次设备本体； ext 路径 ≈ 4 次便宜分发 + 4 次设备本体（其中三个是极小的规约 kernel，设备时间可忽略但每个都要付一次分发）+ pybind 与张量校验。两条路各有各的贵法，而融合把**次数**这一项压到 1。所以本仓的结论句是：**在 torch 集成层，融合数（launch 数）比单 kernel 的快慢更重要。**

**三口径约定**（不得混引）：5.9 µs（裸，scale 预置）/ 65.1 µs（ext 端到端）/ 51.7 µs（Triton 融合）。第三个数跨会话在 **41.6~52 µs** 之间波动（主机侧开销对系统状态敏感），引用时带区间；同尺寸的 Triton 融合有 3 轮锚 42.2±0.4 µs (int8q_1024x1024，data/derived/exp-t02_stability_3rounds.csv)，但**与 ext 头对头的那一对（51.7 / 65.1）仍是单轮**，对外引用必须注明。

#### 3.3.2 一个 3 轮口径的融合证据(本讲义补出)

上面那对头对头数字是单轮，这是它最大的弱点。但同一批 3 轮数据里其实有一组**同类对照且轮数达标**的：同一个 1024² int8 quantize，Triton 融合版 vs **纯 torch eager 版**（scripts/test_ew_gemm.py：72-77 的 `eager_quant`，展开同样是 amax → div → clamp → round → clamp → to 这一串独立算子）：

| 路径 | launch 数（量级） | 3 轮 mean±std(µs) |
|---|---|---|
| Triton 融合 | 1 | **42.16±0.35** |
| pytorch_eager | 多次 | **99.70±1.74** |

（data/derived/exp-t02_stability_3rounds.csv 的 `bench.int8q_1024x1024.*`。）

**这一对是 3 轮的、同进程的、同输入的**，它独立地支持 §3.3.1 的结论：在这个尺寸上， **决定端到端时间的是算子个数，不是单个算子的质量**。它不能替代 ext-CUDA 那一对（对照物不同：一个是 torch eager，一个是绑定的手写 CUDA），但它把"融合有用"这个结论从单轮抬到了 3 轮。**引用时两对要分清：前者证明"融合 > eager 链"，后者证明 "融合 > 更快的单核 + 多次 launch"，后一句才是反直觉的那句。**

#### 3.3.3 融合省的两笔账,分别有多大(本讲义推导)

融合同时省两件事，必须分开算：

1. **省 launch**：少 3 次 torch 分发，按 §3.2.1 的 ~8 µs/次，约 24 µs。
2. **省中间量往返**：eager 链里 `x.abs()` 要写一份 1024² fp32 = 4.19 MB 的中间张量再读回来，`x / scale` 又是一份。按 1008 GB/s，每份中间量的写+读是 $2\times4.19\，\mathrm{MB}/1008\，\mathrm{GB/s} = 8.3\，\mu s$。

两笔加起来的量级（$24 + \sim16$）与实测差（$99.70 - 42.16 = 57.5\，\mu s$）同阶， 但**对不上小数点**——本仓没有逐算子拆分 eager 路径的 launch 数与中间量数， 所以这只是一个量级核对，不是归因。诚实的表述是：**两笔都在起作用，配比未隔离。**

这个结构与 Ivanov 等人的 "Data Movement Is All You Need"(arXiv:2007.00072)诊断的是同一件事。他们对 BERT 训练做算子分类后写道："While tensor contractions account for over 99% of the arithmetic operations performed, they constitute only 61% of the runtime. Over a third (37%) of the runtime in a BERT training iteration is spent in memory-bound operators."并给出方向："fusion is a major opportunity for promoting data reuse, as when operators cover identical iteration spaces, global memory writes and subsequent reads between them can be removed."他们据此把 BERT encoder layer 提速 1.30×、整个 BERT 提速 1.19×，数据移动减少最多 22.91%。

**注意口径差别**：他们省的主要是第 2 笔（中间量），因为训练的算子本身够大、launch 不是瓶颈；本仓在 1024² 这个尺寸上第 1 笔（launch）占比更大。**同一个"融合"动作， 在不同尺寸上兑现的是不同的那笔账**——这是把论文结论搬到自己场景时最容易错的一步。

#### 3.3.4 三条路径并排:把 launch 数与字节数一起摆出来

把 §3.3.1 的三条路径按"launch 数"与"HBM 字节数"两列并排，才看得清各自贵在哪（1024×1024 fp32，本讲义按协议推算字节数，时间为实测）：

| 路径 | launch 数 | 输入读 | 中间量往返 | 输出写 | 实测端到端 |
|---|---|---|---|---|---|
| 裸 CUDA v4（scale 预置） | 1 | 4.19 MB | 0 | 1.05 MB(int8) | **5.9 µs**（EXP-K01，不含封装） |
| ext 绑定端到端 | 4 | 4.19 MB ×2（scale 那趟再读一遍） | abs 的中间张量 4.19 MB 写 + 读 | 1.05 MB | **65.1 µs**（单轮） |
| Triton 融合 | 1 | 4.19 MB | 0 | 1.05 MB + scale | **51.7 µs**（单轮，跨会话 41.6~52） |

**三行的字节数差不到 3 倍，时间差 11 倍**——这就是"这个尺寸上 launch 主导"的最直接证据。把字节折算成时间：ext 路径比裸 kernel 多搬约 12.6 MB，在 1008 GB/s 上是 **12.5 µs**；而实测差 59.2 µs。**剩下的 46.7 µs 只能是分发**，与"4 次 torch 分发
+ pybind + 张量校验"这个量级对得上（§3.2.1 估 torch 单次 ~8 µs）。

**这张表也说明了裸 5.9 µs 为什么不能和另两行相减**：它的口径里没有 scale 计算这件事本身，不是"同一件事的更快实现"，是"少做了一步"。**口径不同的数字相减，得到的是一个没有物理意义的量。**

### 3.4 第四层:CUDA Graph 把 launch 归零

#### 3.4.1 机制:官方文档怎么定义

CUDA C++ Programming Guide 的 "CUDA Graphs" 一章把工作提交拆成三个阶段： **定义**("a program creates a description of the operations in the graph along with the dependencies between them")、**实例化**("takes a snapshot of the graph template, validates it, and performs much of the setup and initialization of work")、**执行**("An executable graph may be launched into a stream ... It may be launched any number of times")。收益写得很直白： "for a GPU kernel with a short execution time, this overhead cost can be a significant fraction of the overall end-to-end execution time. By creating a CUDA graph that encompasses a workflow that will be launched many times, these overhead costs can be paid once for the entire graph during instantiation."

也就是说，§3.2 里那个"与形状无关的主机侧常数"被**摊到一次提交里**。

NVIDIA 的 "Getting Started with CUDA Graphs" 博客给了一组量化对照（20 个短 kernel × 1000 步）：朴素逐个 launch 是 "9.6μs per kernel (including overheads): much higher that the kernel execution time of 2.9μs"；让 launch 重叠后降到 "3.8μs (vs 2.9μs kernel execution time)"；上 graph 后 "3.4μs"。同时给出建图成本： "the time to create and instantiate the graph is relatively large at around 400μs, but this is only performed a single time, so this is only contributes around 0.02μs to our per-kernel cost."

**这组官方数字与本仓数据的关系要说清**：他们的 kernel 本体是 2.9 µs，launch 开销是 6.7 µs(9.6 − 2.9)，graph 后剩 0.5 µs；本仓的 kernel 本体约 3 µs，Triton 的 Python 分发是 33 µs，graph 后剩 3.11 µs。**机制同一个，但本仓的主机侧成本比 C++ 的例子大一个数量级**——这正是"Triton 小核慢"的全部内容，也是为什么本仓的塌缩倍数（11.6×） 比官方例子（2.8×）大得多。**倍数越大，不代表优化越好，只代表原来的坑越深。**

#### 3.4.2 数字

（EXP-T05《CUDA Graph 消 launch 开销实测》，1024² fp32 softmax × 100 调用，3 轮，每次调用 µs）：

| 路径 | eager | + CUDA Graph | 塌缩 |
|---|---|---|---|
| Triton softmax | 36.2±0.1 | **3.11±0.00** | **11.6×**，消掉约 33 µs/调用 |
| torch.softmax | 8.03±0.74 | 4.04±0.01 | 2.0× |

两个推论：①**graph 后 Triton(3.11)反超 torch eager(8.03)与 torch+graph(4.04)**——所以"Triton 小核慢"的正解是**上 Graph，而不是换 CUDA**；②torch+graph 的 4.04 仍慢于 Triton+graph 的 3.11，说明把主机侧因素扣掉之后，**Triton 的 kernel 本体在这个形状上本来就更快**——这与 §3.1 的"设备侧同速"不矛盾：那是带宽主导的 8192²，这是被 L2 装得下的 1024²，两个尺寸问的不是同一个问题。

#### 3.4.3 这个数字的口径上限(本讲义主动指出)

1024² fp32 一读一写 = 8.39 MB，除以 3.11 µs 得 **2.7 TB/s**，远超 HBM 的 1008 GB/s。唯一的解释是这份 4.19 MB 输入在 100 次重放里**常驻 L2**——RTX 4090 的 L2 是 73728 KB（Ada 白皮书 Table 2），4.19 MB 只占 5.7%，**装下绰绰有余**。

把这条量化到可核对的程度（本讲义推导）：若假设整个读写都由 L2 服务，则所需 L2 带宽是 2.7 TB/s。**本仓没有测过 4090 的 L2 实际带宽**，所以"2.7 TB/s 在 L2 能力之内"这句标为未核实；能确定的是它超过 HBM 峰值 2.7 倍，因此**必然主要来自片上**。

所以 3.11 µs 是"输入 L2 常驻 + 同一张量重复"的**下限口径**；真实推理里每步的激活都不同，这个数会变大。**11.6× 的塌缩倍数是主机侧结论，可以外推；3.11 µs 这个绝对值是设备侧结论，不要外推。**（推断，本仓未做"每次换输入"的变体。）

顺带把 torch 那一行也读一遍：torch+graph 4.04 µs 对应 2.08 TB/s，同样只能是 L2； 两条 graph 线的差 0.93 µs 就是两个 kernel 本体 + 节点调度的差。**两条线都被同一个 L2 红利抬着，所以它们之间的比较仍然公平**——这是"对照方也上 graph"这个设计的价值。

#### 3.4.4 捕获的前提:PyTorch 文档写死的四条

PyTorch 文档（"CUDA semantics" 的 CUDA Graphs 一节）把契约写得很清楚： "A CUDA graph is a record of the work (mostly kernels and their arguments) that a CUDA stream and its dependent streams perform"；收益是 "Replaying a graph sacrifices the dynamic flexibility of typical eager execution in exchange for greatly reduced CPU overhead"。约束四条：

1. **地址稳定**："Every replay reads from and writes to the same (virtual) memory addresses"——重放只能改内容不能改地址；
2. **非默认流捕获**："Capture must occur on a non-default stream";
3. **不能有 CPU 同步**：`.item()` 这类把 GPU 结果读回 CPU 的操作会让捕获失败；
4. **先预热**："Before capture, warm up the workload to be captured by running a few eager iterations. Warmup must occur on a side stream."

CUDA 侧还有一条对应的说明：捕获期间 "it is invalid to synchronize or query the execution status of a stream which is being captured or a captured event"（CUDA C++ Programming Guide,stream capture 一节）；以及实例化之后拓扑不可改—— "Major changes to graph structure (such as topology or node types) require re-instantiation"。

**这四条直接决定了工程形态**：动态 shape 需要**分桶捕获**。vLLM 的 `cudagraph_capture_sizes` 就是这件事：按一组预设 batch size 各捕一张图，运行时选 **不小于当前 batch 的最小那张**并用 dummy 请求 padding；落在所有桶之外就退回 eager（vLLM 官方文档 CUDA Graphs 设计页）。本仓把整个引擎 step 捕获下来的尝试没有做， 因为 KV cache 的动态增长需要先静态化（EXP-T05 §7）。

**代价也要说清**：桶越多，捕获时间与显存占用越大（每张图都要持有自己的静态张量）。所以"上 Graph"不是无条件的胜利，它把一部分**运行时灵活性**换成了**吞吐**——这正是 PyTorch 文档那句 "sacrifices the dynamic flexibility ... in exchange for greatly reduced CPU overhead" 的工程含义。

#### 3.4.5 建图成本与摊销点:一个可以先算的判据

Graph 不是免费的：定义 + 实例化要花时间，而且**每个桶各花一次**。官方博客给的量级是 "the time to create and instantiate the graph is relatively large at around 400μs"（20 个节点的图）。摊销点的算式很简单（本讲义推导）：

$$\text{需要重放的次数} = \frac{\text{建图成本}}{\text{每次重放省下的量}}$$

分母是"图里的 launch 数 × 每次 launch 省下的主机侧成本"。代进本仓的场景： 一张 100 节点的图，每节点省 33 µs，一次重放省 3.3 ms；即使建图要 400 µs 量级， **一次重放就回本**。代进一个推理引擎的 decode step（几十到上百个 kernel、 C++ 分发每次几微秒），同样是一两次重放就回本。

**所以建图成本在"重放很多次"的场景里可以忽略，真正的代价是别的两样**： ①**启动延迟** = 桶数 × 建图成本 + 每桶的预热前向；②**显存** = 每张图持有自己的静态输入输出张量。桶越多，这两样越贵——这就是 vLLM 要精心挑 `cudagraph_capture_sizes` 而不是"每个 batch size 都捕一张"的原因。

**本仓没有测过建图成本**（`wall()` 只计时 `g.replay()`，捕获在计时区间之外）， 所以上面的 400 µs 是引用官方博客的量级，本机数值**未核实**；这条口径遗漏在 §8.1 第 2 条与 §8.3 待办里都登记了。

### 3.5 四层合起来:一棵可以当场画的决策树

四层因果链一句话串起来：**设备侧同速 → 差距全在主机侧分发 → 端到端由融合数决定 → 主机侧这一层的终局解是 CUDA Graph。** 由此得到选型决策树：

```
kernel 慢?先分口径,别先改代码
├─ 设备侧计时(或大尺寸点)是否与对照同速?
│  ├─ 否 → 差距真在 kernel:调 tile / num_stages / 寄存器 ILP(讲义 02)
│  └─ 是 → 差距在主机侧,继续往下
└─ 端到端路径上有几次相邻 launch?
   ├─ 多次(有中间张量) → 先做融合:一个 kernel 干完,省 launch 也省中间量往返
   │                      (本仓 int8:4 次 launch 65.1 µs → 1 次 51.7 µs)
   └─ 单次但高频     → 上 CUDA Graph:把分发摊成一次提交
                        (本仓 36.2 → 3.11 µs/调用,11.6×)
      └─ 地址不稳 / 动态 shape? → 分桶捕获;桶太多就退回"写 CUDA/C++ 扩展"
```

**加一条量化的入口判据**（§3.2.2 的算式）：在动手之前先估 $T_{\text{dev}} \approx \text{字节}/\pi_{\text{mem}}$，与你这条栈的主机侧地板比大小。本仓这台机器上 Triton 的地板是 36.2 µs，对应约 36.5 MB 读写量；**低于这个量级的算子， 不上 Graph 或不融合，再怎么调 kernel 都是白费**。

补一条选型直觉：**大 kernel、长序列、融合机会多 → Triton 白给**（FA2 达 SDPA-flash 的 87%、GEMM 打平 cuBLAS，讲义 01/02）；**微 kernel 高频调用 → 裸 CUDA 或 Graph**； **torch 集成层 → 先数 launch 次数，再谈 kernel 快慢**。

### 3.6 硬件与运行时语义层:这些结论被什么定死

#### 3.6.1 "主机侧地板"由什么构成,以及它为什么对系统状态敏感

Triton 每次调用要过的路径（§4 第 2 段是它的代码现场）：Python 层的形状规整（`reshape` / `contiguous`）、输出张量分配（走 caching allocator）、编译期常量推导（`next_power_of_2`、`EXACT`、`num_warps`）、JIT 缓存查找（按签名哈希）、grid 计算， 最后才是驱动的 launch 调用。

**这些步骤有一个共同点：它们跑在 CPU 上，受操作系统调度、Python GIL、缓存命中影响。** 所以主机侧数字的轮间波动天然比设备侧大——本仓的 3 轮 std 直接印证了这一点：

| 量 | 3 轮 std / 相对 | 性质 |
|---|---|---|
| graph 重放（Triton） | 0.00 µs / 0.00% | 设备侧 + 一次提交 |
| eager(Triton) | 0.11 µs / 0.30% | 主机侧，但 Python 路径固定 |
| **torch eager** | **0.737 µs / 9.17%** | 主机侧，C++ dispatcher 受系统状态影响最大 |
| softmax 8192²（设备侧） | 0.000843 ms / 0.14% | 纯设备侧 |

**"设备侧数字稳，主机侧数字飘"是这一整篇的经验规律**，它也解释了 §3.3.1 里那个 41.6~52 µs 的跨会话区间——不是测错了，是那个量本来就飘。

#### 3.6.2 kernel launch 的异步语义:$\max$ 模型的软件基础

CUDA 的 kernel launch 是异步的：host 提交后立即返回。这条语义是 §2.1 那个 $\max$ 模型成立的前提，也是一切"用 wall clock 除以 N"的计时法能工作的前提。

**推论一（计时协议）**：必须在计时区间的**两端**各做一次 `synchronize()`，中间**不做**——scripts/test_cudagraph.py：14-18 的 `wall()` 正是这个形状。中间加同步会把 $\max$ 破坏成 $\sum$，测出的数字"大得很整齐，也很假"。

**推论二（为什么要跑 N=100 次）**：单次调用的 wall clock 里混着计时器自身的开销与首次调用的冷启动；连续发 100 次再除，既摊薄了这些，又让主机与设备的流水进入稳态。 **但这也带来一个副作用**：100 次用同一个输入张量，于是有了 §3.4.3 那个 L2 红利。 **实验设计的每一个选择都同时买到一个好处和一个偏差，能说出偏差在哪才算设计完。**

#### 3.6.3 Graph 节点与 profiling 的坑

graph 内的 kernel 在时间线上默认聚成一个节点，要看内部结构需要 node 级 trace（nsys 的 `cuda-graph-trace=node`）。这条在本仓是**跨仓踩过的教训**（docs/theory/03 §5 的锚）。它与本篇结论互为表里：**Graph 把主机侧成本消掉的同时，也把"每个 kernel 花了多久"这个信息在默认视图里藏了起来**——优化手段与观测手段是耦合的。

#### 3.6.4 L2 的容量层级:为什么 1024² 和 8192² 问的不是同一个问题

| 数据集 | 大小 | 对 L2(73728 KB)| 落在哪一层 |
|---|---|---|---|
| 1024² fp32 输入 | 4.19 MB | 5.7% | **L2 常驻** |
| 1024×1500 fp32 输入 | 6.14 MB | 8.3% | L2 常驻 |
| 8192² fp32 输入 | 268.4 MB | 356% | **必走 HBM** |

**这张表是"两个尺寸问的不是同一个问题"的准确表述**：1024² 那一档的设备时间反映的是 L2 带宽，8192² 那一档反映的是 HBM 带宽。把两档的"设备侧结论"混在一起讲，就会得出 "Triton 有时快有时慢"这种没有信息量的结论。

### 3.7 每个魔法数的来历(理论上界 / 硬件约束 / 实测扫描)

| 参数 | 值 | 类别 | 依据 |
|---|---|---|---|
| `BLOCK`（行核） | `next_power_of_2(n_cols)` | **语言约束** | `tl.arange` 要求编译期 2 的幂（src/elementwise_kernels.py：50、84、121） |
| `num_warps`（行核） | 宽行 8 / 窄行 4，阈值 2048 | **未扫参** | 启发式；src/elementwise_kernels.py：51 的注释给了理由，本仓未做扫描 |
| `EXACT` 快路径 | 整除时启用 | **已证伪的假设的遗留** | §3.2.4；语义无害故保留为物证 |
| int8 clamp 上下界 | ±127（弃 −128） | 理论 | 对称量化要求 $q$ 与 $-q$ 都可表示 |
| `1e-8` 防除零 | 1e-8 | 工程经验 | 全零行的除零保护；量级未扫 |
| N(graph bench) | 100 | 实验设计 | 摊薄单次计时噪声；副作用见 §3.6.2 |
| 预热次数 | 3 | **协议要求** | PyTorch 文档要求捕获前在 side stream 上预热 |
| `it=50, wu=10`(wall) | 50 / 10 | 实验设计 | scripts/test_cudagraph.py：14；未扫，足够让 std 收敛到 0.00-0.11 µs |

**这张表里"未扫参"占两条**，说出来不丢人；**说不出属于哪一类才丢人**。特别是 `num_warps` 那条启发式：它写在注释里、有理由、但没有对照实验，所以它是"有依据的猜测"而不是"实测结论"。

## 4. 代码逐段走读:行核、绑定层与 graph 捕获

按执行顺序走读（引用为仓内真实代码逐字拷贝，标 文件：起-止行）。

**第 1 段 · softmax kernel：减 max 与那条被证伪的快路径**(src/elementwise_kernels.py：58-78)

```python
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
```

角色：讲义 01 §3.1 的非分块版本，同时也是 §3.2.4 那次证伪的**物证**。`EXACT` 分支是当年为验证"mask load 阻断向量化"假设加的，实测数字纹丝不动，假设作废，但因为语义无害而留在仓里。两个数值细节：①慢路径的越界填充是 `-inf` 而不是 0（`exp(-inf)=0` 才不进分母， 补 0 会污染归一化）；②快路径里 `mask = offs < BLOCK` 恒真，存在的唯一理由是与慢路径共用同一个 store 签名。改错会怎样：把 `other` 从 `-inf` 改成 0，非整除列宽的行会多出若干份 $e^{0-\max}$ 的假质量，softmax 的每一项都偏小——而 1024×1024 这种整除形状**测不出来**（它走的是快路径），只有 1024×1500 那一格会炸。

**`EXACT` 是 `tl.constexpr`，这一点决定了它的性能语义**（本讲义推导）：两条分支在 **编译期**分叉成两份 kernel，运行期没有任何判断开销；代价是 JIT 缓存里多一个条目。如果它是运行期参数，每个 program 都要判一次，而且 warp 内如果发散还要串行两条路径。 **"用 constexpr 把控制流搬到编译期"是 Triton 里最常用的一招**，讲义 02 §3.7.2 的 `IEEE_DOT` 是同一招的另一处应用。

**第 2 段 · launcher：每次调用的主机侧工作量就在这几行**(src/elementwise_kernels.py：81-88)

```python
def softmax(x: torch.Tensor):
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    y = torch.empty_like(x2)
    BLOCK = triton.next_power_of_2(x.shape[-1])
    _softmax_kernel[(x2.shape[0],)](x2, y, x.shape[-1], x2.stride(0),
                                    BLOCK=BLOCK, EXACT=(BLOCK == x.shape[-1]),
                                    num_warps=8 if BLOCK >= 2048 else 4)
    return y.reshape(x.shape)
```

角色：§3.2 那 ~30 µs 的**来源现场**。这 8 行每次调用都要执行：reshape 与 `contiguous`、 `empty_like` 分配、`next_power_of_2` 计算、`EXACT` 与 `num_warps` 的推导，然后才进 Triton 的 JIT 缓存查找与 grid 计算。对 8192² 它们完全可以忽略，对 1024² 它们就是全部。 `num_warps` 的启发式（宽行 8、窄行 4）也在这里：宽行要更多 warp 才能把行内 load 与规约流水打满，窄行给 8 warps 反而把行切碎。改错会怎样：把 `BLOCK` 写成 `n_cols` 而不 `next_power_of_2`，`tl.arange` 编译期直接报错——这是 Triton 把"编译期约束"暴露在 Python 侧的典型形态，也是每次调用都要做一次的那类计算。

**逐行给这几行的成本定性**（本讲义推导，量级判断，未逐行计时）：

| 行 | 做了什么 | 每次调用都做？ | 成本性质 |
|---|---|---|---|
| `reshape(...).contiguous()` | 视图变换；若已连续则零拷贝 | 是 | Python 调用 + 可能的拷贝 |
| `torch.empty_like` | 走 caching allocator | 是 | 分配器查找，通常命中缓存 |
| `next_power_of_2` | 纯 Python 整数运算 | 是 | 微秒以下 |
| `EXACT=` / `num_warps=` | 两个 Python 比较 | 是 | 微秒以下 |
| `_softmax_kernel[grid](...)` | 签名哈希 + JIT 缓存查找 + 参数打包 + 驱动 launch | 是 | **主要成本** |

**本仓没有对这几行做逐行 profiling**，所以"主要成本在最后一行"是按经验判断，标为推断。 **能确定的是这一整段的总量**：8×8 那个点测到的 37.4 µs 就是它（§3.2.1）。

**第 3 段 · RMSNorm kernel：融合的另一个样本**(src/elementwise_kernels.py：30-43)

```python
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
```

角色：与第 1 段同构的行并行模式，但它更适合说"融合省的第二笔账"。torch eager 版的 RMSNorm 是 `x.float()` → `pow(2)` → `mean(-1)` → `+eps` → `rsqrt` → `*x` → `*w` → `.half()` 这一串独立算子（scripts/test_ew_gemm.py：41-42、46-47 的参考实现）， 每一步都要写一份与输入同形的中间张量再读回来；Triton 版一次读入、行内规约、一次写出。

3 轮口径（data/derived/exp-t02_stability_3rounds.csv）:

| 形状 | Triton(ms) | pytorch_eager(ms) |
|---|---|---|
| 2048×1024 | 0.03536±0.00050 | 0.10530±0.00085 |
| 2048×4096 | 0.03533±0.00152 | 0.23077±0.00008 |

**注意 Triton 那一列的两格几乎相同（0.0354 / 0.0353），而数据量差 4 倍**——按 §3.2.2 的判据，这两格都还在主机侧地板附近（35 µs 量级 ≈ 36 µs 的地板）。而 eager 那一列随数据量涨了 2.2 倍，说明**eager 路径的设备侧成本已经显形**：它的中间量往返把设备时间抬到了地板之上。**同一张表同时展示了两件事：融合有效，以及本仓 Triton 侧的这两个点仍然是主机侧主导。**（后一句是本讲义按 §3.2.2 的模型推出的读法。）

三个数值细节值得记：①规约统一升 fp32（fp16 平方和几千元素就丢位）；②`other=0.0` 使越界列对平方和零贡献，而分母用真实 `n_cols` 不是 `BLOCK`——**填充与归一化分母是两件事，写在一起最容易错**；③`eps` 放在 `sqrt` 内，全零行也不会除零。改错会怎样：分母写成 `BLOCK`，非 2 的幂列宽（如 1500）的结果会系统性偏小，而 1024/4096 这类整除形状测不出来——**又一个"只在非整除形状上显形"的错误**。

**第 4 段 · int8 融合 kernel：一趟做完就是全部秘密**(src/elementwise_kernels.py：91-113)

```python
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
```

角色：§3.3 反转的我方。它做的事和 ext 路径**完全一样**——absmax 规约、算 scale、缩放、舍入、截断、写回 q 与 scale——区别只在于这些步骤在**同一个 kernel 内**完成，因此 `x` 只从 HBM 读一次，scale 从不落地。三个细节：①`rint` 是就近舍入，不是截断， 截断会引入半个 LSB 的系统性偏差；②`clamp` 到 ±127 而**弃用 -128**，是对称量化的要求（要保证 $q$ 与 $-q$ 都可表示）；③`tl.maximum(scale, 1e-8)` 防全零行除 0。改错会怎样：把 scale 计算挪回 torch 侧（哪怕 kernel 一行不改），端到端立刻退化成 ext 路径那种多 launch 形态——**这就是第 5 段要展示的反面教材**。

**注意 `x` 在这个 kernel 里被用了两次**（算 absmax、算 q），但只从 HBM 读一次——它在寄存器里。这就是融合省的第二笔账在最小尺度上的样子： **"同一份数据被多次使用"这件事，融合把它从 HBM 往返降级成寄存器复用。** eager 路径做不到，因为每个 torch 算子的边界就是一次 HBM 往返。

**第 5 段 · 绑定层：一行 torch 表达式 = 三次 launch**(src/torch_ext/int8_binding.cpp：17-33)

```cpp
void quantize_v4(const float* input, const float* scales, int8_t* output,
                 int channels, int hw);

std::vector<torch::Tensor> int8_quantize_v4(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kFloat32,
                "expect fp32 CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "expect (channels, hw)");
    // kernel 以 (channels, hw) 扁平指针 + 行主序寻址,必须保证内存连续
    auto x = input.contiguous();
    // per-channel absmax/127 对称量化;clamp_min 防全零通道除 0。
    // 这一行展开即 abs→max→div 三次独立 kernel launch——端到端 65.1µs
    // 与裸 kernel 5.9µs 之间的差距主要在此(见文件头口径说明)
    auto scales = (std::get<0>(x.abs().max(1)) / 127.0f).clamp_min(1e-8f);
    auto out = torch::empty_like(x, torch::kInt8);
    quantize_v4(x.data_ptr<float>(), scales.data_ptr<float>(),
                out.data_ptr<int8_t>(), x.size(0), x.size(1));
    return {out, scales};
```

角色：§3.3 反转的对照方，也是"零改动复用"的边界示范——`.cu` 不动一行，绑定层只做张量校验 + scale 计算 + 指针透传。关键在那**一行 C++**：`x.abs()`、`.max(1)`、 `/ 127.0f`、`.clamp_min(...)` 每一个都是一次独立的 torch 算子分发，展开就是三到四次 kernel launch，而它们各自的设备时间都近似 0。**65.1 µs 与裸 kernel 5.9 µs 的差距主要就在这一行**。为什么不能把 scale 也塞进 v4 kernel：那就不叫"零改动复用"了——本仓刻意保留这个约束，好让"对照物口径差异"本身成为可讲的内容。改错会怎样：去掉 `auto x = input.contiguous()`，非连续输入会被 v4 的行主序扁平寻址读成乱码，而且不报错。

**这段代码还示范了一个更一般的现象**：一个 kernel 的**签名**会决定它周围要长出多少胶水。`quantize_v4` 要求 scale 预置，于是调用方必须先算 scale；而算 scale 这件事在 torch 里没有单算子，只能拼。**"接口契约决定端到端性能"——这比"kernel 写得好不好" 更靠前，也更容易被忽略。**（讲义 01 §8.2 里 KV cache 那个 `.contiguous()`、讲义 02 §4 第 4 段里 `weight.t().contiguous()`，都是同一类。）

**第 6 段 · 计时协议：`wall()` 的三个约束**(scripts/test_cudagraph.py：14-18)

```python
def wall(fn, it=50, wu=10):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it*1e3
```

角色：全篇所有 µs 数字的定义。五行代码里有三个必须做对的地方： ①**warmup 10 次**在计时之外，把 JIT 编译、分配器首次分配、缓存冷启动全部排除； ②`synchronize()` 只在**计时区间的两端**各一次，中间的 `it` 次调用之间**不同步**——这正是 §2.1 那个 $\max$ 模型能成立的实验条件；③除以 `it` 得到每次调用的均值。改错会怎样：在循环体内加 `synchronize()`，主机与设备的流水被破坏，测出的就是 $\sum$ 而不是 $\max$——数字会大得很整齐，也很假，而且**你会误以为主机侧成本更高**。

**注意这个函数被 graph 臂与 eager 臂共用**（§4 第 7、8 段），这是公平性的一部分： 两臂唯一的差别只有"传进来的 `fn` 是什么"。

**第 7 段 · graph 捕获协议：预热与 side stream 缺一不可**(scripts/test_cudagraph.py：20-35)

```python
N = 100
x = torch.randn(1024, 1024, device="cuda")
out = {}

# eager 循环:N 次 triton softmax
out["eager_loop_ms"] = wall(lambda: [softmax(x) for _ in range(N)])

# graph 捕获同样的 N 次调用
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    for _ in range(3): softmax(x)          # 预热(JIT+分配器)
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g):
    for _ in range(N): y = softmax(x)
out["graph_replay_ms"] = wall(lambda: g.replay())
```

角色：§3.4 的实验主体。三个必须做对的地方，每一个都能在官方文档里找到对应条款： ①**捕获前预热 3 次**——Triton 的 JIT 编译与 caching allocator 的首次分配都必须发生在捕获**之前**（PyTorch 文档："Before capture， warm up the workload to be captured by running a few eager iterations"）；②预热放在**独立 stream** 上再 `wait_stream`（文档："Warmup must occur on a side stream"；且 "Capture must occur on a non-default stream"）；③捕获的是**同样的 N 次调用**，与 eager 臂逐字一致，这样两臂唯一的差别就只有"有没有录图"。改错会怎样：省掉预热，第一次捕获通常直接抛错或录进一段编译期开销， 得到一个好看但无意义的数字。

**还有一条隐含约束值得点出**：`x` 是**捕获前就分配好的固定张量**，重放时地址不变——这正是 PyTorch 文档那句 "Every replay reads from and writes to the same (virtual) memory addresses"。它同时也是 §3.4.3 里 L2 红利的来源：**同一个地址被重放 100 次， 缓存当然全命中**。**这个实验设计上的选择，直接决定了 3.11 µs 这个数不能外推。**

**第 8 段 · 对照臂与 per-call 换算**(scripts/test_cudagraph.py：37-48)

```python
# torch.softmax 对照(C++ 分发)
out["torch_loop_ms"] = wall(lambda: [torch.softmax(x, -1) for _ in range(N)])
g2 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g2):
    for _ in range(N): y2 = torch.softmax(x, -1)
out["torch_graph_ms"] = wall(lambda: g2.replay())

per = {k: round(v*1000/N, 2) for k, v in out.items()}   # µs/次
print("每次调用成本 (µs):", json.dumps(per, indent=1))
print(f"triton: eager {per['eager_loop_ms']} → graph {per['graph_replay_ms']} "
      f"(消掉 {per['eager_loop_ms']-per['graph_replay_ms']:.1f} µs/调用, "
      f"{per['eager_loop_ms']/per['graph_replay_ms']:.1f}x)")
```

角色：实验的公平性设计。**torch 也上 graph**(`g2`)——如果只给自己上 graph 而让对照留在 eager，那 11.6× 就是自己跟自己比。四条路径同进程、同输入张量、同 `wall()` 计时函数，唯一变量是"哪种分发 × 有没有图"。`per` 那行把总时长除以 N 换算成每调用 µs， 是全篇所有 µs 数字的定义。改错会怎样：忘记除以 N，或者在 eager 臂里每次调用后 `synchronize()`，都会让主机与设备的流水重叠被破坏，测出的就不再是 $\max$ 而是 $\sum$ (§2.1)——数字会大得很整齐，也很假。

**注意 `g2` 没有单独预热**：因为 `torch.softmax` 不需要 JIT 编译，而分配器已经被前面的 eager 臂预热过了。**这是"预热的目的是什么"想清楚之后才敢省的一步**——如果照抄 "每个 graph 前都要预热"这条规则而不理解它，反而看不出这里为什么可以省。（本仓没有做"给 `g2` 也加预热"的对照，所以这一句是设计意图的说明，不是实测结论。）

## 5. 实验数据怎么读

### 5.1 fig3:launch 四口径(对数轴)

figures/fig3_launch_cudagraph.png（脚本 scripts/plot_readme_figures.py:122-149）。

- **轴与口径**：横向条形，x = **每次调用成本（µs，对数轴）**，场景固定为 1024² softmax × 100 调用；四条自上而下 = Triton eager 循环 / torch.softmax eager / torch.softmax + Graph / Triton + Graph 重放；误差条 = 3 轮 std；源数据 data/derived/exp-t05_stability_3rounds.csv。
- **为什么必须对数轴**：36.2 与 3.11 差一个量级，线性轴会把后三条压成贴着零点的一堆， "graph 后 Triton 反超 torch"这个结论在图上就看不见了。**单图单结论**的图，轴的选择是结论的一部分。
- **颜色跟实体不跟排名**（scripts/plot_readme_figures.py：124-129 的注释）：蓝 = Triton， 红 = torch，灰 = torch+graph。读图时先看颜色配对，再看长度——否则容易把"两条蓝的" 误读成同一路径的两次测量。
- **这个设计防了哪些坑**：①四条路径**同进程、同输入张量**，排除环境漂移；②graph 捕获前预热 3 次（第 7 段），排除 JIT 编译混入；③**对照方也上 graph**，排除"只给自己开挂"； ④N=100 摊薄单次计时噪声，再除回 per-call；⑤3 轮 std 让"哪一格不稳"暴露出来。
- **误差条怎么读**：graph 重放的 std 是 **0.00**（3 轮完全一致），eager 是 0.11；而 **torch eager 的 std 达 0.737 µs（相对 9.17%）**，是全表最不稳的一格。这不是噪声， 是**主机侧分发对系统状态敏感**的直接证据（§3.6.1）——同一个机理，也解释了为什么 §3.3.1 里 Triton 融合的端到端数字跨会话在 41.6~52 µs 之间飘。
- **机理账**：消除量 $36.16 - 3.11 = 33.05\ \mu s$/调用，与 §3.2.1 估的"Triton Python 分发 ~30 µs"对得上——**graph 消掉的就是那一段，一分不多一分不少**。剩下的 3.11 µs 的口径上限见 §3.4.3（含 L2 常驻红利）。

**这张图还有一条没写在图上的读法**：graph 重放那一格的 std = 0.00 µs 本身就是证据。一个**完全没有主机侧可变成本**的量，才可能三轮给出逐位相同的读数。**误差条为零不是 "测得准"，是"这个量的性质变了"**——它从一个受操作系统调度影响的量，变成了一个由设备执行时间决定的量。

### 5.2 EXP-T03 的五行表怎么读

| 场景 | Triton | 对照 | 读法 |
|---|---|---|---|
| softmax 8×8（纯开销） | 37.4 µs | torch 8.0 | 主机侧开销的读数 |
| softmax 1024² | 37.6 µs | torch 8.1 / CUDA v4 7.8 | 被开销遮蔽 |
| softmax 8192² | 917 GB/s | torch 921 GB/s | **设备侧同速** |
| int8q 端到端（torch 层） | **51.7 µs**(1 launch) | ext-CUDA v4 65.1(4 launch) | 融合 > 单核快慢 |
| int8q 裸 bench |— | CUDA v4 5.9 µs | 不含封装的下限 |

三条读法纪律：①前两行是**终端级证据**（排障会话），存盘 raw 的同尺寸值是 36.83/7.84 µs； ②第三行的 917/921 是 EXP-T03 §5 的现行口径，按 536.9 MB / 0.585 ms 复算即得同一量级； ③第四行两个数**仍为单轮**，措辞约定明确要求对外引用时带"单轮"。第五行的 5.9 µs 来自另一个仓的实测（EXP-K01）且口径是 **scale 预置**——它和第四行的 65.1 **不是同一件事的两次测量**，是两个口径，混引就是造假。

**给这张表补一列"轮数"，是最省事的防误引措施**：

| 行 | 轮数 | 可外推性 |
|---|---|---|
| 8×8 / 1024²（三点法） | 终端级，存盘同尺寸有 3 轮锚 | 结论稳，数值随软件栈变 |
| 8192² 同速 | 存盘 1 轮 + derived 3 轮 | 结论稳，数值随硬件带宽变 |
| int8q 端到端 51.7 / 65.1 | **单轮** | **最弱的一格**，只做定性用 |
| int8q 裸 5.9 | 跨仓单值 | 口径不同，不可与上一行相减 |

**最弱的那一格恰好是结论最反直觉的那一格**——这是本篇必须自己说出来的话（§7 第 11 问会再答一次）。§3.3.2 给出的 3 轮同类对照（42.16 vs 99.70 µs）就是为了补这个洞：它证明"融合 > 多算子链"这半句，但**证明不了"融合 > 更快的单核"这半句**。

### 5.3 哪些数字能外推,哪些不能

| 数字 | 能外推的部分 | 不能外推的部分 |
|---|---|---|
| 917 / 921 GB/s(8192²) | "带宽主导尺寸下两边同速"这个**结论** | 数值，随卡的 HBM 带宽变 |
| ~36 µs Triton 地板 | "Triton 的主机侧地板远高于 torch"这个**序关系** | 数值，随 Triton/Python 版本漂移最快 |
| 11.6× 塌缩 | "Graph 消掉的就是主机侧分发"这个**机制** | 倍数，它等于"原来的坑有多深" |
| 3.11 µs 重放 | 无（含 L2 常驻红利） | 全部；换成每步不同的激活会变大 |
| 51.7 / 65.1 µs | "融合数比单核快慢更重要"这个**方向** | 数值，且**仅单轮** |
| 42.16 / 99.70 µs（int8q 3 轮） | "融合 > 多算子链"这半句，3 轮支持 | 数值；且它证明不了"融合 > 更快的单核" |
| 交叉点 ≈ 36.5 MB | **算式**（地板 × 带宽） | 数值，随地板与带宽同时变 |

**一句话规律**：比值里主机侧那一半通常可外推，设备侧那一半通常不可（§2.4 公理 C）。把这张表背下来，比背具体数字有用得多。

## 6. 误区与边界

1. **"Triton 比 CUDA 慢"**——本仓**禁裸说**（措辞约定）。同一个 kernel 在本仓数据里有三个答案：设备侧同速、单次调用贵 ~25-30 µs、端到端可能反超。说清你测的是 kernel 还是调用链，是这道题的全部。
2. **"小尺寸测出的 4.4× 是 kernel 差距"**——三点法证明它是与形状无关的**主机侧常数** (§3.2)。基于这个误判去优化 kernel，改多少遍数字都不动。
3. **"mask load 阻断了向量化"**——本仓的跑前假设，**被对照实验证伪**：加了整除快路径后数字纹丝不动（§3.2.4）；3 轮 stability 事后给出 0.4σ 的不显著差（§3.2.3）。快路径代码至今留在 src/elementwise_kernels.py：63-69 作为物证。教训的一般形式： 先设计"如果不是 X 会怎样"的对照，再动手改代码。
4. **"更快的 kernel 一定赢"**——CUDA v4 裸跑快一个量级，套上 3 次 torch 前置 launch 后端到端 65.1 输给 Triton 融合的 51.7(§3.3.1)。集成层要先数 launch 次数。 **但这一对是单轮**(§5.2)，引用必须带这个限定。
5. **"graph 重放 3.11 µs 就是这个 kernel 的真实成本"**——不是。它含 L2 常驻红利（8.39 MB / 3.11 µs = 2.7 TB/s，远超 HBM；4.19 MB 输入只占 72 MB L2 的 5.7%）， 真实推理里每步激活都不同，这个绝对值会变大（§3.4.3）。可以外推的是 **11.6× 这个主机侧塌缩倍数**，不是 3.11 这个值。
6. **"上了 Graph 就万事大吉"**——捕获要求地址稳定、非默认流、无 CPU 同步、先预热（PyTorch 文档四条，§3.4.4）；动态 shape 要分桶（vLLM 的 `cudagraph_capture_sizes`）， 桶多了捕获时间与显存都要涨；把整个引擎 step 捕获下来还需要先把 KV cache 的动态增长静态化，本仓没做（EXP-T05 §7）。
7. **"11.6× 说明 Triton 的 Graph 支持比 torch 好"**——不对，恰恰相反。塌缩倍数大是因为 **原来的主机侧成本高**（§3.4.1 与官方博客 2.8× 的对照）。倍数衡量的是坑的深度， 不是解法的质量。真正该看的是 graph 后的绝对值：3.11 vs 4.04 µs。
8. **"设备侧同速这条结论普遍成立"**——只在**带宽主导尺寸**成立（§3.1）。1024² 那一档 graph 之后 Triton 3.11 快于 torch 4.04，说明在 L2 主导的尺寸上两者**并不同速**。 **"同速"是一个带尺寸定语的结论。**
9. **"N 取多大都一样，反正要除回去"**——不一样。N=1 时测的是 $T_{\text{host}}+T_{\text{dev}}$（首次调用没有流水可言），N 大了才收敛到 $\max$；偏差量级是 $1/N$(§2.3)。**N 是实验设计的一部分，不是无关紧要的常数。**
10. **"讲义 02 的 num_stages 结论对这些行核也适用"**——不适用。Triton 文档写明 kernel 参数版的 num_stages "only pipelines loads that feed into `dot` operations"， 而本篇的三个行核里**一个 `dot` 都没有**。同一个仓里两类 kernel 的调优旋钮不同， 照搬会白忙一场。

**适用边界**：全部数字来自单卡 RTX 4090、fp32 行核（softmax / RMSNorm / int8 quantize） 与 8×8 / 1024² / 1024×1500 / 2048×{1024,4096} / 8192² 几个尺寸；三点法的中间数字为终端级证据；**binding 端到端（51.7 / 65.1 µs）仍为单轮**；裸 CUDA 的 5.9 µs 来自 Kernel_Optimazation 仓且口径为 scale 预置；§3.2.2 的交叉点（$\approx 2136^2$）是推导值， **该尺寸本仓未测**；L2 实际带宽未测，§3.4.3 只能给"必然主要来自片上"这个下界结论； 本容器无性能计数器权限，stall 级归因用变体对照替代（docs/theory/04）。

## 7. 连环追问

1. **Q：一句话回答"Triton 比 CUDA 慢吗"？** 不能一句话回答——必须分口径：设备侧（本仓测的行核与 GEMM 上，带宽主导尺寸同速）、单次调用（Triton 贵 ~25-30 µs）、端到端（看融合数，可能反超）。
2. **Q：三点法为什么能证明"时间不在 kernel 里"？** 它是一个极限夹逼：8×8 把 $T_{\text{dev}}\to0$，测到的 37.4 µs 只能是主机侧； 8192² 把 $T_{\text{dev}}$ 拉到远大于主机侧，两边打平；1024² 的 37.6 µs 与 37.4 相差不到 1%，落在主机侧那一端（§3.2.1）。**只用了一个 $\max$ 模型，没有额外假设。**
3. **Q：有没有第四个点？** 有，而且是事后才发现的：1024×1500 的数据量比 1024×1024 多 46%，时间在 3 轮口径下不可区分（0.4σ，§3.2.3）。它同时也复核了那个被证伪的 mask 假设。
4. **Q：Triton 的 launch 为什么比 torch 贵？** 每次调用要过 Python 包装：reshape/contiguous、输出分配、`next_power_of_2` 之类的编译期常量推导、JIT 缓存查找、grid 计算（第 2 段的 8 行代码）。**本仓没有逐行 profiling，"主要在 JIT 缓存查找与驱动 launch"是推断。**
5. **Q：int8 那三个数字分别是什么？** 5.9 µs = 裸 CUDA v4（scale 预置，EXP-K01）；65.1 µs = ext 绑定端到端（4 次 launch）； 51.7 µs = Triton 单 kernel 融合（跨会话 41.6~52 µs）。三口径不得混引，后两个仍为单轮。
6. **Q：为什么"更快的 kernel"会输？** 因为 v4 的签名要求 scale 预置，绑定层那一行 `x.abs().max(1)/127` 展开成三次 torch launch（第 5 段）。融合把次数压到 1，省的是分发次数与中间量往返两笔（§3.3.3）。更一般的教训是：**接口契约决定它周围要长出多少胶水。**
7. **Q：融合省的两笔账，哪笔更大？** 看尺寸。本仓 1024² 上 launch 那笔约 24 µs、中间量那笔约 16 µs 量级，同阶但未隔离（§3.3.3）；Ivanov 等人在 BERT 训练上省的主要是中间量（算子够大，launch 不是瓶颈）。 **同一个动作在不同尺寸上兑现的是不同的账。**
8. **Q：CUDA Graph 消掉的到底是什么？** 每次调用的主机侧分发——录成图后由驱动一次性提交（CUDA 文档：开销"paid once for the entire graph during instantiation"）。实测消掉约 33 µs/调用，恰好等于 §3.2.1 估的 Triton 分发段（§5.1 的机理账）。
9. **Q：graph 之后为什么 Triton 反而比 torch 快？** 主机侧因素被扣掉后剩的是 kernel 本体：Triton+graph 3.11 µs vs torch+graph 4.04 µs (§3.4.2)。所以"小核慢"的正解是上 Graph，不是换 CUDA。注意这与 §3.1 的"设备侧同速" 不矛盾：那是 HBM 主导的 8192²，这是 L2 主导的 1024²(§3.6.4)。
10. **Q：那是不是所有小 kernel 都该上 Graph？** 前提是**地址稳定**且调用序列固定（PyTorch 文档四条，§3.4.4）。动态 shape 要分桶捕获， 桶太多时捕获时间与显存反过来吃掉收益；这时才轮到"写 CUDA/C++ 扩展"这条路（§3.5 的决策树）。而且**先算再测**：用 §3.2.2 的算式估一下 $T_{\text{dev}}$， 如果它已经远超主机侧地板，上 Graph 收益接近零。
11. **压力问 Q：11.6× 这个数字，是不是靠"100 次调用同一个张量"刷出来的？** 部分是。诚实拆开：**主机侧那一半是真的**——每次调用省掉的 ~33 µs 分发与张量内容无关，换成每次不同的输入也照样省。**设备侧那一半有水分**——3.11 µs 里含"4.19 MB 输入常驻 L2"的红利（§3.4.3 的 2.7 TB/s 反推，而 L2 有 72 MB），真实推理中激活每步都换， 设备段会变长，于是**倍数会缩小**。所以可外推的是"Graph 消掉主机侧分发"这个机制与 ~33 µs 的量级，不是 11.6 这个具体倍数。本仓没做"每次换输入"的变体，如实标为推断。
12. **压力问 Q：你最反直觉的那条结论（更快的 kernel 输掉端到端），证据是最弱的一格， 这不是很尴尬吗？** 是，而且必须自己先说：51.7 / 65.1 那一对是**单轮**（§5.2 的轮数列）。本仓的补救是给同一结论找了一个 3 轮的同类证据——1024² int8 quantize 上 Triton 融合 42.16±0.35 µs vs pytorch_eager 99.70±1.74 µs(§3.3.2)。但要说清：它只证明"融合 > 多算子链"， **证明不了"融合 > 更快的单核 + 多次 launch"**。要把后半句也做实，需要把 ext 那一对补到 3 轮（§8.3 列为待办）。**在补上之前，那句话只能带"单轮"限定引用。**
13. **压力问 Q：这四层结论换个 GPU、换个 torch 版本还成立吗？** 机制成立，数字不承诺。四层里只有第一层（设备侧同速）依赖硬件带宽，另外三层依赖 **软件栈**：Python 分发成本随 Triton 版本变，torch 的分发成本随 dispatcher 实现变， graph 的节点开销随驱动变。本仓全部数字锁在单卡 RTX 4090、一套固定 venv 上（各 record §2 的环境行），换栈就要重测——尤其"Triton ~30 µs"这个数，是本篇最容易随版本漂移的一个。**但 §3.2.2 那个"先算交叉点"的方法不随栈变**：换栈只需要重新测一次地板，算式照用。
14. **压力问 Q：你说 graph 重放 std = 0.00，是不是位数不够看不出来？** 诚实答：derived 表里 `per_call_us.graph_replay_ms` 的 mean 是 3.11、std 是 0； 对应的 `total_ms.graph_replay_ms` 是 0.310944±0.000387，相对 0.12%—— **不是零，是被 per-call 那一步的两位小数截断了**。所以准确说法是"三轮的每调用读数在两位小数上完全相同"，而不是"完全没有波动"。**报数字时把精度链说清楚， 比报一个漂亮的 0 更可信。**

## 8. 工业对照与延伸

### 8.1 论文/文档怎么说 vs 本项目实测:逐条对照

| # | 来源与声称 | 本仓实测（EXP 锚） | 差异分析 |
|---|---|---|---|
| 1 | NVIDIA "Getting Started with CUDA Graphs"：逐个 launch "9.6μs per kernel (including overheads)： much higher that the kernel execution time of 2.9μs"；上 graph 后 "3.4μs" | 本仓 Triton eager 36.16±0.11 µs → graph 3.11±0.00 µs（EXP-T05 3 轮） | **机制一致，量级差一个数量级**。博客的主机侧是 C++ launch（6.7 µs 开销），本仓是 Python 分发（33 µs）。**塌缩倍数 2.8× vs 11.6× 的差，全部来自"原来的坑有多深"**，不是解法优劣 |
| 2 | 同上：建图 + 实例化约 400 µs，摊到每 kernel 约 0.02 µs | 本仓**未单独测建图成本**（捕获在计时区间之外） | 本仓的 `wall()` 只测 `g.replay()`，建图成本不在内。**这是一个已知的口径遗漏**：若某场景每次都要重新建图（动态 shape 桶未命中），400 µs 量级的成本会完全吃掉收益。本仓未测该场景 |
| 3 | CUDA C++ Programming Guide：graph 的收益是把开销 "paid once for the entire graph during instantiation" | 本仓消掉 33.05 µs/调用，与三点法估的 Python 分发段 ~30 µs 吻合 | **文档给机制，本仓给量**。这是本篇最干净的一条"文档 → 实测"对应 |
| 4 | PyTorch 文档：捕获四条约束（非默认流、无 CPU 同步、地址不变、先预热） | scripts/test_cudagraph.py：27-35 逐条实现 | 一致。**第三条("Every replay reads from and writes to the same (virtual) memory addresses")同时是 3.11 µs 那个 L2 红利的来源**——约束与偏差是同一件事的两面 |
| 5 | PyTorch 文档：graph "sacrifices the dynamic flexibility of typical eager execution in exchange for greatly reduced CPU overhead" | 本仓只测了 "reduced CPU overhead" 那一半 | 本仓**没有量化 "sacrificed flexibility" 的代价**（分桶捕获的显存与捕获时间）。vLLM 的做法是这条代价的工程形态，本仓只做了单算子最小样本 |
| 6 | vLLM 文档：按一组 batch size 捕图，运行时选不小于当前 batch 的最小那张并 padding，落空则退回 eager | 本仓只录了单个算子的 100 次调用 | **同一机理的最小样本 vs 生产形态**。本仓的缺口是 KV cache 静态化（EXP-T05 §7），不是机制理解 |
| 7 | Ivanov 等 arXiv：2007.00072：BERT 训练里 "tensor contractions account for over 99% of the arithmetic operations ... only 61% of the runtime"；"37% of the runtime ... memory-bound operators"；融合后 encoder layer 1.30×、整个 BERT 1.19×、数据移动减少最多 22.91% | 本仓 1024² int8 quantize：融合 42.16±0.35 vs eager 99.70±1.74 µs（3 轮） | **同一结论方向，不同瓶颈构成**：论文的负载里 launch 不是瓶颈（算子足够大），省的主要是中间量往返；本仓在这个尺寸上 launch 占比更大（§3.3.3）。**倍数不可比，机制可比** |
| 8 | NVIDIA GPU Performance Background User's Guide §4：三个限制因子 "memory bandwidth， math bandwidth and latency" | 本仓四层拆解里的"设备侧"对应前两个，"主机侧"**不在这三个之内** | **这是官方模型的一个盲区**：它描述的是设备内部的限制因子，而本篇一半的内容（主机侧分发）发生在设备之外。**把设备侧性能模型套到 launch 主导的场景上，会得不到任何解释** |
| 9 | Ada 白皮书 Table 2：L2 Cache Size 73728 KB | 本仓 3.11 µs 反推出 2.7 TB/s 的等效带宽，4.19 MB 输入只占 L2 的 5.7% | **容量对得上，带宽未核实**。本仓没有测 4090 的 L2 实际带宽，只能断言"超过 HBM 峰值 2.7 倍，故必然主要来自片上" |
| 10 | Triton 文档：kernel 参数版 num_stages "only pipelines loads that feed into `dot` operations" | 本仓行核里**没有 `dot`**，所以 num_stages 对它们无效 | 一致且值得点出：讲义 02 那套流水结论**不适用于本篇的行核**。同一个仓里两类 kernel 的调优旋钮不同，混用会白忙 |

**总结这张表的读法**：十条里第 3、4 两条是"文档给机制、本仓给量"的干净对应； 第 1、7 两条是"机制相同、量级不可比"；第 2、5、9 三条是**本仓已知的口径遗漏或未测项**； 第 8 条指出官方性能模型在本篇场景下的盲区。**能列出自己没测什么，和列出测了什么一样重要。**

### 8.2 与生产实现的差距各在哪一层

- **vLLM 的 CUDA Graph 用法**：decode 阶段每步几十上百次小 kernel，launch 海正是它的主要开销来源，所以整个 decode step 被录成图并按 batch size 分桶（`cudagraph_capture_sizes`）；还有 piecewise 模式——把不兼容 graph 的算子（如某些 attention 实现）留在 eager，其余进图。本仓 EXP-T05 只录了单个算子的 100 次调用， 是同一机理的最小样本；把引擎整 step 捕获需要 KV cache 静态化，本仓未做（EXP-T05 §7）。
- **profiling 的坑**：graph 内的 kernel 在时间线上默认聚成一个节点，要看内部结构需要 node 级 trace（nsys 的 `cuda-graph-trace=node`）。这是本仓在另一个仓踩过的教训， 与本篇的结论互为表里（§3.6.3；docs/theory/03 §5 的锚）。
- **torch.compile / inductor**：走的是"自动融合"这条腿——把相邻的逐元素算子拼成一个 kernel，机理与 §3.3 的手工融合相同，区别是它由编译器决定融合边界。本仓手工融合的 int8 quantize 正是这类算子的典型形态（规约 + 逐元素，一趟可完成）。
- **算子库的 epilogue 融合**：生产实现更进一步，把量化/激活/加 bias 融进 GEMM 的 epilogue，连"独立的逐元素 kernel"都不留。讲义 02 §5.3 那个 72.9 TFLOPS 的在线量化口径，正是"没做 epilogue 融合"的成本读数。
- **接口契约这一层**：本篇第 5 段那个"kernel 签名要求 scale 预置 → 周围长出三次 launch"的现象，在生产里是通过**同时设计 kernel 与调用契约**来避免的。本仓刻意保留这个约束（零改动复用），好让口径差异本身成为可讲的内容——但要清楚这是教学取舍， 不是工程最优。

### 8.3 这一篇没做的事(供下一步)

按"验证成本从低到高"排：

1. 往 scripts/test_ew_gemm.py 的形状列表里加 $2048^2$ fp32 一行，验证 §3.2.2 推出的主机/设备交叉点（改一行）。
2. 把 int8q 的 ext-CUDA 对照补到 3 轮，把 §7 第 12 问那个"最弱的一格"补实（重跑一次）。
3. 做一个"每次换输入张量"的 graph 变体，量化 §3.4.3 里 L2 红利的大小（改几行）。
4. 单独测一次建图 + 实例化成本（§8.1 第 2 条的口径遗漏）。
5. 逐行 profile 第 2 段那 8 行，把"主机侧成本主要在哪一步"从推断变成实测。
6. 扫 `num_warps` 的阈值（§3.7 里唯一的"未扫参"旋钮）。

### 8.4 延伸阅读(带精确出处,每条一句话说明它能解决什么疑问)

**论文**

1. Ivanov， Dryden， Ben-Nun， Li， Hoefler， "Data Movement Is All You Need： A Case Study on Optimizing Transformers"， arXiv：2007.00072，算子分类与 Table 1、融合一节。——想知道"transformer 里到底有多少时间花在非矩阵乘算子上"（99% 的 FLOP 只占 61% 的时间，37% 在访存受限算子里）以及"融合为什么是主要机会"， 读这两处；也是把本仓 1024² 的小样本放进真实负载语境的最好参照。
2. Williams, Waterman & Patterson, "Roofline: an insightful visual performance model for multicore architectures", CACM 52(4):65-76, DOI:10.1145/1498765.1498785。——§3.1 里"两边都贴 roofline 就说明 kernel 没差距"这条推理的框架来源。

**官方文档**

3. NVIDIA CUDA C++ Programming Guide，"CUDA Graphs" 一章：三阶段模型（定义 / 实例化 / 执行）、stream capture 的限制（"it is invalid to synchronize or query the execution status of a stream which is being captured"）、实例化后拓扑不可改。——想知道 graph 到底是什么、为什么它能把开销"paid once"，读这一章。
4. NVIDIA Technical Blog， "Getting Started with CUDA Graphs"。——想要一组官方的量化对照（9.6 → 3.8 → 3.4 µs/kernel，建图 400 µs）来校准自己的数字，读它； §8.1 第 1、2 条的对照原件。
5. PyTorch 文档 "CUDA semantics" 的 CUDA Graphs 一节。——捕获的四条约束（非默认流、无 CPU 同步、地址不变、先在 side stream 预热）与那句 "sacrifices the dynamic flexibility ... in exchange for greatly reduced CPU overhead"；§4 第 7 段每一行的依据。
6. vLLM 官方文档的 CUDA Graphs 设计页（`cudagraph_capture_sizes`、 FULL / PIECEWISE / FULL_AND_PIECEWISE 模式、按桶选图与 padding）。——想知道 "分桶捕获在生产里长什么样"以及"哪些算子必须留在 eager"，读它。
7. NVIDIA， "GPU Performance Background User's Guide" §4 Understanding Performance。——三个限制因子的官方定义；同时注意它**不覆盖主机侧**，这是 §8.1 第 8 条要点出的盲区。
8. NVIDIA， "NVIDIA Ada GPU Architecture" 白皮书 Appendix A Table 2。——1008 GB/s 与 73728 KB L2 这两个常数，§3.1 与 §3.4.3 全靠它们。
9. Triton 官方文档与本机 3.6.0 源码（`triton.Config`、`tl.range` 的 num_stages 说明）。——用来确认"讲义 02 的流水旋钮对本篇的行核无效"（行核里没有 `dot`）， §8.1 第 10 条的依据。

**源码与本仓证据**

10. src/elementwise_kernels.py：12-22—— 文件头的三口径标注，引用任何 int8 数字前先读这段。
11. src/elementwise_kernels.py：63-69 与：100-102—— 被证伪假设的物证（`EXACT` 快路径）， 与 §3.2.4 的复盘对读。
12. src/torch_ext/int8_binding.cpp：26-32—— 一行 torch 表达式展开成多次 launch 的现场； "接口契约决定胶水量"这条教训的最短样本。
13. scripts/test_cudagraph.py：14-18 与：27-35—— 计时协议（两端同步、中间不同步）与捕获协议（预热 / side stream / 录 N 次）。
14. data/derived/exp-t02_stability_3rounds.csv—— 本篇所有 3 轮数字的源： softmax 三尺寸、int8q 融合 vs eager、rmsnorm 两尺寸。
15. data/derived/exp-t05_stability_3rounds.csv—— 四条 graph/eager 路径的 per-call 与 total 两套读数（§7 第 14 问那个"std = 0.00 是怎么回事"就靠它）。
16. records/EXP-T03_ports_and_binding.md §5-§7—— 五行表、三口径约定、mask 假设的证伪全过程与单轮限定。
17. records/EXP-T05_cudagraph.md §5-§7—— 塌缩数字、捕获前提与开放问题（整 step 捕获需要 KV cache 静态化）。
18. docs/theory/03_triton_vs_cuda.md §2—— 排障三步（现象 → 假设一证伪 → 假设二坐实） 的完整叙述，以及选型准则的原始表述。
