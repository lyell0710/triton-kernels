# 讲义 03 · "Triton 比 CUDA 慢"的四层拆解:设备侧、launch、融合、CUDA Graph

> 读者:准备校招面试的作者本人,以及被"小 kernel 慢了 4 倍"卡住的工程师。
> 读法:不跳步。每个论断后面跟着它的证据锚(EXP 编号 / 文件:行号 / raw 路径),
> 所有数字与仓内现行口径逐字一致,来源见 records/ 与 data/derived/。

## 1. 这一篇回答什么问题

"Triton 比 CUDA 慢吗"是本仓明令**禁止裸答**的问题(本仓措辞约定)。原因不是政治
正确,是这句话在本仓的数据里同时有三个互相矛盾的答案。读完你应当能:

- 手推**三点法**:为什么用"极小尺寸 + 目标尺寸 + 带宽主导尺寸"三个点,就能把
  "时间到底在不在 kernel 里"这件事**证明**出来,而不是猜出来。
- 说清四层因果链:设备侧同速(8192² softmax **917 / 921 GB/s**,双双贴 roofline 91%)
  → 主机侧分发差(Triton ~30 µs > torch ~8 µs > 裸 CUDA ~5 µs)→ 端到端被**融合数**
  反转(1 次 launch 的 Triton 融合赢 4 次 launch 的 ext-CUDA)→ CUDA Graph 把 launch
  塌缩 **11.6×(36.2 → 3.11 µs/调用)**。
- 答上"int8 那三个数字到底哪个是哪个":**5.9 µs(裸 CUDA v4,scale 预置)/ 65.1 µs
  (ext 绑定端到端,4 次 launch)/ 51.7 µs(Triton 单 kernel 融合,跨会话 41.6~52 µs
  波动)**——三口径不得混引,且 binding 端到端两数**仍为单轮**。
- 拿出一棵能当场画的决策树:什么时候用 Triton、什么时候写 CUDA、什么时候上 Graph。

## 2. 直觉与第一性原理

**一次 kernel 调用到底花在哪**:成本分两段——**主机侧**(Python/C++ 分发、参数处理、
JIT 缓存查找、grid 计算、wrapper 里的临时张量分配)与**设备侧**(kernel 本体执行)。
在一个不做同步的循环里连续发射时,两段是**流水并行**的:主机在为第 $i+1$ 次调用做
准备,设备还在跑第 $i$ 次。于是
$$T_{\text{每调用}} \approx \max(T_{\text{主机}},\ T_{\text{设备}})$$
不是相加。这个 $\max$ 就是全篇的第一性原理:**你测到的数字,只反映两者中更大的那个。**

**没有这层区分会怎样**:在 1024² 上测出 Triton 37.6 µs vs torch 8.1 µs,得出
"Triton 慢 4.4 倍"的结论,然后去优化 kernel——tile、向量化、规约树改一遍,数字纹丝
不动。这正是本仓真实发生过的事(§3.2 的证伪案例),而它浪费的不是算力,是人的时间。

**日常类比与它的失效点**:像快递的"下单 + 分拣 + 派送"。类比在三处失效:①快递的三段
是串行相加的,GPU 的主机段与设备段是流水并行的(所以是 $\max$ 不是 $\sum$);
②"合并订单"在快递里省的是运费,在 GPU 上**同时**省两笔——少一趟 launch,以及少一份
中间张量的 HBM 往返(§3.3 的融合就吃这两笔);③快递没有"把整条派送路线录下来重放"
这种操作,而 CUDA Graph 恰恰是这个(§3.4)。

## 3. 完整推导与机制

### 3.1 第一层:带宽主导尺寸下,设备侧同速

先把设备侧单独看清楚。8192×8192 fp32 的行 softmax,一次读一次写:
$$2 \times 8192^2 \times 4\,\mathrm{B} = 536.9\ \mathrm{MB}$$
存盘 raw(data/raw/EXP-T02/ew_gemm_bench.json)里 Triton 0.58497 ms、torch 0.58252 ms,
换算即 **0.92 TB/s 量级**;仓内现行口径记 **917 / 921 GB/s**(EXP-T03 §5),对 4090 的
1008 GB/s roofline 是 **91%**,与 kperf 卡片"带宽 91%、occ 67%(regs 限)"一致
(终端级证据,登记于 EXP-T06 §7)。

**为什么"两边都贴 roofline"就能推出"kernel 没差距"**:两边都被同一堵 HBM 带宽墙卡住,
任何写法差异(向量化、bank conflict、规约树形状)都只能在墙内做文章,不可能反映到
墙外的总时长上。反过来说,**这个结论只在带宽主导尺寸成立**——它不是"Triton 和 CUDA
一样快"的普遍证明,而是"在这个尺寸上差距不可测"的诚实陈述。

### 3.2 第二层:三点法把时间从 kernel 里赶出来

**三点法的构造**(EXP-T03,同一个 softmax kernel,只换形状):

| 点 | 设备工作量 | Triton | 对照 | 这个点证明什么 |
|---|---|---|---|---|
| 8×8 | ≈ 0 | 37.4 µs | torch 8.0 | 纯主机侧开销的**读数** |
| 1024² | 小 | 37.6 µs | torch 8.1 / CUDA v4 7.8 | 37.6 ≈ 37.4 → 时间**不在 kernel 里** |
| 8192² | 大(带宽主导) | 917 GB/s | torch 921 GB/s | 设备侧**同速**(§3.1) |

推理链一步一步:①8×8 的设备工作量近似 0,所以那 37.4 µs **只能**是主机侧;
②1024² 的 37.6 µs 与 37.4 µs 相差不到 1%,说明这个尺寸上设备侧仍被主机侧盖住;
③8192² 让设备侧显形,两边打平。三个点连起来,结论是**唯一的**:小尺寸的 4.4× 差距
是一个**与形状无关的主机侧常数**,不是 kernel 的差距。

于是有了三口径的分发成本:**Triton 的 Python 分发 ~30 µs > torch 的 C++ 分发 ~8 µs >
裸 CUDA ~5 µs**。Triton 贵在每次调用都要过 Python 包装(参数处理、JIT 缓存查找、
grid 计算)以及 wrapper 里的临时分配。

**数字分层要说清楚**(诚实度要求):三点法用的是排障会话的**终端级证据**
(EXP-T03 §5 登记);同尺寸的存盘 raw 值是 36.83 / 7.84 µs,3 轮 stability 是
36.33±0.44 / 8.24±0.36 µs(data/derived/exp-t02_stability_3rounds.csv)。三组数字
同量级,**结论不依赖小数点**——但引用时要说清是哪一组。

**一个被证伪的假设,完整复盘**(本仓最有教学价值的一次失败):
- **跑前假设**:小尺寸慢是因为掩码 load 阻断了 128-bit 向量化访存。
- **动作**:给 softmax 与 int8 quantize 各加一条"整除时走无 mask 快路径"的分支
  (`EXACT`,src/elementwise_kernels.py:63-69 与 :100-102)。
- **实测**:数字**纹丝不动**。假设作废。
- **处理**:快路径语义无害,**代码留在仓里**作为"猜测必须交给对照实验"的物证;
  记录里保留全过程(EXP-T03 §7)。随后改测三点法,才坐实了主机侧假设。

方法论提炼:**"我觉得瓶颈是 X"必须先设计出一个"如果不是 X 会怎样"的对照,再动手改
代码**;否则改完看不出差别时,你连"是没效果还是改错了"都分不清。

### 3.3 第三层:端到端反转,融合数比单核快慢更重要

第三层的对象换成 int8 per-channel quantize,因为它有**两种实现路径**可比:

- **ext-CUDA 路径**:复用 Kernel_Optimazation 的 `quantize_v4` CUDA kernel(零改动),
  用 torch extension 绑进来。但 v4 的签名要求 **scale 预置**,所以 scale 只能在绑定层
  用 torch 算——`(std::get<0>(x.abs().max(1)) / 127.0f).clamp_min(1e-8f)` 这**一行**
  展开就是 abs → max → div 三次独立 kernel launch,加上 v4 本身共 4 次
  (src/torch_ext/int8_binding.cpp:26-32)。端到端 **65.1 µs**。
- **Triton 融合路径**:absmax 规约、缩放、舍入、写回 scale **一趟做完**,
  **1 次 launch**(src/elementwise_kernels.py:91-113)。端到端 **51.7 µs**。
- **裸 kernel 口径**:同一个 v4 kernel 单独 bench 只有 **5.9 µs**(EXP-K01,口径 =
  scale 预置,不含 torch 封装)。

**"更快的 kernel 输掉端到端"**:v4 本体比 Triton 版快一个量级,端到端却输 13.4 µs。
账要这么算(不能简单地"3 × 8 µs = 24 µs"):Triton 路径 ≈ 1 次贵分发 + 1 次设备本体;
ext 路径 ≈ 4 次便宜分发 + 4 次设备本体(其中三个是极小的规约 kernel,设备时间可忽略但
每个都要付一次分发)+ pybind 与张量校验。两条路各有各的贵法,而融合把**次数**这一项
压到 1。所以本仓的结论句是:**在 torch 集成层,融合数(launch 数)比单 kernel 的快慢
更重要。**

**三口径约定**(不得混引):5.9 µs(裸,scale 预置)/ 65.1 µs(ext 端到端)/
51.7 µs(Triton 融合)。第三个数跨会话在 **41.6~52 µs** 之间波动(主机侧开销对系统
状态敏感),引用时带区间;同尺寸的 Triton 融合有 3 轮锚 42.2±0.4 µs
(int8q_1024x1024,data/derived/exp-t02_stability_3rounds.csv),但**与 ext 头对头的
那一对(51.7 / 65.1)仍是单轮**,对外引用必须注明。

### 3.4 第四层:CUDA Graph 把 launch 归零

**机制**:graph 捕获把一段调用序列(这里是 100 次 softmax)录成一张有向图,重放时由
驱动一次性提交整张图,Python/C++ 的每次分发全部变成图里的节点。也就是说,§3.2 里那个
"与形状无关的主机侧常数"被**摊到一次提交里**。

**数字**(EXP-T05,1024² fp32 softmax × 100 调用,3 轮,每次调用 µs):

| 路径 | eager | + CUDA Graph | 塌缩 |
|---|---|---|---|
| Triton softmax | 36.2±0.1 | **3.11±0.00** | **11.6×**,消掉约 33 µs/调用 |
| torch.softmax | 8.03±0.74 | 4.04±0.01 | 2.0× |

两个推论:①**graph 后 Triton(3.11)反超 torch eager(8.03)与 torch+graph(4.04)**
——所以"Triton 小核慢"的正解是**上 Graph,而不是换 CUDA**;②torch+graph 的 4.04 仍慢于
Triton+graph 的 3.11,说明把主机侧因素扣掉之后,**Triton 的 kernel 本体在这个形状上
本来就更快**——这与 §3.1 的"设备侧同速"不矛盾:那是带宽主导的 8192²,这是被 L2 装得下
的 1024²,两个尺寸问的不是同一个问题。

**这个数字的口径上限(本讲义主动指出)**:1024² fp32 一读一写 = 8.39 MB,除以 3.11 µs
得 **2.7 TB/s**,远超 HBM 的 1008 GB/s。唯一的解释是这份 4 MB 输入在 100 次重放里
**常驻 L2**。所以 3.11 µs 是"输入 L2 常驻 + 同一张量重复"的**下限口径**;真实推理里
每步的激活都不同,这个数会变大。**11.6× 的塌缩倍数是主机侧结论,可以外推;3.11 µs
这个绝对值是设备侧结论,不要外推。**(推断,本仓未做"每次换输入"的变体。)

**捕获的前提**:地址稳定——静态输入/输出张量,重放时只能改内容不能改地址;动态 shape
需要**分桶捕获**(vLLM 的 `cudagraph_capture_sizes` 就是这件事)。本仓把整个引擎 step
捕获下来的尝试没有做,因为 KV cache 的动态增长需要先静态化(EXP-T05 §7)。

### 3.5 四层合起来:一棵可以当场画的决策树

四层因果链一句话串起来:**设备侧同速 → 差距全在主机侧分发 → 端到端由融合数决定 →
主机侧这一层的终局解是 CUDA Graph。** 由此得到选型决策树:

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

补一条选型直觉:**大 kernel、长序列、融合机会多 → Triton 白给**(FA2 达 SDPA-flash
的 87%、GEMM 打平 cuBLAS,讲义 01/02);**微 kernel 高频调用 → 裸 CUDA 或 Graph**;
**torch 集成层 → 先数 launch 次数,再谈 kernel 快慢**。

## 4. 代码逐段走读:行核、绑定层与 graph 捕获

按执行顺序走读(引用为仓内真实代码逐字拷贝,标 文件:起-止行)。

**第 1 段 · softmax kernel:减 max 与那条被证伪的快路径**(src/elementwise_kernels.py:58-78)

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

角色:讲义 01 §3.1 的非分块版本,同时也是 §3.2 那次证伪的**物证**。`EXACT` 分支是当年
为验证"mask load 阻断向量化"假设加的,实测数字纹丝不动,假设作废,但因为语义无害而
留在仓里。两个数值细节:①慢路径的越界填充是 `-inf` 而不是 0(`exp(-inf)=0` 才不进分母,
补 0 会污染归一化);②快路径里 `mask = offs < BLOCK` 恒真,存在的唯一理由是与慢路径
共用同一个 store 签名。改错会怎样:把 `other` 从 `-inf` 改成 0,非整除列宽的行会多出
若干份 $e^{0-\max}$ 的假质量,softmax 的每一项都偏小——而 1024×1024 这种整除形状**测
不出来**(它走的是快路径),只有 1024×1500 那一格会炸。

**第 2 段 · launcher:每次调用的主机侧工作量就在这几行**(src/elementwise_kernels.py:81-88)

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

角色:§3.2 那 ~30 µs 的**来源现场**。这 8 行每次调用都要执行:reshape 与 `contiguous`、
`empty_like` 分配、`next_power_of_2` 计算、`EXACT` 与 `num_warps` 的推导,然后才进
Triton 的 JIT 缓存查找与 grid 计算。对 8192² 它们完全可以忽略,对 1024² 它们就是全部。
`num_warps` 的启发式(宽行 8、窄行 4)也在这里:宽行要更多 warp 才能把行内 load 与
规约流水打满,窄行给 8 warps 反而把行切碎。改错会怎样:把 `BLOCK` 写成 `n_cols` 而不
`next_power_of_2`,`tl.arange` 编译期直接报错——这是 Triton 把"编译期约束"暴露在
Python 侧的典型形态,也是每次调用都要做一次的那类计算。

**第 3 段 · int8 融合 kernel:一趟做完就是全部秘密**(src/elementwise_kernels.py:91-113)

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

角色:§3.3 反转的我方。它做的事和 ext 路径**完全一样**——absmax 规约、算 scale、缩放、
舍入、截断、写回 q 与 scale——区别只在于这些步骤在**同一个 kernel 内**完成,因此
`x` 只从 HBM 读一次,scale 从不落地。三个细节:①`rint` 是就近舍入,不是截断,
截断会引入半个 LSB 的系统性偏差;②`clamp` 到 ±127 而**弃用 -128**,是对称量化的要求
(要保证 $q$ 与 $-q$ 都可表示);③`tl.maximum(scale, 1e-8)` 防全零行除 0。
改错会怎样:把 scale 计算挪回 torch 侧(哪怕 kernel 一行不改),端到端立刻退化成 ext
路径那种多 launch 形态——**这就是第 4 段要展示的反面教材**。

**第 4 段 · 绑定层:一行 torch 表达式 = 三次 launch**(src/torch_ext/int8_binding.cpp:17-33)

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

角色:§3.3 反转的对照方,也是"零改动复用"的边界示范——`.cu` 不动一行,绑定层只做
张量校验 + scale 计算 + 指针透传。关键在第 29 行那**一行 C++**:`x.abs()`、`.max(1)`、
`/ 127.0f`、`.clamp_min(...)` 每一个都是一次独立的 torch 算子分发,展开就是三到四次
kernel launch,而它们各自的设备时间都近似 0。**65.1 µs 与裸 kernel 5.9 µs 的差距主要
就在这一行**。为什么不能把 scale 也塞进 v4 kernel:那就不叫"零改动复用"了——本仓刻意
保留这个约束,好让"对照物口径差异"本身成为可讲的内容。改错会怎样:去掉
`auto x = input.contiguous()`,非连续输入会被 v4 的行主序扁平寻址读成乱码,而且不报错。

**第 5 段 · graph 捕获协议:预热与 side stream 缺一不可**(scripts/test_cudagraph.py:20-35)

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

角色:§3.4 的实验主体。三个必须做对的地方:①**捕获前预热 3 次**——Triton 的 JIT 编译
与 caching allocator 的首次分配都必须发生在捕获**之前**,否则要么把编译过程录进图、
要么捕获直接失败;②预热放在**独立 stream** 上再 `wait_stream`,是 PyTorch graph
捕获的标准前置(避免默认流上的历史操作被卷进来);③捕获的是**同样的 N 次调用**,与
eager 臂逐字一致,这样两臂唯一的差别就只有"有没有录图"。改错会怎样:省掉预热,
第一次捕获通常直接抛错或录进一段编译期开销,得到一个好看但无意义的数字。

**第 6 段 · 对照臂与 per-call 换算**(scripts/test_cudagraph.py:37-48)

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

角色:实验的公平性设计。**torch 也上 graph**(`g2`)——如果只给自己上 graph 而让对照
留在 eager,那 11.6× 就是自己跟自己比。四条路径同进程、同输入张量、同 `wall()` 计时
函数,唯一变量是"哪种分发 × 有没有图"。`per` 那行把总时长除以 N 换算成每调用 µs,
是全篇所有 µs 数字的定义。改错会怎样:忘记除以 N,或者在 eager 臂里每次调用后
`synchronize()`,都会让主机与设备的流水重叠被破坏,测出的就不再是 $\max$ 而是 $\sum$
(§2)——数字会大得很整齐,也很假。

## 5. 实验数据怎么读

### 5.1 fig3:launch 四口径(对数轴)

figures/fig3_launch_cudagraph.png(脚本 scripts/plot_readme_figures.py:122-149)。

- **轴与口径**:横向条形,x = **每次调用成本(µs,对数轴)**,场景固定为 1024² softmax
  × 100 调用;四条自上而下 = Triton eager 循环 / torch.softmax eager / torch.softmax +
  Graph / Triton + Graph 重放;误差条 = 3 轮 std;源数据
  data/derived/exp-t05_stability_3rounds.csv。
- **为什么必须对数轴**:36.2 与 3.11 差一个量级,线性轴会把后三条压成贴着零点的一堆,
  "graph 后 Triton 反超 torch"这个结论在图上就看不见了。**单图单结论**的图,轴的选择
  是结论的一部分。
- **颜色跟实体不跟排名**(scripts/plot_readme_figures.py:124-129 的注释):蓝 = Triton,
  红 = torch,灰 = torch+graph。读图时先看颜色配对,再看长度——否则容易把"两条蓝的"
  误读成同一路径的两次测量。
- **这个设计防了哪些坑**:①四条路径**同进程、同输入张量**,排除环境漂移;②graph 捕获
  前预热 3 次(第 5 段),排除 JIT 编译混入;③**对照方也上 graph**,排除"只给自己开挂";
  ④N=100 摊薄单次计时噪声,再除回 per-call;⑤3 轮 std 让"哪一格不稳"暴露出来。
- **误差条怎么读**:graph 重放的 std 是 **0.00**(3 轮完全一致),eager 是 0.11;而
  **torch eager 的 std 达 0.737 µs(相对 9.17%)**,是全表最不稳的一格。这不是噪声,
  是**主机侧分发对系统状态敏感**的直接证据——同一个机理,也解释了为什么 §3.3 里
  Triton 融合的端到端数字跨会话在 41.6~52 µs 之间飘。**设备侧数字稳,主机侧数字飘**,
  是这一整篇的经验规律。
- **机理账**:消除量 $36.16 - 3.11 = 33.05\ \mu s$/调用,与 §3.2 估的"Triton Python
  分发 ~30 µs"对得上——**graph 消掉的就是那一段,一分不多一分不少**。剩下的 3.11 µs
  的口径上限见 §3.4(含 L2 常驻红利)。

### 5.2 EXP-T03 的五行表怎么读

| 场景 | Triton | 对照 | 读法 |
|---|---|---|---|
| softmax 8×8(纯开销) | 37.4 µs | torch 8.0 | 主机侧开销的读数 |
| softmax 1024² | 37.6 µs | torch 8.1 / CUDA v4 7.8 | 被开销遮蔽 |
| softmax 8192² | 917 GB/s | torch 921 GB/s | **设备侧同速** |
| int8q 端到端(torch 层) | **51.7 µs**(1 launch) | ext-CUDA v4 65.1(4 launch) | 融合 > 单核快慢 |
| int8q 裸 bench | — | CUDA v4 5.9 µs | 不含封装的下限 |

三条读法纪律:①前两行是**终端级证据**(排障会话),存盘 raw 的同尺寸值是 36.83/7.84 µs;
②第三行的 917/921 是 EXP-T03 §5 的现行口径,按 536.9 MB / 0.585 ms 复算即得同一量级;
③第四行两个数**仍为单轮**,LEDGER 明确要求对外引用时带"单轮"。第五行的 5.9 µs 来自
另一个仓的实测(EXP-K01)且口径是 **scale 预置**——它和第四行的 65.1 **不是同一件事的
两次测量**,是两个口径,混引就是造假。

## 6. 误区与边界

1. **"Triton 比 CUDA 慢"**——本仓**禁裸说**(措辞约定)。同一个 kernel 在本仓数据里
   有三个答案:设备侧同速、单次调用贵 ~25-30 µs、端到端可能反超。说清你测的是 kernel
   还是调用链,是这道题的全部。
2. **"小尺寸测出的 4.4× 是 kernel 差距"**——三点法证明它是与形状无关的**主机侧常数**
   (§3.2)。基于这个误判去优化 kernel,改多少遍数字都不动。
3. **"mask load 阻断了向量化"**——本仓的跑前假设,**被对照实验证伪**:加了整除快路径
   后数字纹丝不动(§3.2)。快路径代码至今留在
   src/elementwise_kernels.py:63-69 作为物证。教训的一般形式:先设计"如果不是 X 会
   怎样"的对照,再动手改代码。
4. **"更快的 kernel 一定赢"**——CUDA v4 裸跑快一个量级,套上 3 次 torch 前置 launch 后
   端到端 65.1 输给 Triton 融合的 51.7(§3.3)。集成层要先数 launch 次数。
5. **"graph 重放 3.11 µs 就是这个 kernel 的真实成本"**——不是。它含 L2 常驻红利
   (8.39 MB / 3.11 µs = 2.7 TB/s,远超 HBM),真实推理里每步激活都不同,这个绝对值
   会变大(§3.4)。可以外推的是 **11.6× 这个主机侧塌缩倍数**,不是 3.11 这个值。
6. **"上了 Graph 就万事大吉"**——捕获要求地址稳定;动态 shape 要分桶(vLLM 的
   `cudagraph_capture_sizes`);把整个引擎 step 捕获下来还需要先把 KV cache 的动态增长
   静态化,本仓没做(EXP-T05 §7)。

**适用边界**:全部数字来自单卡 RTX 4090、fp32 行核(softmax / int8 quantize)与
1024²/8192² 两个尺寸族;三点法的中间数字为终端级证据;**binding 端到端(51.7 / 65.1 µs)
仍为单轮**;裸 CUDA 的 5.9 µs 来自 Kernel_Optimazation 仓且口径为 scale 预置;
本容器无性能计数器权限,stall 级归因用变体对照替代(docs/theory/04)。

## 7. 连环追问

1. **Q:一句话回答"Triton 比 CUDA 慢吗"?**
   不能一句话回答——必须分口径:设备侧(本仓测的行核与 GEMM 上同速)、单次调用
   (Triton 贵 ~25-30 µs)、端到端(看融合数,可能反超)。
2. **Q:三点法为什么能证明"时间不在 kernel 里"?**
   因为 8×8 的设备工作量近似 0,它测到的 37.4 µs 只能是主机侧;而 1024² 的 37.6 µs 与
   之相差不到 1%,说明这个尺寸的时间也几乎全在主机侧(§3.2)。
3. **Q:Triton 的 launch 为什么比 torch 贵?**
   每次调用要过 Python 包装:参数处理、`next_power_of_2` 之类的编译期常量推导、JIT
   缓存查找、grid 计算,以及 wrapper 里的临时分配(第 2 段的 8 行代码)。
4. **Q:int8 那三个数字分别是什么?**
   5.9 µs = 裸 CUDA v4(scale 预置,EXP-K01);65.1 µs = ext 绑定端到端(4 次 launch);
   51.7 µs = Triton 单 kernel 融合(跨会话 41.6~52 µs)。三口径不得混引,后两个仍为单轮。
5. **Q:为什么"更快的 kernel"会输?**
   因为 v4 的签名要求 scale 预置,绑定层那一行 `x.abs().max(1)/127` 展开成三次 torch
   launch(第 4 段)。融合把次数压到 1,省的是分发次数与中间量往返两笔。
6. **Q:CUDA Graph 消掉的到底是什么?**
   每次调用的主机侧分发——录成图后由驱动一次性提交。实测消掉约 33 µs/调用,恰好等于
   §3.2 估的 Triton 分发段(§5.1 的机理账)。
7. **Q:graph 之后为什么 Triton 反而比 torch 快?**
   主机侧因素被扣掉后剩的是 kernel 本体:Triton+graph 3.11 µs vs torch+graph 4.04 µs
   (§3.4)。所以"小核慢"的正解是上 Graph,不是换 CUDA。
8. **Q:那是不是所有小 kernel 都该上 Graph?**
   前提是**地址稳定**且调用序列固定。动态 shape 要分桶捕获,桶太多时捕获与显存成本
   反过来吃掉收益;这时才轮到"写 CUDA/C++ 扩展"这条路(§3.5 的决策树)。
9. **Q:融合和 Graph 是替代关系吗?**
   不是,是叠加关系,而且**顺序有讲究**:先融合(减少节点数与中间张量),再上 Graph
   (摊平剩下的分发)。融合还能省 HBM 往返,这是 Graph 给不了的。
10. **Q:你怎么知道 8192² 那个点是带宽主导而不是别的?**
    手算:536.9 MB / 0.585 ms ≈ 0.92 TB/s,对 1008 GB/s 的 roofline 是 91%
    (§3.1);kperf 卡片的"带宽 91%、算力 0%"是同一结论的另一个来源(终端级证据)。
11. **压力问 Q:11.6× 这个数字,是不是靠"100 次调用同一个张量"刷出来的?**
    部分是。诚实拆开:**主机侧那一半是真的**——每次调用省掉的 ~33 µs 分发与张量内容
    无关,换成每次不同的输入也照样省。**设备侧那一半有水分**——3.11 µs 里含"4 MB 输入
    常驻 L2"的红利(§3.4 的 2.7 TB/s 反推),真实推理中激活每步都换,设备段会变长,
    于是**倍数会缩小**。所以可外推的是"Graph 消掉主机侧分发"这个机制与 ~33 µs 的量级,
    不是 11.6 这个具体倍数。本仓没做"每次换输入"的变体,这一点如实标为推断。
12. **压力问 Q:这四层结论换个 GPU、换个 torch 版本还成立吗?**
    机制成立,数字不承诺。三层里只有第一层(设备侧同速)依赖硬件带宽,另外三层依赖
    **软件栈**:Python 分发成本随 Triton 版本变,torch 的分发成本随 dispatcher 实现变,
    graph 的节点开销随驱动变。本仓全部数字锁在单卡 RTX 4090、一套固定 venv 上
    (各 record §2 的环境行),换栈就要重测——尤其"Triton ~30 µs"这个数,是本篇最
    容易随版本漂移的一个。

## 8. 工业对照与延伸

- **vLLM 的 CUDA Graph 用法**:decode 阶段每步几十上百次小 kernel,launch 海正是它的
  主要开销来源,所以整个 decode step 被录成图并按 batch size 分桶
  (`cudagraph_capture_sizes`)。本仓 EXP-T05 只录了单个算子的 100 次调用,是同一机理的
  最小样本;把引擎整 step 捕获需要 KV cache 静态化,本仓未做(EXP-T05 §7)。
- **profiling 的坑**:graph 内的 kernel 在时间线上默认聚成一个节点,要看内部结构需要
  node 级 trace(nsys 的 cuda-graph-trace=node)。这是本仓在另一个仓踩过的教训,
  与本篇的结论互为表里(docs/theory/03 §5 的锚)。
- **torch.compile / inductor**:走的是"自动融合"这条腿——把相邻的逐元素算子拼成一个
  kernel,机理与 §3.3 的手工融合相同,区别是它由编译器决定融合边界。本仓手工融合的
  int8 quantize 正是这类算子的典型形态(规约 + 逐元素,一趟可完成)。
- **算子库的 epilogue 融合**:生产实现更进一步,把量化/激活/加 bias 融进 GEMM 的
  epilogue,连"独立的逐元素 kernel"都不留。讲义 02 §5.2 那个 72.9 TFLOPS 的在线量化
  口径,正是"没做 epilogue 融合"的成本读数。

延伸阅读(源码/文档锚):

1. src/elementwise_kernels.py:12-22 —— 文件头的三口径标注,引用前先读这段。
2. src/torch_ext/int8_binding.cpp:26-32 —— 一行 torch 表达式展开成多次 launch 的现场。
3. scripts/test_cudagraph.py:27-35 —— 捕获协议(预热 / side stream / 录 N 次)。
4. docs/theory/03_triton_vs_cuda.md §2 —— 排障三步(现象 → 假设一证伪 → 假设二坐实)
   的完整叙述,以及选型准则的原始表述。
5. records/EXP-T03_ports_and_binding.md §5-§7 与 records/EXP-T05_cudagraph.md §5-§7
   —— 五行表、三口径约定、单轮限定与 graph 捕获的开放问题。
