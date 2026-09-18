# EXP-T10 · Triton FA2 backward(简化版):重算而非存储

> **一句话结论**：FA2 反向正确性 **6/6**(三梯度 max abs err ≤4.8e-3,gate 2e-2),性能 **S=4096 达 SDPA-flash backward 的 91.2%±0.2%、S=2048 达 87.8%±0.3%**(3 轮 mean±std);核心设计是**用重算换存储**——forward 只存行统计量 LSE(空间 O(S) 而非 O(S²)),backward 需要 P 时用 Q·Kᵀ 现场重算,把 HBM 从 O(S²) 拉回 O(S·D)。这与 forward 的 87% 是同一量级的**抽象税**,不是实现失误。

## 0. 元信息
| 日期 | 2026-09-16 | 环境 | v0.25.1-venv(triton3.6/torch2.11), RTX 4090 | 状态 | 完成 |
|---|---|---|---|---|---|
关联：EXP-T01《Triton FA2 forward(简化版):正确性 + 调优 + 对标》(前置,共用 Q/K/V/LSE 口径);阶段一"Flash Attention Triton 简化版"的**收尾项**(T01 §7 登记的"仅 forward"缺口)。

## 1. 目的与假设
我补上 FA2 的 backward。跑前锁定的判定有两条：**三梯度 dq/dk/dv 的 max abs err <2e-2**（与 forward gate 同口径：fp16 输入/fp32 累加，对 torch autograd 的朴素 fp32 反向比对）；**S=4096 效率 ≥SDPA-flash backward 的 70%**（参照 forward 的 87% 留余量）。

## 2. 环境与配置
- `src/fa2_bwd.py`:三个 kernel——① preprocess(Δ=ΣO⊙dO)② dK/dV ③ dQ;`src/fa2_fwd.py` 增加 `return_lse=True`(多输出 LSE=m+log l)。
- 参考物:**torch autograd** 对朴素物化 attention(fp32,repeat_interleave 展开 GQA)求导——这是"同算子同语义"的参考,不是另一个手写实现。
- 对标物:torch SDPA **FLASH_ATTENTION 后端**的 backward(同 dtype/shape);朴素 fp32 反向作第二对照(仅 S≤2048,显存 O(S²))。
- 协议:bench 100 iters + warmup 20;3 轮独立进程;tile 默认 BM64/BN64/w4/s2。

## 3. 步骤
跑 `scripts/test_fa2_bwd.py`（6 正确性形状 + 4 序列长 bench ×3 轮），再聚合到 `data/derived/exp-t10_stability_3rounds.csv`。

## 4. 原始数据
- `data/EXP-T10/20260915T1637_fa2_bwd_r{1,2,3}.json`(首字段 provenance,含 correctness + bench)
- `data/derived/exp-t10_stability_3rounds.csv`(3 轮 mean±std 聚合)
- 正确性明细(6 形状逐梯度误差)为终端级证据,逐轮打印于 stdout。

## 5. 结果

**正确性(6 形状,3 轮逐位一致)**

| B | Hq | Hkv | S | D | causal | fwd_err | dq_err | dk_err | dv_err | pass |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 8 | 512 | 64 | ✓ | 9.83e-04 | 1.09e-03 | 1.10e-03 | 1.44e-03 | ✓ |
| 2 | 16 | 16 | 1024 | 128 | ✓ | 1.13e-03 | 1.41e-03 | 1.80e-03 | 1.97e-03 | ✓ |
| 1 | 16 | 8 | 1024 | 128 | ✓ | 1.02e-03 | 2.88e-03 | 2.34e-03 | 2.62e-03 | ✓ |
| 1 | 16 | 4 | 777 | 128 | ✓ | 9.73e-04 | 1.24e-03 | 1.67e-03 | 2.99e-03 | ✓ |
| 1 | 8 | 8 | 512 | 64 | ✗ | 1.42e-04 | 4.00e-04 | 3.13e-04 | 2.06e-04 | ✓ |
| 1 | 32 | 8 | 2048 | 128 | ✓ | 1.07e-03 | 1.26e-03 | 2.04e-03 | 4.75e-03 | ✓ |

覆盖:GQA 2:1 / 4:1、非整除 S=777、非 causal、双 head_dim、Qwen3-8B 形状族。

**性能(3 轮 mean±std;效率 = SDPA 时间 ÷ 本实现时间,与 T01 同口径)**

| S | 本实现 ms | SDPA-flash ms | 效率 | vs 朴素 fp32 反向 |
|---|---|---|---|---|
| 512 | 0.2012±0.0004 | 0.4312±0.0010 | 214.4% | 9.47× |
| 1024 | 0.4230±0.0262 | 0.4996±0.0158 | 118.1% | 10.40× |
| 2048 | **1.2729±0.0035** | 1.1179±0.0027 | **87.8%** | 13.54× |
| 4096 | **4.1498±0.0076** | 3.7838±0.0074 | **91.2%** | —（O(S²) 显存不够） |

**FLOPs 口径**:backward ≈ 10·S²·D per head —— 5 个 `2·S²·D` 的矩阵乘:`QKᵀ`(重算 S) / `dO·Vᵀ`(dP) / `Pᵀ·dO`(dV) / `dS·K`(dQ) / `dSᵀ·Q`(dK);causal 减半。S=4096 时本实现 **83.1 TFLOPS**。

> **勘误(2026-09-16,由 Kernel_Optimazation#EXP-K12 反查发现)**:本记录初版写的是 `8·S²·D`,那是**不重算**的口径(存了 P,只需 4 个矩阵乘)。本实现重算 S、实际执行 5 个,正确口径是 `10·S²·D`。故绝对 TFLOPS 由 66.5 更正为 **83.1**(§5.2 表内的 ours_tflops 同比例 ×1.25)。
> **与 SDPA 的效率比值不受影响** —— `flops_bwd` 对被测项与对照项同乘,91.2%/87.8% 这两个口径仍然成立。

## 6. 分析与结论
**两条假设均成立**,但性能呈现**明显的尺寸依赖**,值得写清:
- **S≥2048（稳定区）**：87.8% / 91.2%，轮间 std ≤0.3%。这是可对外引用的数字，与 forward 的 87% 同量级。
- **S≤1024（不稳区）**：512 时 214%（反超 2.1×）、1024 时 118% 但 std 6%。**我不主张「小尺寸反超」**：同一次实验的单轮跑里，SDPA 的 512 值是 0.2809 ms，3 轮里却是 0.4312±0.0010，说明 SDPA 在短序列下的后端选择与计时本身不稳；本实现侧反而稳定（±0.2%）。诚实结论：**短序列区间两者不可比，本实现只保证自己可复现**。
- **backward 的 TFLOPS（83.1）为什么低于 forward（123）**：backward 每 FLOP 要配套读更多数据（重算要再读一遍 Q/K/V/dO），算术强度低于 forward，更接近带宽侧。这正是 recomputation 的代价，不是缺陷。（数值经 2026-09-16 勘误：原写 66.5，系 FLOPs 口径用了不重算的 8·S²·D。）

**三个实现要点(面试可讲)**:
1. **重算而非存储**:不读任何 S×S 张量,换来 HBM 从 O(S²) 回到 O(S·D)。多花的 FLOPs 是设计的一部分。
2. **两个 kernel 镜像对偶**:dK/dV 内核让 K/V 块驻留、M 方向循环;dQ 内核让 Q/dO 驻留、N 方向循环。两边累加器都**驻寄存器、写回一次、零原子** —— 原子加会让 fp32 求和顺序不确定,违反本仓 3 轮逐位可比的复现纪律。
3. **causal 的三角性要用两次**:forward 收缩 N 循环的**上界**,backward 抬升 M 循环的**下界**,合计省掉一半算力。

## 7. 异常、偏差与开放问题
**三个实现 bug(全部实测踩到,按"指纹→根因"记录)**:

| # | 指纹 | 根因 |
|---|---|---|
| 1 | `illegal memory access` + 梯度全 nan | dKdV 里用 **q 的 stride** 索引 LSE,而 LSE 是 (B,Hq,S) 独立连续张量,偏移被放大 D 倍 → 越界 |
| 2 | 同一次运行仍 `illegal memory access` | forward 的 **LSE store 用了 O 的 stride**(同类错误的反向版)→ 越界写 |
| 3 | 修好后 **dv 完全正确(1e-3)而 dq/dk 整体放大 8 倍** | 漏了 scale 的梯度链:`s=scale·(q·kᵀ)` 故 ∂s/∂q=scale·k,dq/dk 需再乘 scale;√D=8 恰好是放大倍数。**dv 不经过 scale,所以它一直是对的** —— 这个"单边正确"的指纹直接指向了根因 |

排查方法值得记：把 kernel 输出与**手写公式**（Python/torch 一行的 dS=P⊙(dP−Δ)）和 **autograd** 三者并排。结果 kernel 与手写公式吻合到 8e-3，而手写公式与 autograd 差 8.0 倍，**一步就把「kernel 写错」排除，锁定到数学**。

**开放问题**:
- **未做 split-K**:dK/dV 内核的 M 循环在长序列下是串行瓶颈(与 EXP-T04 flash-decoding 的问题同构),S 再大时这里是第一优化点。
- **GQA 组内求和是串行循环**(未做并行归约),G=4 时 dK/dV 的计算量是单头的 4 倍。
- **未做 dropout / alibi / paged**;未测 S>4096。
- 未与 **cuDNN 的 attention backward** 对照(只对了 SDPA-FLASH 与朴素 fp32)。
- 未测 fp32 路径(IEEE_DOT 口径):本实验只覆盖 fp16 输入。

## 8. 下游影响
- **补齐了 EXP-T01 §7 的「仅 forward」缺口**，FA2 在 Triton 侧现在是 fwd+bwd 闭环。
- **简历句候选**：「Triton FA2 简化版 fwd+bwd 闭环：backward 6/6 对 autograd 通过（≤4.8e-3），S=4096 达 SDPA-flash backward 的 91%（3 轮）」——与既有的 forward 87% 并列，共同支撑「抽象税」叙事。
- **方法论沉淀**：bug#3 的「单边正确」指纹，加上「kernel/手写公式/autograd 三方并排」排查法，建议进 `docs/theory/01_flashattention.md` 的调试小节。
- 红线：短序列（≤1024）的对比数字**不进对外文本**，因为 SDPA 侧不稳、不可比。
