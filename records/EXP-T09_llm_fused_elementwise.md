# EXP-T09 · Triton 版 LLM 融合逐元素算子(fused_add_rmsnorm / rope / silu_and_mul)

> **一句话结论**：HBM 区间里 Triton、手写 CUDA、torch.compile 三者两两差距 <2%，全部收敛到峰值 88-92%；分水岭是**融不融合**（相对 pytorch_eager 1.7-5.2x），不是用什么语言写。

## 0 元信息

| 项 | 值 |
|---|---|
| 日期 | 2026-08-26 |
| 环境 | 4090 容器，venv:/root/venvs/main(torch 2.13.0+cu132, triton 3.7.1) |
| 状态 | 完成 |
| 关联 | **数字的权威在 Kernel_Optimazation#EXP-K05《LLM 融合逐元素算子三件套》**（三种实现同一 harness 受测）；本记录只登记 Triton 侧的实现与踩坑。引擎接入见 llm-engine#EXP-D23《融合逐元素算子接入》 |
| 产物 | `src/llm_fused.py` |

## 1 目的与假设

为姊妹仓 Kernel_Optimazation 的三个手写 CUDA 算子提供同协议的 Triton 对照臂， 补上本项目「什么时候该用 CUDA」判断曲线的第三个点：**访存主导的融合逐元素算子**。前两点已有：计算主导 GEMM 手写够到真 cuBLAS 85.6%（Kernel_Optimazation#EXP-K02《CUDA Tensor Core GEMM 版本梯》）； 融合型 attention 的 wmma 版只够到自家 Triton 28%（Kernel_Optimazation#EXP-K03《CUDA FA2 forward 简化版版本梯》）。

假设：本类算子上 Triton 与手写 CUDA 在 HBM 区间打平（±5%），因为两者撞同一堵带宽墙。

## 2 环境与配置

三个 kernel 实现在 `src/llm_fused.py`，由 Kernel_Optimazation 各子项目的 `bench.py` 以可选臂 import（单一事实源：实现只有一份，不在两仓各放一份）。所有臂共用 `Kernel_Optimazation/scripts/bench_common.py::timeit` 的计时， 因此本次跨语言比较是**同 harness 实测级**，不再是本项目此前的"跨 harness 推断级"。

## 3 步骤

见 Kernel_Optimazation#EXP-K05 §3（在各算子子目录跑 `bench.py`，Triton 臂自动挂载）。

## 4 原始数据

`/root/projects/Kernel_Optimazation/{fused-norm,rope,activation}/project-proof/data/` 下的 3 轮 raw 与 derived（本仓不复制一份数字，避免两代数字并存）。

## 5 结果

HBM 区间（3 轮 mean，GB/s，占 4090 的 1008 峰值%）：

| 算子 | Triton | 手写 CUDA 最优 | torch.compile |
|---|---|---|---|
| fused_add_rmsnorm | 922.1 (91.5%) | 920.3 (91.3%) | 917.4 (91.0%) |
| rope | 898.5 (89.1%) | 905.9 (89.9%) | 877.2 (87.0%) |
| silu_and_mul | 928.0 (92.1%) | 927.7 (92.0%) | 925.7 (91.8%) |

L2 常驻与 decode 区间 Triton 明显落后（手写快 1.7-2.9x / 5-6x），明细见 Kernel_Optimazation#EXP-K05 §5。

## 6 分析与结论

**假设成立：HBM 区间三种实现两两差距均 <2%，全部收敛到 88-92% 峰值。** 分水岭是"融不融合"（相对 pytorch_eager 1.7-5.2x），不是"用什么语言写"(<2%)。

Triton 在这类算子上有一处结构优势值得记：**手写 CUDA 需要显式写出"寄存器缓存版"（fused-norm 的 v4）才能把整行留在片上，而 Triton 的自然写法天然就是那一版**——一个 program 处理一行，整行常驻寄存器，归约由编译器生成。代价是 Triton 无法表达 rope 的"q/k 合并 launch"（手写 v2 那一级），见 §7。

落后的两个区间也有明确归因：decode 是 Triton 的 Python 分发开销（与 EXP-T03《三件套移植 + torch 绑定》的 ~30us 一致）；L2 区间是手写版更细的访存与线程配置控制。

## 7 异常、偏差与开放问题

**rope 的粒度试错三轮，每轮都是访存形态问题，是本记录最有价值的部分：**

1. 一个 program 一个（token， head）：T=32768/HQ=32 时发 100 万个 program， 每个只搬 64 个元素，调度开销吃掉一半设备时间。
2. 一个 program 一个 token、内部开 `[BQ, HALF]` 二维 tile：program 数降到 T， 但行跨度是 D（读 128B 跳 128B），访存被拆成半事务。
3. q/k 合进一个 kernel、用 mask 选择：有效带宽**恰好**是手写版的一半。 "恰好一半"说明搬了两倍字节，即两路 masked load 都发出了实际访存—— **Triton 的 mask 保证语义正确，不保证被 mask 掉的那路不产生事务**。最终版拆成单张量 kernel、q/k 分两次 launch。

**一个 harness 层面的教训（诊断路径值得复刻）。** 上述第 3 步之前，Triton 臂一直测出"慢 2 倍"。排查顺序：换粒度（无效）→ 加 `tl.max_contiguous`/`tl.multiple_of` 连续性断言（无效）→ **dump PTX**，看到 `ld.global.v4.b32`，证明向量化本身没问题 → 单独 benchmark 该 kernel，得 907 GB/s，与手写持平。真因在 bench 侧： `q.clone()` 被写进了被计时的闭包，每次迭代白搬 320MB。 **"恰好慢 2 倍"这种整数倍关系是 harness bug 的典型指纹，不是性能现象**； 就地算子的 bench 尤其容易在这里翻车——正确性需要干净副本，时延不需要。

其他：
- Triton 版 rope 比手写少一级优化（q/k 合并 launch），decode 区间的跨语言比较因此含一次额外 launch 的偏差，不是纯 kernel 差异。
- 未做 autotune（三个 kernel 的 `num_warps`/`BLOCK` 取经验值）。 rope 曾单独扫过 BLOCK×num_warps 共 20 组，最优 907.6 与所用配置 901.9 相差 0.6%，说明该算子对配置不敏感；另两个未扫。

## 8 下游影响

- `src/llm_fused.py` 已被 llm-engine 作为 `LLME_FUSED=triton` 后端接入（llm-engine#EXP-D23：TTFT -20.5%、TPOT -20.8%）。
- 本项目「Triton vs CUDA」的结论从两点扩为三点曲线；HBM 区间的"打平"结论为**同 harness 实测级**，可去掉推断级限定。
