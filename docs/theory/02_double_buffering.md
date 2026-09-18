---
topic: 双缓冲 / 软件流水(GEMM)
status: 完成(实证=EXP-T02《流水线 GEMM》)
---

# 02 · 双缓冲:一件事的 CUDA 写法与 Triton 写法

> 8/24 勘误：全文数字以存盘 raw(data/raw/EXP-T02/ew_gemm_bench.json) 改写，首轮未存盘数字（162.1@s4 等）作废（EXP-T02 §7）； 旧版见 docs/archive/02_double_buffering_20260823.md。

## 1. 一句话结论

GEMM 主循环里「载入下一块」和「计算当前块」天然可重叠。CUDA 里手写两块 shared memory + cp.async 交替（双缓冲），Triton 里写 `num_stages=N` 让编译器生成 N 级软件流水。本仓实测（4090，fp16 4096³，单轮，EXP-T02）：**1 级（无重叠）131.9 TFLOPS → 最优 stages=3 160.5 TFLOPS（+22%），与 cuBLAS（159.8）打平（差 0.4% 内）**；8B up_proj 形状 154.4（stages=4）**反超 cuBLAS（147.3）4.8%**。

## 2. 机制(一步一步)

**为什么要重叠**：主循环每轮 = 载入 A/B 的 K 块（global→片上，数百周期延迟）+ tl.dot（tensor core）。串行执行时 tensor core 在等数据，GEMM 就从算力受限退化成延迟受限。

**CUDA 手写版长什么样**（第 5 课「中下」的正主）：
```
__shared__ half smA[2][BM][BK], smB[2][BK][BN];
预取 tile0 → __pipeline_commit
for k:
    预取 tile(k+1) 进 buf[k+1 & 1]      // cp.async,不占用计算单元
    __pipeline_wait_prior(1)             // 等 tile(k) 就绪即可
    mma 计算 buf[k & 1]
    __syncthreads()
```
要点：cp.async 直通 global→shared（绕过寄存器往返）；两块缓冲让「搬运 k+1」与「计算 k」在时间上错开；深度>2 时是环形多缓冲。

**Triton 版**：同一个朴素循环 `load → dot → 指针步进`，`num_stages=N` 时编译器自动：①分配 N 份 tile 的 shared memory；②把 load 提前 N-1 轮发射（cp.async）；③插 wait 屏障。**你写数据流，它排流水**。

**实测的细腻处（比「双缓冲有用」更值钱）**：stages 1→2 只 +1%（131.9→133.5），**2→3 才跳 +20%（133.5→160.5）**。原因是 Ada 上 global→shared 的延迟 ≳ 一轮 tl.dot 的时长，2 级只藏了发射、藏不满整段延迟，3 级起气泡才填平。最优深度在 3/4 之间随形状摇摆（square4k 为 3、qwen8b 为 4，EXP-T02 §6），我不宣称唯一最优深度。「双缓冲」是流水思想的最小版，不是终点，深度要按延迟/计算比配。代价同样可测：stage 数 ∝ shared memory 占用（FA2 的 BN=128 配置就是这么 OOM 的，EXP-T01《Triton FA2 forward》）。

**另一半性能：grouped launch**（kernel 第 24-33 行）：把 CTA 按 GROUP_M 分组蛇形排，同组共享 B 块 → L2 命中率↑。它与流水线正交，一起构成「Triton GEMM 打平 cuBLAS（限两测形状 fp16）」的两条腿。

## 3. 本项目实证(EXP-T02,4090,fp16→fp32 累加;单轮存盘值)

| 形状 | stages=1 | 2 | 3 | 4 | cuBLAS |
|---|---|---|---|---|---|
| 4096³ | 131.9 | 133.5 | **160.5** | 157.1 | 159.8 TFLOPS |
| 2048×4096×12288(Qwen3-8B up_proj) | 127.6 | 129.4 | 151.0 | **154.4** | 147.3 |

square4k 最优 stages=3，与 cuBLAS 打平（差 0.4% 内）；qwen8b 最优 stages=4，反超 4.8%。数据：data/raw/EXP-T02/ew_gemm_bench.json（与 EXP-T02 §5-6 一致；cuBLAS=torch.matmul dispatch，cuBLASLt）。正确性：相对误差 ~7e-4（fp16 输入 fp32 累加 vs torch.matmul）。

## 4. 面试追问 Q&A

- **Q：cp.async 和普通 load 的区别？** 普通 load 走寄存器中转（global→reg→shared，占用发射槽和寄存器）；cp.async 由 DMA 引擎直写 shared，计算指令流不停。这是能「白嫖」重叠的硬件前提（SM80+）。
- **Q：双缓冲为什么在你的数据里几乎没用？** 不是没用，是不够深：重叠上限 = min（搬运时长，计算时长）。延迟长于一轮计算时，2 级只消一部分，加深到 3-4 级气泡才填满。反过来，shared memory 紧张的 kernel 只配得起 2 级，这是资源换深度。
- **Q：Hopper 上这套还成立吗？** 思想成立，机制换代：TMA（批量张量搬运）替 cp.async，wgmma 异步矩阵指令自带流水语义，DeepGEMM 即按此设计。sm_89 没有 TMA/wgmma，这就是「4090 走 mma、DeepGEMM Hopper-only」的硬件根源（阶段二 FP8 GEMM 的预习点）。
- **Q：怎么确认瓶颈真的是延迟没藏住？** NCU 的 long scoreboard stall 占比（本容器无计数器权限，方法论保留：Laptop 时代四 kernel 的 stall 归因手法同样适用于此）。

## 5. 延伸(锚点)

CUDA C++ Programming Guide §pipeline primitives;CUTLASS pipeline 文档； Triton compiler 的 software pipelining pass;DeepGEMM README(Hopper wgmma/TMA)；本仓 `src/gemm_pipelined.py`。
