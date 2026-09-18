# 面试讲稿 · triton-kernels(现行唯一版,2026-08-24)

> 用法规则：每句量化主张先过 LEDGER.md 措辞红线表；括号内限定词是措辞的一部分，引用时不得剥离。全部数字=存盘 raw 值（注明"终端级"者除外；仍标**单轮**者即确为单轮口径，如 T03 binding 端到端）；≥3 轮 stability 已闭环（各 record §7 backlog 已清），headline 数字改引 3 轮 mean/std。

## 0. 三十秒开场

「我用 Triton 从零写了 FA2 forward、流水线 GEMM、FP8 per-block GEMM、 flash-decoding 和 MoE permute 一组 kernel，全部在 RTX 4090 上对标官方实现并落存盘 raw；更重要的是我能讲清每个差距的来源——设备侧、launch、融合三个口径分开算账。」

## 1. FA2(EXP-T01)——87% 必带四要素限定

讲法：「我的 **80 行简化版、仅 forward** 的 FA2，在 **4K 形状（B1·H32/8 GQA·D128）** 上达到 **对照 SDPA flash 后端** 的 **87%**（1.119 vs 0.979 ms，123 TFLOPS，单轮）。」
- 四要素（简化版 / 仅 forward / 该形状 / 对照=SDPA-flash）缺一不引（红线表）；严格值 87.45% 不进位。
- 慢的 13% 去向：tensor-core 布局微调、K/V 双缓冲 cp.async、warp 专业化——抽象税（theory/01 §4）。
- 追问 S=512:88%（该形状实值）；S=1024 只有 75%——不挑数字讲， 整表在 theory/01 §3。

## 2. 流水线 GEMM(EXP-T02)——"打平/反超"只限两测形状

讲法：「同一 kernel 只调 num_stages：4096³ 上 131.9→160.5 TFLOPS (stages=3)，与 cuBLAS(159.8)**打平（差 0.4% 内）**；Qwen3-8B up_proj 形状 154.4(stages=4)**反超 cuBLAS(147.3)4.8%**。限 fp16、这两个测过的形状，未做全形状扫描。」
- cuBLAS=torch.matmul dispatch(cuBLASLt)——对照物命名诚实（红线表）。
- 细腻处：2 级双缓冲仅 +1%，3 级才 +20%——流水深度按 延迟/计算比 配； 最优 stage 在 3/4 间随形状摇摆，不宣称唯一最优深度。
- 只引存盘 raw 轮（EXP-T02《流水线 GEMM》§7 勘误：首轮未存盘数字作废）。

## 3. Triton vs CUDA(EXP-T03)——永远分三口径

讲法：「这题不能裸答快慢，要分三层：①设备侧：8192² softmax 917 vs 922 GB/s，**同速**（双双贴 roofline 91%）；②launch：同一 1024² softmax Triton 37.6 vs torch 8.1 µs，而 8×8 纯开销 37.4≈37.6，证明时间不在 kernel 里；裸 CUDA 扩展量级 ~5.9µs（int8 v4，Kernel_Optimazation#EXP-K01《四 kernel 4090 重基准》，scale 预置口径）； ③端到端：Triton 单 kernel 融合 51.7µs 反超 "更快的 CUDA kernel + 3 次前置 launch" 的 65.1µs——融合数比单核快慢更重要。」
- int8 三数字（5.9 裸 / 65.1 ext / 51.7 融合）口径不得混引（红线表）； 融合数字跨会话波动 41.6~52µs，引用带区间（EXP-T03《三件套移植 + torch 绑定》§7）。
- 第四层（EXP-T05《CUDA Graph 消 launch 开销实测》）：CUDA Graph 把 launch 塌缩 **11.6×**（36.8→3.1µs /调用），graph 后 Triton 反超 torch eager——"Triton 小核慢"的正解是上 Graph，不是换 CUDA。

## 4. FP8 GEMM(EXP-T06)——1.5× 是预量化孤立 GEMM

讲法：「把 DeepGEMM 的 per-block 缩放代数搬到 sm_89 mma 路线： **预量化孤立 GEMM** 227.7/235.7 TFLOPS，vs fp16 cuBLAS(152.0/155.2) **1.5×**——这不是端到端推理提速；在线量化端到端只有 72.8/64.4 TF， 量化 kernel（torch 实现）是瓶颈，真实 serving 里权重预量化、激活量化融合进上游算子。」
- kernel 精确性 1.9e-4 与量化本体误差 ~3.7e-2（相对）两层分开报。
- 追问必答：DeepGEMM 为什么 Hopper-only——wgmma/TMA vs mma/cp.async 的指令世代界线（theory/06 §2 表）。

## 5. MoE permute(EXP-T07)——12.5× 的对照物是 torch 参考

讲法：「unpermute 用 **gather 式**（每 token 收 topk 行加权求和）替 scatter-add：无原子、求和顺序确定；vs **torch 参考实现（pytorch eager 四趟 kernel 路径）** 0.084 vs 1.054 ms，**12.5×**(bf16，T4096/D2048/ E60/topk4)。来源仍是融合——theory/03 第三层的又一实例。」
- 索引构建 0.27ms 反而是大头（> 两次搬运之和）——这就是 vLLM 给它写 moe_align_block_size 专用 kernel 的实证理由（专用化在 backlog）。

## 6. 收束句

「所有数字仓内 raw 可复算，措辞红线表逐条对着数据写——我能对每个数字说清它的口径、对照物和不成立的条件。」
