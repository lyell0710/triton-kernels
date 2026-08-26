---
topic: FlashAttention-2 forward
status: 完成(实证=EXP-T01《Triton FA2 forward》)
---

# 01 · FlashAttention:从"为什么慢"推到"为什么快"

## 1. 一句话结论

标准 attention 慢在**把 S×S 矩阵写去 HBM 再读回来**；FA 用 online softmax 把整行 softmax 改写成"分块流式 + 两个跑动统计量随时修正"，S² 中间量永不落显存——HBM 流量从 O(S²) 降到 O(S·D)，长序列下把 memory-bound 拉回 compute-bound。本仓 80 行 Triton 简化版实测达官方 SDPA-flash 的 87%（4K，严格值 87.45% 不进位；EXP-T01）。

## 2. 机制(一步一步)

**第 0 步 · 标准算法的账**：O = softmax(QKᵀ/√d)V。S=4096， H=32 时 S² 矩阵 fp32 = 2GB 级读写 ×2（写 S、读 S 做 softmax、再读做 PV）。 4090 带宽 1008GB/s → 光搬这个矩阵就是毫秒级，而 tensor core 算力闲着。 naive 实测 6.7ms@2K vs 我们 0.33ms——20 倍就差在这。

**第 1 步 · softmax 为什么"看似不能分块"**：softmax(x)_i = e^{x_i}/Σe^{x_j} 需要**整行**的 max（数值稳定）和 sum。K 分块后每块只算出行的一段， max/sum 都不完整。

**第 2 步 · online softmax（核心恒等式）**：维护跑动 max m 和跑动和 l。新块进来，新 max m' = max(m， max（块）)；旧的部分和用系数 α = e^{m−m'} 一乘就换到新基准：l' = α·l + Σe^{x_new−m'}。关键：**输出累加器 acc 同样可以被 α 修正**(acc' = α·acc + P_new·V_new)， 所以连"最后再除 l"都能推迟到全部块流完——一遍循环出最终结果。（本仓 kernel 第 69-75 行就是这五行数学。）

**第 3 步 · FA1→FA2 改了什么**（面试高频）：
- 循环反转：FA1 外层 K/V 内层 Q（每个 K 块要读写所有 Q 块的 m/l/acc 到 HBM）；FA2 外层 Q 内层 K/V——**m/l/acc 常驻寄存器**，零中间落存。
- 非 matmul FLOPs 削减：rescale 从"每块除 l"改为"最后一次除"。
- 并行度：seq 维也切块进 grid(本仓 grid = (S/BLOCK_M， B·H))。

**第 4 步 · 本仓实测的调优课**(EXP-T01)：
- BLOCK_M 64→128 + warps 4→8:4K 序列 1.362→1.126ms(+17%)。机理：M 块越大，每次载入的 K/V 被更多 Q 行复用（算术强度↑）， softmax 簿记摊薄；warps 增多喂饱更大的 tile。
- BLOCK_N=128 直接 OOM shared memory（需求 160KB > Ada 上限）—— tile 不是越大越好，片上资源是硬约束。
- causal 提前终止：行块只扫到对角(hi = (pid_m+1)·BLOCK_M)， FLOPs 直接减半——mask 不只是"填 -inf"，更是"根本不算"。

## 3. 本项目实证(EXP-T01,4090,fp16,B1·H32/8(GQA 4:1)·D128)

| S | 本仓 Triton | SDPA-flash | naive fp32 | 效率 vs flash |
|---|---|---|---|---|
| 512 | 0.038 ms | 0.033 | 0.319 | 88% |
| 1024 | 0.122 | 0.092 | 1.723 | 75% |
| 2048 | 0.330(104 TFLOPS) | 0.282(122) | 6.698 | 86% |
| 4096 | 1.119(123 TFLOPS) | 0.979(140) | — | **87%** |

数据：data/raw/EXP-T01/fa2_bench.json（单轮。8/24 勘误：4K 效率 88%→**87%**，严格值 87.45% 不进位；S=512 的 88% 为该形状实值，保留）。

正确性：6 形状（MHA/GQA 2:1/4:1、非整除 seq 777、双 head_dim） max abs err ≤2e-3 全过（fp32 精算参考）。

## 4. 面试追问 Q&A

- **Q： l 为什么可以最后才除？** 除法对 V 的线性组合是标量缩放， 与累加交换；数值上 fp32 累加器扛得住（err 实测 2e-3 量级）。
- **Q： GQA 在 kernel 里改了什么？** 一行：hkv = hq // GQA_GROUP——寻址映射，不改算法；收益在 KV 读取量（H_kv < H_q）。
- **Q： 为什么你的版本比官方慢 13%？** 官方 CUDA 版有 tensor-core 布局微调、K/V 双缓冲 cp.async、块内 warp 专业化；Triton 把这些交给编译器， 换 80 行可读实现——这 13% 就是抽象税的定价（本仓观点：可讲清楚的 87% 好过讲不清楚的 100%）。
- **Q： decode(S_q=1)也用这个 kernel 吗？** 不该：M 维只有 1，tile 全浪费——decode 用 split-K 型 flash-decoding（沿 KV 维并行再规约）， 这是本仓 backlog（与 vLLM paged attention 的衔接点）。
- **Q： backward 为什么难得多？** 需要重算 P（存不下），且 dQ/dK/dV 三路归约方向不同（dQ 行向、dK/dV 列向）→ 原子加或二次分块；本仓只做 forward 并如实声明。

## 5. 延伸(锚点)

论文 FlashAttention(2205.14135)§3.1 online softmax；FA2(2307.08691) §3 两处循环反转与非 matmul FLOPs；Triton 官方 fused-attention tutorial（本仓实现独立手写，事后与其对照校核结构一致）；vLLM paged attention = 本 kernel + block table 间接寻址（见 vllm/experiments 白板图 2）。
