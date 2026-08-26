# EXP-T02 · 流水线 GEMM:num_stages 扫描量化"双缓冲的贡献"

> **一句话结论**：`num_stages` 扫描量化了「双缓冲值多少」：2 级只给 +1%，**3 级才给 +20%**——4096³ 最优 160.5 TFLOPS 与 cuBLAS 打平（差 0.4%），Qwen3-8B up_proj 上反超 4.8%。

## 0. 元信息
| 日期 | 2026-08-23 | 环境 | 同 T01 | 状态 | 完成 |
|---|---|---|---|---|---|
关联：阶段一"GEMM 双缓冲升级"；llm-engine#EXP-D16《接入流水线（双缓冲）GEMM》前置；第 5 课中下实操。

## 1. 目的与假设
同一 kernel 只变 num_stages（1=无重叠，2=双缓冲，3/4=深流水），假设： stages≥2 显著快于 1，且可逼近 cuBLAS(≥90%)。

## 2. 环境与配置
src/gemm_pipelined.py(grouped launch GROUP_M=8 + BM128/BN128/BK64/w8); fp16→fp32 累加；对照=torch.matmul（cublas 命名）。

## 3. 步骤
scripts/test_ew_gemm.py 的 GEMM 段：两形状 × stages{1..4} × cuBLAS。

## 4. 原始数据
data/raw/EXP-T02/ew_gemm_bench.json。

## 5. 结果(以存盘 raw 为准;首轮未存盘数字作废,见 §7)
4096³:131.9/133.5/**160.5**/157.1 TFLOPS(stages1→4)vs cuBLAS 159.8——最优 stages=3，与 cuBLAS 打平（差 0.4%）; Qwen3-8B up_proj:127.6→**154.4**(stages4)vs cuBLAS 147.3（**反超 4.8%**）。相对误差 ~1e-3。

## 6. 分析与结论
假设成立（**打平/反超 cuBLAS**）；修正认知：2 级双缓冲仅 +1%， **3 级才 +20%**——Ada 上搬运延迟长于一轮 dot，深度须按延迟/计算比配（theory/02 §2）。"双缓冲的贡献"应表述为"流水化的贡献（+21-22%），其中 2 级只占小头"。最优 stage 在 3/4 间随形状/会话摇摆（square4k 为 3， qwen8b 为 4），不宣称唯一最优深度。

## 7. 异常、偏差与开放问题
- **勘误（8/24 审计）**：本记录首版引用了第一轮（未存盘）bench 数字（162.1@s4 等），与存盘 raw（第二轮，加大尺寸点后重跑）漂移 ±3TFLOPS、最优 stage 由 4 变 3——已全部以 raw 为准改写；教训：**结果只准引存盘轮**。
- raw 的 provenance sha=pre-commit（建仓首 commit 前跑批），代码版本以首 commit 274acb2 的 src/ 为准（kernel 未再改动，e0921c4 仅加 fp32 路径）。
- 未扫 BM/BN/BK 全空间；int8/fp8 变体留阶段二；NCU stall 佐证因容器计数器无权限缺席（与 K01 同限制）。

- backlog（2026-08-24 审计）：本记录/README 引用的关键数字为**单轮** bench，待 GPU 空闲补 ≥3 轮 stability（mean/std 落 stability 文件）。

## 8. 下游影响
llm-engine#EXP-D16 以 src/gemm_pipelined.linear 接入；阶段二 FP8 GEMM 的 mma 路线预习点已写入 theory/02 Q&A（Hopper TMA/wgmma 界线）。

- **backlog 闭环（2026-08-24 晚）**:≥3 轮 stability 已补——square4k stages3 159.4±1.2 vs cublas 160.0±0.7（打平复现）;qwen8b s4 反超复现。 raw = data/raw/EXP-T02/*_stability_r{1,2,3}.json，聚合 = data/derived/exp-t02_stability_3rounds.csv。
