# EXP-T02 · 流水线 GEMM:num_stages 扫描量化"双缓冲的贡献"

> **一句话结论**：`num_stages` 扫描量化了「双缓冲值多少」：2 级只给 +1%，**3 级才给 +20%**——4096³ 最优 160.5 TFLOPS 与 cuBLAS 打平（差 0.4%），Qwen3-8B up_proj 上反超 4.8%。

## 0. 元信息
| 日期 | 2026-08-23 | 环境 | 同 T01 | 状态 | 完成 |
|---|---|---|---|---|---|
关联：阶段一"GEMM 双缓冲升级"；llm-engine#EXP-D16《接入流水线（双缓冲）GEMM》前置；第 5 课中下实操。

## 1. 目的与假设
同一个 kernel 只变 num_stages（1=无重叠，2=双缓冲，3/4=深流水）。跑前假设：stages≥2 显著快于 1，且能逼近 cuBLAS（≥90%）。

## 2. 环境与配置
src/gemm_pipelined.py(grouped launch GROUP_M=8 + BM128/BN128/BK64/w8); fp16→fp32 累加；对照=torch.matmul（cublas 命名）。

## 3. 步骤
跑 scripts/test_ew_gemm.py 的 GEMM 段，扫两个形状 × stages{1..4} × cuBLAS。

## 4. 原始数据
data/raw/EXP-T02/ew_gemm_bench.json。

## 5. 结果(以存盘 raw 为准;首轮未存盘数字作废,见 §7)
4096³ 上 stages1→4 依次 131.9/133.5/**160.5**/157.1 TFLOPS，cuBLAS 159.8；最优是 stages=3，与 cuBLAS 打平（差 0.4%）。Qwen3-8B up_proj 上 127.6→**154.4**（stages4），cuBLAS 147.3，**反超 4.8%**。相对误差 ~1e-3。

## 6. 分析与结论
假设成立，**打平/反超 cuBLAS**。这里修正了一条认知：2 级双缓冲只给 +1%，**3 级才给 +20%**。Ada 上一次搬运的延迟长于一轮 dot，流水深度必须按延迟/计算比配（theory/02 §2）。所以「双缓冲的贡献」这个说法应改成「流水化的贡献（+21-22%），其中 2 级只占小头」。最优 stage 在 3/4 之间随形状与会话摇摆（square4k 为 3，qwen8b 为 4），我不宣称唯一最优深度。

## 7. 异常、偏差与开放问题
- **勘误（8/24 审计）**：本记录首版引用了第一轮未存盘的 bench 数字（162.1@s4 等）。它与存盘 raw（第二轮，加大尺寸点后重跑）漂移 ±3TFLOPS，最优 stage 也从 4 变成 3。现已全部以 raw 为准改写。教训：**结果只准引存盘轮**。
- raw 的 provenance sha=pre-commit，因为跑批在建仓首 commit 之前；代码版本以首 commit 274acb2 的 src/ 为准（kernel 未再改动，e0921c4 只加了 fp32 路径）。
- 未扫 BM/BN/BK 全空间。int8/fp8 变体留到阶段二。NCU stall 佐证缺席，因为容器无计数器权限（与 K01 同一限制）。

- backlog（2026-08-24 审计）：本记录与 README 引用的关键数字都是**单轮** bench，等 GPU 空闲再补 ≥3 轮 stability（mean/std 落 stability 文件）。

## 8. 下游影响
llm-engine#EXP-D16 已通过 src/gemm_pipelined.linear 接入。阶段二 FP8 GEMM 的 mma 路线预习点写进了 theory/02 的 Q&A（Hopper TMA/wgmma 界线）。

- **backlog 闭环（2026-08-24 晚）**：≥3 轮 stability 已补。square4k stages3 159.4±1.2 vs cublas 160.0±0.7，打平复现；qwen8b s4 反超复现。raw = data/raw/EXP-T02/*_stability_r{1,2,3}.json，聚合 = data/derived/exp-t02_stability_3rounds.csv。
