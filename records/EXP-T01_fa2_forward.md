# EXP-T01 · Triton FA2 forward(简化版):正确性 + 调优 + 对标

## 0. 元信息
| 日期 | 2026-08-23 | 环境 | v0.25.1-venv(triton3.6/torch2.11), RTX 4090 | 状态 | 完成 |
|---|---|---|---|---|---|
关联：阶段一"Flash Attention Triton 简化版"；llm-engine#EXP-D15《接入自研 Triton FA2》前置。

## 1. 目的与假设
从零写 FA2 forward(causal+GQA)，假设（跑前锁）：6 形状 max abs err <2e-2 且 ≥SDPA-flash 60% 效率。

## 2. 环境与配置
src/fa2_fwd.py（~80 行 kernel）;fp16 输入 fp32 累加；参考=fp32 精算（repeat_interleave 展开 GQA）;bench 100 iters+warmup 20。

## 3. 步骤
scripts/test_fa2.py（6 正确性形状 + 4 序列长 bench）→ tile 扫描 7 配置 → 优配回填默认 → 复跑落 raw。

## 4. 原始数据
data/raw/EXP-T01/fa2_bench.json（provenance 首字段）；tile 扫描为终端级证据（数字记录于 theory/01 §2 第 4 步）。

## 5. 结果
正确性 6/6（err ≤2e-3，含 GQA 4:1 与 S=777 非整除）。 S=512/1K/2K/4K:0.038/0.122/0.330/1.119 ms = SDPA-flash 的 88/75/86/**87**%（4K 严格值 87.45%，不进位）; vs naive fp32 快 8~20×。优配 BM128/BN64/w8/s2（4K 上比 BM64 +17%）; BN=128 撞 shared memory 上限（160KB）OOM。

## 6. 分析与结论
两条假设均成立。调优两课：M 块增大→K/V 复用与簿记摊薄；tile 受片上资源硬约束。与官方差 12% = tensor-core 布局/双缓冲/warp 专业化的抽象税（theory/01 §4）。

## 7. 异常、偏差与开放问题
仅 forward；decode(S_q=1)不适用本 kernel（需 flash-decoding split-K， backlog）；tile 扫描未存 raw（终端级）；raw provenance sha=pre-commit（建仓首 commit 前跑批，代码=首 commit 274acb2 版）。

- backlog（2026-08-24 审计）：本记录/README 引用的关键数字为**单轮** bench，待 GPU 空闲补 ≥3 轮 stability（mean/std 落 stability 文件）。

## 8. 下游影响
llm-engine#EXP-D15 接入（attention_impl 指针处）；简历句候选见 README。

- **backlog 闭环（2026-08-24 晚）**:≥3 轮 stability 已补——87.2%(1.1184±0.0015 vs sdpa 0.9749±0.0024,S=4096)——87% 口径维持。 raw = data/raw/EXP-T01/*_stability_r{1,2,3}.json，聚合 = data/derived/exp-t01_stability_3rounds.csv。
