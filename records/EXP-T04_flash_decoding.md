# EXP-T04 · Flash-Decoding(split-K decode attention)

> **一句话结论**：split-K decode attention 在长 Skv 上跑出提速，短 Skv 段被两次 Triton launch 的地板压平；本实验最贵的一课是接线：对 KV cache 切片做 `.contiguous()` 会每层每步整拷，TPOT 反而 +0.5ms。

## 0. 元信息
| 日期 | 2026-08-24 | 环境 | v0.25.1-venv， RTX 4090 | 状态 | 完成 |
|---|---|---|---|---|---|
关联：EXP-T01《Triton FA2 forward》§7 backlog;llm-engine decode 路径接入。

## 1. 目的与假设
Sq=1 时并行度改从 KV 维取，用 split-K 两段式：先分段算部分统计量，再 online 归并。跑前假设：长 KV 显著快于 naive，正确性落在 naive 参考的 bf16 噪声级。

## 2. 环境与配置
src/flash_decode.py（partial + combine 两 kernel;GQA 原生，不 repeat kv; splits 数按填满 SM 启发式，next_pow2）。对照 naive_attention(llm-engine)。

## 3. 步骤
先测 4 档 Skv 的正确性与速度，再接到引擎（fa2_attention 的 Sq==1 分支），然后跑 fp32 probe 与 d14 bench。

## 4. 原始数据
kernel 级对照为终端级证据（数表见 §5）；引擎级： llm-engine data/raw/EXP-D11/20260824T02*_probe.json、EXP-D14/20260824T020513。

## 5. 结果
kernel 级（B1 H16/8 D128, bf16, vs naive[repeat kv]）:
| Skv | err | flash_decode | naive | 提速 |
|---|---|---|---|---|
| 512 | 2.0e-3 | 0.094 ms | 0.114 | 1.21x |
| 2048 | 9.8e-4 | 0.092 | 0.117 | 1.27x |
| 8192 | 4.9e-4 | 0.091 | 0.117 | 1.28x |
| 32768 | 2.6e-4 | 0.147 | 0.352 | **2.39x** |
引擎级（0.6B，prompt512）：fp32 probe PASS（5.96e-5）；TPOT 28.99 与 decode-naive 持平。512 上下文时 attention 只占 decode 时间 <10%，持平合理；收益随上下文长度增长（见 kernel 表）。

## 6. 分析与结论
- 短 Skv 段 fd 时间平坦（~0.09ms），这是两次 Triton launch 的地板，T03 的结论在这里再现。长 Skv 上带宽差距才显形，因为 naive 要物化 scores 并读 repeat 后的 kv。
- **接线教训（先犯后改）**：首版对 k/v cache 切片做了 .contiguous()，于是每层每步整拷一遍 KV cache，TPOT 反而 +0.5ms。kernel 本就能吃任意 stride，去掉即恢复。kernel 快而引擎不快，常见根因就是接入胶水里的隐藏拷贝。
- GQA 原生（读未 repeat 的 kv）是 kernel 级的另一半收益。引擎接线没兑现它，因为 attention_impl 契约传入的是已 repeat 的 kv（D15 §7 已注）。

## 7. 异常、偏差与开放问题
kernel 级 bench 未存 raw，属终端级证据。splits 启发式未扫参。引擎侧的长上下文（8K/32K prompt）bench 未跑，因为 d14 的 PROMPT_LEN 固定 512，参数化留待后续。

- backlog（2026-08-24 审计）：本记录与 README 引用的关键数字都是**单轮** bench，等 GPU 空闲再补 ≥3 轮 stability（mean/std 落 stability 文件）。

## 8. 下游影响
llm-engine 的 decode 路径已升级，LLME_ATTN=fa2 现在覆盖 prefill+decode。theory/05 与 vLLM flash-decoding/paged attention 是面试衔接点。

- **backlog 闭环 + 口径勘正（补测，scripts/test_flash_decode.py，3 轮）**：①fd 侧完全复现，32K 为 0.147→0.151±0.005 ms。②把对照物口径拆成三臂后，32K 提速 = **2.24±0.11×（naive repeat 预置）/ 5.17±0.24×（含 repeat 实体化，GQA 原生免掉的正是这份拷贝）**。③**§5 原表 Skv≤8192 各行作废**——三轮实测的 repeat 预置口径下，fd 反而只有 0.86-0.88×，原因是 split-K 归并开销在短上下文不划算；原单轮 1.21-1.28× 属当年未存脚本的混合口径，不可复现。所以 flash-decoding 的正确定位是长上下文武器。④另附 Qwen3-8B 形状变体（H32/fp16）：32K 达 9.0×（repeat 内计）。raw=data/raw/EXP-T04/(manifest)，聚合=data/derived/exp-t04_stability_3rounds.csv。原「未存 raw/未扫参」的缺口就此关闭，splits 启发式扫参仍开放。
