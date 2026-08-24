# EXP-T04 · Flash-Decoding(split-K decode attention)

## 0. 元信息
| 日期 | 2026-08-24 | 环境 | v0.25.1-venv, RTX 4090 | 状态 | 完成 |
|---|---|---|---|---|---|
关联:EXP-T01 §7 backlog;llm-engine decode 路径接入。

## 1. 目的与假设
Sq=1 时并行度改从 KV 维取(split-K 两段式:分段部分统计量 + online 归并)。
假设:长 KV 显著快于 naive;正确性 = naive 参考 bf16 噪声级。

## 2. 环境与配置
src/flash_decode.py(partial + combine 两 kernel;GQA 原生,不 repeat kv;
splits 数按填满 SM 启发式,next_pow2)。对照 naive_attention(llm-engine)。

## 3. 步骤
4 档 Skv 正确性+速度 → 引擎接线(fa2_attention 的 Sq==1 分支)→
fp32 probe → d14 bench。

## 4. 原始数据
kernel 级对照为终端级证据(数表见 §5);引擎级:
llm-engine data/raw/EXP-D11/20260824T02*_probe.json、EXP-D14/20260824T020513。

## 5. 结果
kernel 级(B1 H16/8 D128, bf16, vs naive[repeat kv]):
| Skv | err | flash_decode | naive | 提速 |
|---|---|---|---|---|
| 512 | 2.0e-3 | 0.094 ms | 0.114 | 1.21x |
| 2048 | 9.8e-4 | 0.092 | 0.117 | 1.27x |
| 8192 | 4.9e-4 | 0.091 | 0.117 | 1.28x |
| 32768 | 2.6e-4 | 0.147 | 0.352 | **2.39x** |
引擎级(0.6B, prompt512):fp32 probe PASS(5.96e-5);TPOT 28.99 与
decode-naive 持平——512 上下文时 attention 占 decode 时间 <10%,持平合理;
收益随上下文长度增长(kernel 表)。

## 6. 分析与结论
- 短 Skv 段 fd 时间平坦(~0.09ms)= 两次 Triton launch 地板(T03 结论
  再现);长 Skv 带宽差距显形(naive 物化 scores + 读 repeat 后的 kv)。
- **接线教训(先犯后改)**:首版对 k/v cache 切片 .contiguous() → 每层每步
  整拷 KV cache,TPOT 反而 +0.5ms;kernel 本吃任意 stride,去掉即恢复。
  "接入胶水的隐藏拷贝"是 kernel 快引擎不快的常见根因。
- GQA 原生(读未 repeat 的 kv)是 kernel 级另一半收益,引擎接线因
  attention_impl 契约传入已 repeat 的 kv 而未兑现(D15 §7 已注)。

## 7. 异常、偏差与开放问题
kernel 级 bench 未存 raw(终端级);splits 启发式未扫参;引擎长上下文
(8K/32K prompt)bench 未跑(d14 PROMPT_LEN 固定 512,参数化留待)。

## 8. 下游影响
llm-engine decode 路径升级(LLME_ATTN=fa2 现覆盖 prefill+decode);
theory/05;vLLM flash-decoding/paged attention 的面试衔接点。
