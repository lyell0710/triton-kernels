# EXP-T02 · 流水线 GEMM:num_stages 扫描量化"双缓冲的贡献"

## 0. 元信息
| 日期 | 2026-08-23 | 环境 | 同 T01 | 状态 | 完成 |
|---|---|---|---|---|---|
关联:阶段一"GEMM 双缓冲升级";llm-engine#EXP-D16 前置;第5课中下实操。

## 1. 目的与假设
同一 kernel 只变 num_stages(1=无重叠,2=双缓冲,3/4=深流水),假设:
stages≥2 显著快于 1,且可逼近 cuBLAS(≥90%)。

## 2. 环境与配置
src/gemm_pipelined.py(grouped launch GROUP_M=8 + BM128/BN128/BK64/w8);
fp16→fp32 累加;对照=torch.matmul(cublas 命名)。

## 3. 步骤
scripts/test_ew_gemm.py 的 GEMM 段:两形状 × stages{1..4} × cuBLAS。

## 4. 原始数据
data/raw/EXP-T02/ew_gemm_bench.json。

## 5. 结果
4096³:133.4/134.8/160.7/**162.1** TFLOPS(stages1→4)vs cuBLAS 160.5;
Qwen3-8B up_proj 形状:127→**157.1** vs 155.7。相对误差 ~1e-3。

## 6. 分析与结论
假设成立且超出(**追平/反超 cuBLAS**);修正认知:2 级双缓冲仅 +1%,
**3 级才 +19%**——Ada 上搬运延迟长于一轮 dot,深度须按延迟/计算比配
(theory/02 §2)。"双缓冲的贡献"应表述为"流水化的贡献(+21%),其中
2 级只占小头"。

## 7. 异常、偏差与开放问题
未扫 BM/BN/BK 全空间(固定一组 tile);int8/fp8 变体留阶段二;
NCU stall 佐证因容器计数器无权限缺席(与 K01 同限制)。

## 8. 下游影响
llm-engine#EXP-D16 以 src/gemm_pipelined.linear 接入;阶段二 FP8 GEMM
的 mma 路线预习点已写入 theory/02 Q&A(Hopper TMA/wgmma 界线)。
