---
topic: FP8 GEMM(per-block scaling)与 Ada/Hopper 界线
status: 完成(实证=EXP-T06)
---

# 06 · FP8 GEMM:缩放代数与指令世代

## 1. 一句话结论
把 DeepGEMM 的缩放策略(权重 128×128 块 scale + 激活 per-token-group
scale)搬到 sm_89 的 mma 路线:fp8 e4m3 tl.dot + BLOCK_K=128 与缩放组
硬对齐,反量化融合进累加——**227.7/235.7 TFLOPS,比 fp16 cuBLAS 快
1.5×**;kernel 精确性 1.9e-4,fp8 量化本体误差 ~3.7e-2(相对)。

## 2. 机制
- **为什么要细粒度 scale**:fp8 e4m3 动态范围仅 ±448、尾数 3 位;
  per-tensor scale 会被 outlier 撑爆。块级(W)+ token 组级(A)把
  量化域缩小到局部,误差 ~1e-2 量级可控。
- **融合的关键**:BLOCK_K=128=缩放组 → 每个 K 组内 scale 恒定,
  组内 fp8 dot 完毕再乘 sa(行向量)×sb(标量,BLOCK_N=128 与权重块
  对齐的前提),partial 并入 fp32 acc——**从不物化 fp16 权重**。
- **Ada vs Hopper(面试题眼)**:同一缩放代数,两代指令:
  | | sm_89(本实现) | sm_90(DeepGEMM 本体) |
  |---|---|---|
  | 矩阵指令 | mma.sync(同步) | wgmma(异步,warpgroup) |
  | 搬运 | cp.async | TMA(张量批搬运) |
  | 细粒度缩放 | 累加器侧手乘 | wgmma 原生 scale 槽/CUDA core 累加 |
  DeepGEMM Hopper-only 的原因就是后两列;讲清这条界线=讲清"为什么
  不能拿 DeepGEMM 直接跑 4090"。与 vllm/experiments#EXP-016 的
  oracle/fp8.py capability 分派(90/100 快路径跳过 89)同一事实的两面。

## 3. 本项目实证(EXP-T06)
| 形状 | fp8 prequant | fp16 cuBLAS | 加速 | kernel 精确性 | 量化误差 |
|---|---|---|---|---|---|
| 4096³ | **227.7 TF** | 152.0 | 1.50× | 1.9e-4 | 3.6e-2 |
| 8B up_proj | **235.7 TF** | 155.2 | 1.52× | 1.9e-4 | 3.9e-2 |
在线量化端到端 72.8 TF——量化 kernel(torch 实现)是瓶颈;真实 serving
中权重预量化、激活量化融合进上游算子(如 RMSNorm epilogue),此列仅示成本。

## 4. 面试追问 Q&A
- **Q: fp8 理论 2× 只吃到 1.5×,差哪?** 缩放乘法+fp32 累加占了算力、
  fp8 mma 峰值本身打折(与 fp16 同引擎不同吞吐配比);NCU 不可用,
  以 kperf 卡片定界(compute-bound,~70% fp8 峰值)。
- **Q: e4m3 vs e5m2 怎么选?** 前向权重/激活用 e4m3(要精度);
  梯度用 e5m2(要范围)。本实现推理侧,全 e4m3。

## 5. 延伸(锚点)

DeepGEMM README 与 fp8 GEMM 核心实现(128×128 块 scale 设计、
wgmma/TMA 依赖——Hopper-only 的出处);PTX ISA mma 章 fp8 段
(sm_89 e4m3 吞吐配比);vllm/experiments#EXP-016(oracle/fp8.py 的
capability 分派:90/100 快路径跳过 89);本仓 `src/fp8_gemm.py` 与
`scripts/test_fp8_gemm.py`。
