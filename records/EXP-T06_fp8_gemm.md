# EXP-T06 · FP8 GEMM(per-block scaling,sm_89 mma 路线)

> **一句话结论**：sm_89 上的 FP8 GEMM 做到 227.7 TFLOPS，相对 fp16 cuBLAS **1.50×**——fp8 理论的 2× 只兑现四分之三，缺口在缩放乘法与 fp32 累加占算力、以及 fp8 mma 配比打折。

## 0. 元信息
| 日期 | 2026-08-24 | 环境 | v0.25.1-venv(triton3.6), RTX 4090 | 状态 | 完成 |
|---|---|---|---|---|---|
关联：阶段二"FP8 GEMM(per-block scaling)，对标 DeepGEMM"。

## 1. 目的与假设
DeepGEMM 缩放策略（W 128×128 块 scale + A per-token-group）在 Ada mma 路线落地。假设（跑前锁）：fp8 vs fp16 cuBLAS ≥1.3×；kernel 精确性（vs 反量化参考）≤1e-3 相对。

## 2. 环境与配置
src/fp8_gemm.py：e4m3（满量程 448），BLOCK_K=128=缩放组、BLOCK_N=128= 权重块（sb 取标量的前提）；反量化融合进 fp32 累加，不物化 fp16 权重。正确性双层：①kernel vs 逐块反量化 fp32 精算；②量化保真 vs 原 fp16。

## 3. 步骤
scripts/test_fp8_gemm.py：两形状 × {fp8 prequant / fp8 在线量化端到端 / fp16 triton / fp16 cuBLAS}。

## 4. 原始数据
data/raw/EXP-T06/fp8_gemm_bench.json（provenance 首字段）。

## 5. 结果
| 形状 | fp8 prequant | fp16 cuBLAS | 加速 | kernel 精确性 | 量化误差（vs fp16） |
|---|---|---|---|---|---|
| 4096³ | **227.7 TFLOPS** | 152.0 | **1.50×** | 1.9e-4 | 3.6e-2 rel |
| 8B up_proj | **235.7** | 155.2 | **1.52×** | 1.9e-4 | 3.9e-2 rel |
在线量化端到端 72.8/64.4 TFLOPS（torch 侧量化 kernel 是瓶颈，仅示成本； 真实 serving 权重预量化、激活量化融合上游）。

## 6. 分析与结论
两假设成立。fp8 理论 2× 只兑现 1.5×：缩放乘法+fp32 累加占算力、fp8 mma 配比打折（kperf 定界 compute-bound ~70% fp8 峰值；NCU 不可用）。 Ada/Hopper 界线（mma+cp.async vs wgmma+TMA）= theory/06 表格， 与 vllm/experiments#EXP-016《D4 FP8 vs W4A16 同卡对比》的 capability 分派互为表里。

## 7. 异常、偏差与开放问题
tile 未扫全空间；激活量化融合（RMSNorm epilogue）未做；e5m2 路线未测； BLOCK_N 硬绑 128（权重块对齐）——解耦需 sb 向量化，backlog。

- backlog（2026-08-24 审计）：本记录/README 引用的关键数字为**单轮** bench，待 GPU 空闲补 ≥3 轮 stability（mean/std 落 stability 文件）。
- kperf 三卡观测数字（softmax 8192²：带宽 91%/occ 67% regs 限；FA2 S=4K：算力 74%/occ 17%，regs 213；GEMM 4096³：算力 98%/occ 17%，regs 170）为**终端级证据**（kperf.py 终端输出，未存 raw）；theory/04 §3 引用以本条为锚。

## 8. 下游影响
阶段二清单 FP8 GEMM 项闭环；面试句："同一缩放代数在 Ada 用 mma 落地 1.5×，并能讲清 DeepGEMM 为什么 Hopper-only"。

- **backlog 闭环（2026-08-24 晚）**:≥3 轮 stability 已补——prequant 228.1±1.3 TFLOPS（227.7 单轮口径成立）。 raw = data/raw/EXP-T06/*_stability_r{1,2,3}.json，聚合 = data/derived/exp-t06_stability_3rounds.csv。
