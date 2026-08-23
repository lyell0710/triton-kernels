# triton-kernels — Triton 侧算子:FA2 / 流水线 GEMM / 三件套移植

阶段一算子计划的 Triton 主战场;llm-engine 的 D15/D16 以本仓为依赖。
CUDA 侧四 kernel 见 Kernel_Optimazation(4090 重测 = 其 EXP-K01)。

## 主结果(RTX 4090,全部 raw 可复算)

| 项 | 数字 | 出处 |
|---|---|---|
| **FA2 forward(80 行简化版)** | SDPA-flash 的 **88%**(4K:1.119 vs 0.979ms,123 TFLOPS);6 形状正确性全过(GQA/非整除) | EXP-T01 |
| **流水线 GEMM** | stages 1→4:133→**162 TFLOPS,追平 cuBLAS(160)**;细腻处:双缓冲(2级)仅 +1%,3 级才 +19% | EXP-T02 |
| **Triton vs CUDA 真相** | 设备侧同速(8192² softmax 917 vs 922 GB/s);小核差距=launch 开销(37 vs 8 vs 5.9µs);端到端融合(1 launch)反超快核+多 launch | EXP-T03 |
| **torch 绑定** | CUDA v4 quantize 零改动绑进 extension,1M 元素逐位一致 | EXP-T03 |

## EXP 索引

| 编号 | slug | 日期 | 状态 | 关键数字(指针) |
|---|---|---|---|---|
| [EXP-T01](records/EXP-T01_fa2_forward.md) | fa2_forward | 2026-08-23 | 完成 | 88% of SDPA@4K(data/raw/EXP-T01/) |
| [EXP-T02](records/EXP-T02_gemm_pipeline.md) | gemm_pipeline | 2026-08-23 | 完成 | 162 TFLOPS 追平 cuBLAS(data/raw/EXP-T02/) |
| [EXP-T03](records/EXP-T03_ports_and_binding.md) | ports_and_binding | 2026-08-23 | 完成 | launch 三层反转(EXP-T02 json + EXP-T03/) |

## 措辞红线表

| 红线 | 当前 | 说明 |
|---|---|---|
| "追平 cuBLAS" | ✅ 可用 | 限两测形状 fp16;未全形状扫描(T02 §7) |
| "88% of FA2" | ✅ 可用 | 对照=SDPA flash 后端;须带"简化版/仅 forward" |
| "Triton 比 CUDA 慢/快" | 🚫 禁裸说 | 必须区分 设备侧(同速)/launch(慢 25µs)/端到端(看融合),T03 三口径 |
| int8 三数字 | 限定 | 5.9µs(裸,scale 预置)/65µs(ext)/52µs(triton 融合)口径不得混引 |

## 结构

src/{fa2_fwd,gemm_pipelined,elementwise_kernels}.py + torch_ext/;
scripts/{test_fa2,test_ew_gemm}.py;docs/theory/01-03(五节全实证);
records/T01-03;data/raw/。
