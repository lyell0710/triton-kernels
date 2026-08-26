# LEDGER — triton-kernels 对内状态账本

> **本文件是状态与措辞的唯一权威;README 为对外版,措辞以本表为准。**
> 台账/红线/勘误/待办只在这里维护;README 不再出现日期、状态列与内部流程词。

## 🧪 实验台账(状态唯一权威)

| 编号 | slug | 名称 | 日期 | 状态 | 关键数字(指针) |
|---|---|---|---|---|---|
| [EXP-T01](records/EXP-T01_fa2_forward.md) | fa2_forward | Triton FA2 forward(简化版):正确性 + 调优 + 对标 | 2026-08-23 | 完成 | 87% of SDPA@4K(data/raw/EXP-T01/) |
| [EXP-T02](records/EXP-T02_gemm_pipeline.md) | gemm_pipeline | 流水线 GEMM:num_stages 扫描量化"双缓冲的贡献" | 2026-08-23 | 完成 | 160.5 TFLOPS 打平 cuBLAS/8B 形状反超 4.8%(data/raw/EXP-T02/) |
| [EXP-T03](records/EXP-T03_ports_and_binding.md) | ports_and_binding | 三件套移植 + torch 绑定:launch 开销与融合的三层反转 | 2026-08-23 | 完成 | launch 三层反转(EXP-T02 json + EXP-T03/) |
| [EXP-T04](records/EXP-T04_flash_decoding.md) | flash_decoding | Flash-Decoding(split-K decode attention) | 2026-08-24 | 完成 | 32K 上下文 vs naive **2.39×**,GQA 原生;引擎 probe PASS |
| [EXP-T05](records/EXP-T05_cudagraph.md) | cudagraph | CUDA Graph 消 launch 开销实测(launch 三层结论的"解法"层) | 2026-08-24 | 完成 | launch 塌缩 **11.6×**(36.8→3.1µs/调用,data/raw/EXP-T05/) |
| [EXP-T06](records/EXP-T06_fp8_gemm.md) | fp8_gemm | FP8 GEMM(per-block scaling,sm_89 mma 路线) | 2026-08-24 | 完成 | per-block FP8 **227.7/235.7 TFLOPS = 1.5× fp16 cuBLAS**(data/raw/EXP-T06/) |
| [EXP-T07](records/EXP-T07_moe_permute.md) | moe_permute | MoE Permute/Unpermute(dispatch-combine 单卡版) | 2026-08-24 | 完成 | unpermute **12.5×** vs torch,gather 式无原子(data/raw/EXP-T07/) |
| [EXP-T08](records/EXP-T08_smem_stage_probe.md) | smem_stage_probe | num_stages 与 shared memory 份数的映射:编译期资源探针 | 2026-08-25 | 完成 | 缓冲份数 = **num_stages−1**(编译期 metadata.shared 实测,data/raw/EXP-T08/) |

> **stability(2026-08-24 晚)**:headline 数字已全部 ≥3 轮复测,mean/std 见 `data/derived/exp-t01_stability_3rounds.csv` 等五份(T01/02/05/06/07);T05 launch 塌缩勘误 11.8×→**11.6×**。
> 阶段二增量(8/24):FP8 per-block GEMM(theory/06)、MoE permute/unpermute(theory/07,对照 DeepEP)、flash-decoding(theory/05)、CUDA Graph(theory/03 第四层)、kperf 无计数器观测(theory/04);TP=2 引擎侧见 llm-engine#EXP-D22《TP=2 张量并行》。

## 📏 措辞红线表

**诚实度文化(本仓的差异化)**:每个进 README 的数字带 provenance(raw 首行/manifest 记录环境+命令+sha)且 ≥3 轮复测落 mean/std;假设在跑之前锁定判定阈值,证伪照登不删(T03 的 mask 假设证伪过程、"≤2×"假设错得有价值,全部保留);勘误留痕——T02 首轮未存盘数字作废、T05 11.8×→11.6× 修正,均在 record §7 与本台账可查,旧值废弃后禁止复活。

| 红线 | 当前 | 说明 |
|---|---|---|
| "打平/反超 cuBLAS" | ✅ 可用 | 限两测形状 fp16(square4k 打平 0.4% 内、8B up_proj 反超 4.8%);未全形状扫描;只引存盘 raw 轮(T02 §7 勘误);cuBLAS=torch.matmul dispatch(cuBLASLt) |
| FP8 "1.5×" | 限定 | **预量化孤立 GEMM vs fp16 cuBLAS**,非端到端推理提速;在线量化端到端另列(72.8TF) |
| "87% of SDPA-flash" | ✅ 可用 | 完整限定:**简化版、仅 forward、4K 形状(B1·H32/8·D128)、对照=SDPA flash 后端**;缺一不引 |
| "Triton 比 CUDA 慢/快" | 🚫 禁裸说 | 必须区分 设备侧(同速)/launch(慢 25µs)/端到端(看融合),T03 三口径 |
| int8 三数字 | 限定 | 5.9µs(裸,scale 预置)/65µs(ext)/52µs(triton 融合)口径不得混引 |
| 关键数字 stability | ✅ 已补 | 3 轮 mean/std 落 data/derived/(2026-08-24);headline 全部复现(T05 修正 11.6×) |
| flash-decoding 2.39× | 限定 | 单轮、kernel 级终端级证据(record §7 登记);README 对外统一写"单轮" |

## 📌 内部约定与待办

- **远程**:本仓当前**仅本地**(无 git remote);推 GitHub 由用户建 repo 后 `git remote add origin ... && git push`(待用户动作)。
- bench 只追加 UTC 前缀新文件,不覆盖已有 raw;`data/raw/EXP-*/manifest.txt` 记录 sha256+provenance。
- 图表全部由 `scripts/plot_readme_figures.py` 从 derived 数据生成;对外图脚注不带日期(源数据文件+硬件+轮数)。
- docs/talk/ 讲稿逐句过本表红线;引用限定词不得剥离。
- 阶段三剩余:读 DeepEP(纯阅读)与真实 PR(用户动作)。

## 2026-08-25 增补

- EXP-T04 stability backlog 闭环:3 轮 + 三臂口径拆分(2.24×/5.17×);§5 旧表小 Skv 行作废(混合口径不可复现);README/简历已同步双口径。变体(H32/fp16)入 raw 备查。
- EXP-T03 残留:设备侧 8192² 同速已被 T02 3 轮覆盖(0.5824/0.5844 ms);**binding 端到端(51.7/65.1µs)仍为单轮**,对外引用须带"单轮"。
- 注释校验勘正:flash_decode.py §8→§7 指针、elementwise/README 922→921 GB/s(以 record 917/921 为准)。
