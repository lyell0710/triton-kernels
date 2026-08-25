# triton-kernels — Triton 手写 LLM 算子:FA2 / 流水线 GEMM / FP8 / flash-decoding / MoE / CUDA Graph

从零手写并系统 benchmark 一组 LLM 推理核心算子(RTX 4090),证明两件事:
**①简化实现能逼近/打平生产级库(SDPA-flash、cuBLAS);②每个数字可追溯到落盘 raw、≥3 轮复测、限定口径如实声明。**
下游消费方:[llm-engine](https://github.com/lyell0710/llm-engine)(D15/D16/decode 路径以本仓为依赖)。

## 🎯 Headline 结果(RTX 4090,3 轮 mean±std,除注明)

| 结果 | 数字 | 限定口径(红线表管辖) | 证据 |
|---|---|---|---|
| **FA2 forward,80 行简化版** | **SDPA-flash 的 87%**(S=4K:1.118±0.002 vs 0.975±0.002 ms,123 TFLOPS) | 简化版、仅 forward、4K 形状(B1·H32/8·D128)、对照=SDPA flash 后端 | EXP-T01 · `data/derived/exp-t01_stability_3rounds.csv` |
| **流水线 GEMM 打平 cuBLAS** | 4096³ fp16:**160.5 TFLOPS vs 159.8**(stages=3;3 轮复现 159.4±1.2 vs 160.0±0.7);8B up_proj **反超 4.8%** | 限两测形状 fp16;cuBLAS=torch.matmul dispatch(cuBLASLt) | EXP-T02 · `data/derived/exp-t02_stability_3rounds.csv` |
| **FP8 per-block GEMM** | **228.1±1.3 TFLOPS = 1.5× fp16 cuBLAS**(DeepGEMM 缩放策略在 Ada mma 落地) | 预量化孤立 GEMM,非端到端(在线量化端到端 72.9,量化 kernel 是瓶颈) | EXP-T06 · `data/derived/exp-t06_stability_3rounds.csv` |
| **flash-decoding(split-K)** | 32K 上下文 **2.39× vs naive**,GQA 原生不 repeat KV | 单轮,kernel 级终端级证据;引擎 fp32 probe PASS | EXP-T04 |
| **MoE unpermute** | **12.5× vs torch**(1.053±0.002 → 0.0845±0.0001 ms),gather 式无原子 | 单卡 permute/unpermute,T4096/D2048/E60/top4 | EXP-T07 · `data/derived/exp-t07_stability_3rounds.csv` |
| **CUDA Graph 消 launch** | 每调用 **36.2±0.1 → 3.11 µs = 11.6× 塌缩**,graph 后 Triton 反超 torch | 1024² softmax ×100 调用;地址稳定前提(动态 shape 需分桶) | EXP-T05 · `data/derived/exp-t05_stability_3rounds.csv` |

## 📊 图表(全部由 `scripts/plot_readme_figures.py` 从 derived 数据生成)

![FA2 vs SDPA](figures/fig1_fa2_vs_sdpa.png)

> 简化版 FA2 forward 随序列长逼近 SDPA-flash,S=4K 达 87%(形状 B1·H32/8·D128,fp16)。
> source: `data/derived/exp-t01_stability_3rounds.csv`(2026-08-24)

![GEMM stages 扫描](figures/fig2_gemm_stages.png)

> 同一 kernel 只变 num_stages:2 级双缓冲仅 +1%,3 级流水才 +21%,与 cuBLAS(torch.matmul dispatch)打平在误差条内——Ada 上搬运延迟长于一轮 dot,流水深度须按延迟/计算比配。
> source: `data/derived/exp-t02_stability_3rounds.csv`(2026-08-24)

![launch 四口径](figures/fig3_launch_cudagraph.png)

> "Triton 小核慢"的正解是上 CUDA Graph 而不是换 CUDA:graph 重放把每调用 36.2µs 塌缩到 3.11µs,反超 torch eager 与 torch+graph(对数轴)。
> source: `data/derived/exp-t05_stability_3rounds.csv`(2026-08-24)

## 🔬 代码导览

**在线 softmax 主循环**(`src/fa2_fwd.py`,完整 kernel ~80 行)——FA2 快的本质是不物化 S×S 注意力矩阵,用 m/l 两个跑动统计量边算边校正:

```python
# online softmax 的三个跑动量:行最大值 m、行和 l、输出累加 acc(全 fp32)
m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

for start_n in range(0, hi, BLOCK_N):   # K/V 分块流过片上,HBM 读写 O(S·D)
    ...                                 # load k/v;qk = q·kᵀ·scale + causal mask
    # online 更新:新块并入后,旧的 exp 统一乘 alpha 校正到新基准 m_new
    m_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.exp(m_i - m_new)         # 旧统计量的指数校正因子
    p = tl.exp(qk - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    m_i = m_new

acc = acc / l_i[:, None]                # 归一化推迟到循环外,全程无 S×S 物化
```

**流水线 GEMM**(`src/gemm_pipelined.py`)——CUDA 手写双缓冲(两块 shared memory + cp.async 交替)在 Triton 里是同一循环体加一个编译器旋钮,本仓把"双缓冲带来多少"变成可测数字(见 fig2):

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k0 in range(0, K, BLOCK_K):
    a = tl.load(a_ptrs, mask=..., other=0.0)  # num_stages=N 让编译器把 load
    b = tl.load(b_ptrs, mask=..., other=0.0)  # 与 dot 软件流水化:N=1 无重叠,
    acc = tl.dot(a, b, acc)                   # N=2 即双缓冲,N≥3 更深流水,
    a_ptrs += BLOCK_K * stride_ak             # shared memory 占用 ∝ N
    b_ptrs += BLOCK_K * stride_bk
```

其余:`src/fp8_gemm.py`(e4m3 per-block 反量化融合进 fp32 累加)、`src/flash_decode.py`(split-K 两段式 + online 归并)、`src/moe_permute.py`(gather 式 unpermute,无原子、求和顺序确定)。

## 🚀 复现 Quickstart

```bash
# 环境:CUDA GPU + torch ≥2.x + triton ≥3.x(实测环境见各 record §2)
python scripts/test_fa2.py           # FA2:6 形状正确性 gate + S=512..4K bench
python scripts/test_ew_gemm.py       # GEMM stages 扫描 + RMSNorm/softmax/int8
python scripts/test_fp8_gemm.py      # FP8 per-block GEMM 四口径
python scripts/test_moe_permute.py   # MoE permute/unpermute + 往返一致性
python scripts/test_cudagraph.py     # launch 四口径(eager/graph × triton/torch)

# 图表重算(仅需 matplotlib,从 data/derived/ 读数)
python scripts/plot_readme_figures.py
```

bench 只追加 UTC 前缀新文件,不覆盖已有 raw;`data/raw/EXP-*/manifest.txt` 记录 sha256+provenance。

## 🧭 仓库结构

```
src/          fa2_fwd / gemm_pipelined / fp8_gemm / flash_decode / moe_permute
              / elementwise_kernels + torch_ext/(CUDA kernel 绑定)
scripts/      test_*(正确性 gate + bench)、plot_readme_figures、kperf(无计数器观测)
records/      EXP-T01~T07 八节实验记录(假设跑前锁定,勘误留痕)
data/         raw/EXP-T*(不可变,含 3 轮 stability)+ derived/(聚合 mean/std)
figures/      全部脚本生成(本 README 三图)
docs/theory/  01 FlashAttention / 02 双缓冲 / 03 Triton vs CUDA 四口径 / 04 无 NCU 观测
              / 05 flash-decoding / 06 FP8@Ada(mma vs wgmma/TMA 界线)/ 07 MoE dispatch-combine
docs/talk/    面试讲稿(逐句过红线表)   docs/archive/  被取代文档(superseded 标注)
```

## 🧪 实验台账(状态唯一权威)

| 编号 | slug | 日期 | 状态 | 关键数字(指针) |
|---|---|---|---|---|
| [EXP-T01](records/EXP-T01_fa2_forward.md) | fa2_forward | 2026-08-23 | 完成 | 87% of SDPA@4K(data/raw/EXP-T01/) |
| [EXP-T02](records/EXP-T02_gemm_pipeline.md) | gemm_pipeline | 2026-08-23 | 完成 | 160.5 TFLOPS 打平 cuBLAS/8B 形状反超 4.8%(data/raw/EXP-T02/) |
| [EXP-T03](records/EXP-T03_ports_and_binding.md) | ports_and_binding | 2026-08-23 | 完成 | launch 三层反转(EXP-T02 json + EXP-T03/) |
| [EXP-T04](records/EXP-T04_flash_decoding.md) | flash_decoding | 2026-08-24 | 完成 | 32K 上下文 vs naive **2.39×**,GQA 原生;引擎 probe PASS |
| [EXP-T05](records/EXP-T05_cudagraph.md) | cudagraph | 2026-08-24 | 完成 | launch 塌缩 **11.6×**(36.8→3.1µs/调用,data/raw/EXP-T05/) |
| [EXP-T06](records/EXP-T06_fp8_gemm.md) | fp8_gemm | 2026-08-24 | 完成 | per-block FP8 **227.7/235.7 TFLOPS = 1.5× fp16 cuBLAS**(data/raw/EXP-T06/) |
| [EXP-T07](records/EXP-T07_moe_permute.md) | moe_permute | 2026-08-24 | 完成 | unpermute **12.5×** vs torch,gather 式无原子(data/raw/EXP-T07/) |

> **stability(2026-08-24 晚)**:headline 数字已全部 ≥3 轮复测,mean/std 见 `data/derived/exp-t01_stability_3rounds.csv` 等五份(T01/02/05/06/07);T05 launch 塌缩勘误 11.8×→**11.6×**。
> 阶段二增量(8/24):FP8 per-block GEMM(theory/06)、MoE permute/unpermute(theory/07,对照 DeepEP)、flash-decoding(theory/05)、CUDA Graph(theory/03 第四层)、kperf 无计数器观测(theory/04);TP=2 引擎侧见 llm-engine#EXP-D22。

## 📏 措辞红线表 + 方法论

**诚实度文化(本仓的差异化)**:每个进 README 的数字带 provenance(raw 首行/manifest 记录环境+命令+sha)且 ≥3 轮复测落 mean/std;假设在跑之前锁定判定阈值,证伪照登不删(T03 的 mask 假设证伪过程、"≤2×"假设错得有价值,全部保留);勘误留痕——T02 首轮未存盘数字作废、T05 11.8×→11.6× 修正,均在 record §7 与台账可查,旧值废弃后禁止复活。

| 红线 | 当前 | 说明 |
|---|---|---|
| "打平/反超 cuBLAS" | ✅ 可用 | 限两测形状 fp16(square4k 打平 0.4% 内、8B up_proj 反超 4.8%);未全形状扫描;只引存盘 raw 轮(T02 §7 勘误);cuBLAS=torch.matmul dispatch(cuBLASLt) |
| FP8 "1.5×" | 限定 | **预量化孤立 GEMM vs fp16 cuBLAS**,非端到端推理提速;在线量化端到端另列(72.8TF) |
| "87% of SDPA-flash" | ✅ 可用 | 完整限定:**简化版、仅 forward、4K 形状(B1·H32/8·D128)、对照=SDPA flash 后端**;缺一不引 |
| "Triton 比 CUDA 慢/快" | 🚫 禁裸说 | 必须区分 设备侧(同速)/launch(慢 25µs)/端到端(看融合),T03 三口径 |
| int8 三数字 | 限定 | 5.9µs(裸,scale 预置)/65µs(ext)/52µs(triton 融合)口径不得混引 |
| 关键数字 stability | ✅ 已补 | 3 轮 mean/std 落 data/derived/(2026-08-24);headline 全部复现(T05 修正 11.6×) |

## 🔗 相关仓

- [vllmExperience](https://github.com/lyell0710/vllmExperience) — vLLM 源码级实验(CUDA Graph/分派机制等,与本仓 T05/T06 互为表里)
- [Kernel_Optimazation](https://github.com/lyell0710/Kernel_Optimazation) — CUDA 手写四 kernel(本仓 T03 的 CUDA 侧对照,int8 v4 被零改动绑进本仓)
- [llm-engine](https://github.com/lyell0710/llm-engine) — 自研推理引擎,本仓 FA2/GEMM/flash-decode 的接入方(D15/D16/D22)

**远程**:本仓当前**仅本地**(无 git remote);推 GitHub 由用户建 repo 后 `git remote add origin ... && git push`。
