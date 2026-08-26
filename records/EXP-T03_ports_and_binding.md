# EXP-T03 · 三件套移植 + torch 绑定:launch 开销与融合的三层反转

## 0. 元信息
| 日期 | 2026-08-23 | 环境 | 同 T01 | 状态 | 完成 |
|---|---|---|---|---|---|
关联：阶段一"RMSNorm/Softmax+INT8 移植 Triton"+"PyTorch 绑定"； 对照引用 Kernel_Optimazation#EXP-K01《四 kernel 4090 重基准》（4090 CUDA 数字）。

## 1. 目的与假设
移植三行核并回答"Triton vs CUDA 性能差在哪"；绑定 CUDA v4 kernel 进 torch extension。假设（跑前）：Triton 与 CUDA 差距 ≤2×。

## 2. 环境与配置
src/elementwise_kernels.py（行并行模式）+ src/torch_ext/int8_binding.cpp（cpp_extension.load，复用 Kernel_Optimazation quantize_v4.cu 零改动）。

## 3. 步骤
正确性 gate → bench → 假设一（mask 阻断向量化）对照证伪 → 假设二（launch 开销）三点法坐实 → 绑定正确性（逐元素）+ 端到端对比。

## 4. 原始数据
data/raw/EXP-T02/ew_gemm_bench.json（行核段）、EXP-T03/binding_bench.txt； 排障中间数字为终端级（theory/03 §2 记全过程）。

## 5. 结果
正确性：RMSNorm err≤1e-3、softmax≤2e-6、int8 逐元素 0 mismatch（绑定版 1M 元素全等）。性能三层：①8192² softmax 0.585 vs torch 0.583 ms（917/921 GB/s，同速，roofline 91%）；②1024² 差 4.4× 全为 launch 开销（8×8 纯开销 37.4 vs 8.0µs）；③torch 层端到端：Triton 融合 41.6µs（存盘 raw；绑定对比会话为 51.7）反超 ext-CUDA 65.1µs (1 vs 4 launch)；裸 CUDA v4 5.9µs（EXP-K01，口径=scale 预置）。 RMSNorm vs eager：1.8×/6.3×。

## 6. 分析与结论
原假设（≤2×）在小尺寸上"错得有价值"：设备侧其实 1.0×，差距全在主机侧——结论升级为选型准则（大核用 Triton/小核高频用 CUDA+Graph/集成层看 launch 数），theory/03 §2。

## 7. 异常、偏差与开放问题
mask 假设证伪过程保留（快路径代码留仓，虽无性能差，语义无害）； CUDA Graph 消 launch 的实测未做（backlog）；int8 对照物三口径（裸/ext/融合）已在表内注明，引用时不得混；triton 融合数字跨会话 41.6~52µs 波动（launch 开销对系统状态敏感），引用带区间； raw provenance sha=pre-commit（同 T01 §7 说明）。

- backlog（2026-08-24 审计）：本记录/README 引用的关键数字为**单轮** bench，待 GPU 空闲补 ≥3 轮 stability（mean/std 落 stability 文件）。

## 8. 下游影响
简历"Triton vs CUDA"问答有自家三层数据；绑定工装可复用到 FA kernel。
