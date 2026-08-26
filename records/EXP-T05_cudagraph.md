# EXP-T05 · CUDA Graph 消 launch 开销实测(launch 三层结论的"解法"层)

## 0. 元信息
| 日期 | 2026-08-24 | 环境 | v0.25.1-venv， RTX 4090 | 状态 | 完成 |
|---|---|---|---|---|---|
关联：EXP-T03《三件套移植 + torch 绑定》backlog。

## 1. 目的与假设
把 N=100 次 launch 主导的小 kernel(1024² softmax)录成一张 graph 重放。假设：每调用成本塌缩到 kernel 本体量级（消掉 ~30µs Python 分发）。

## 2-3. 配置与步骤
scripts/test_cudagraph.py：eager 循环 vs torch.cuda.graph 捕获重放， triton 与 torch.softmax 双对照；预热跑 3 次（JIT+分配器）后捕获。

## 4. 原始数据
data/raw/EXP-T05/cudagraph_bench.json（provenance 首字段）。

## 5. 结果(每次调用 µs)
| 路径 | eager | graph | 消除 |
|---|---|---|---|
| triton softmax | 36.76 | **3.11** | **11.8×，消 33.6µs/调用** |
| torch.softmax | 7.93 | 4.04 | 2.0× |
graph 后 triton(3.11)反超 torch eager(7.93)与 torch graph(4.04)。

## 6. 分析与结论
假设成立且超预期：graph 重放把 Python/C++ 分发全部变成 graph 节点， 剩下的 3.1µs ≈ kernel 本体+节点调度——**"Triton 小核慢"的正解是上 Graph，不是换 CUDA**（graph 后 triton kernel 本体反而最快）。这正是 vLLM 用 CUDA Graph 吃 decode launch 海的机理（vllm/experiments#EXP-014《D1 MoE decode 分解》的 graph-trace 教训同根）。

## 7. 异常、偏差与开放问题
捕获要求地址稳定（静态输入/输出）；动态 shape 需分桶捕获（vLLM 的 cudagraph_capture_sizes 即此）；llm-engine 整 step 捕获未做（KV cache 动态增长需静态化改造，backlog）。

- backlog（2026-08-24 审计）：本记录/README 引用的关键数字为**单轮** bench，待 GPU 空闲补 ≥3 轮 stability（mean/std 落 stability 文件）。

## 8. 下游影响
launch 三层结论（T03）补全为四层：设备同速 → 分发差 → 融合反转 → **graph 归零**；theory/03 增补。

- **backlog 闭环（2026-08-24 晚）**:≥3 轮 stability 已补——eager 36.16±0.11µs / replay 3.11±0.00µs → 塌缩修正 11.8×→11.6×。 raw = data/raw/EXP-T05/*_stability_r{1,2,3}.json，聚合 = data/derived/exp-t05_stability_3rounds.csv。
