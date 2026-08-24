# EXP-T07 · MoE Permute/Unpermute(dispatch-combine 单卡版)

## 0. 元信息
| 日期 | 2026-08-24 | 环境 | v0.25.1-venv, RTX 4090 | 状态 | 完成 |
|---|---|---|---|---|---|
关联:阶段二"MoE Permute/Unpermute(Triton)";对照 DeepEP 讲 dispatch-combine。

## 1. 目的与假设
实现按专家排序的 permute 与加权收回的 unpermute。假设:Triton 数据搬运
≥ torch 参考;往返一致性 = bf16 噪声级;无原子实现可行。

## 2. 环境与配置
src/moe_permute.py:索引构建 torch argsort(stable)+ bincount;
permute=行 gather;unpermute=**gather 式**每 token 收 topk 行加权求和
(无原子、求和顺序确定)。形状 T4096/D2048/E60/topk4(Qwen1.5-MoE 族)。

## 3. 步骤
scripts/test_moe_permute.py:往返一致性 + 专家分段单调性断言 + 三段 bench。

## 4. 原始数据
data/raw/EXP-T07/moe_permute_bench.json。

## 5. 结果
| 项 | Triton | torch 参考 | 加速 |
|---|---|---|---|
| permute | 0.081 ms | 0.146 | 1.8× |
| unpermute | 0.084 ms | 1.054 | **12.5×** |
| 索引构建(torch) | 0.266 ms | — | 未专用化 |
往返一致 7.6e-3(bf16 加权和噪声);计数守恒 16384=T·topk;分段单调 PASS。

## 6. 分析与结论
- unpermute 12.5× 的来源=融合(torch 路径四趟 kernel/中间量 vs 单 kernel
  一读一写)——theory/03 第三层的又一实例。
- gather 式换掉 scatter-add:无原子、数值顺序确定(与 EXP-017 的重排
  数值教训呼应:能选顺序确定的实现就选)。
- **索引构建成大头(0.27ms > 两次搬运之和)**——这就是 vLLM 给它写专用
  CUDA kernel(moe_align_block_size)的实证理由;专用化 backlog。

## 7. 异常、偏差与开放问题
kernel 级 bench 未存逐轮 raw(json 存均值);未做 padding 对齐 grouped
GEMM 的 block 边界(接真实 fused_moe 时需要);跨卡版(all-to-all)不做
——阶段三读 DeepEP 的切入问题已在 theory/07 列出。

- backlog(2026-08-24 审计):本记录/README 引用的关键数字为**单轮** bench,待 GPU 空闲补 ≥3 轮 stability(mean/std 落 stability 文件)。

## 8. 下游影响
阶段二 MoE Permute 项闭环;与 EXP-014(fused_moe 56.4%)拼成完整
"MoE 层内时间去哪了"图景。

- **backlog 闭环(2026-08-24 晚)**:≥3 轮 stability 已补——unpermute 1.053±0.002 / 0.0845±0.0001 ms = 12.46×(12.5× 口径维持)。
  raw = data/raw/EXP-T07/*_stability_r{1,2,3}.json,聚合 = data/derived/exp-t07_stability_3rounds.csv。
