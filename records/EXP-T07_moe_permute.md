# EXP-T07 · MoE Permute/Unpermute(dispatch-combine 单卡版)

> **一句话结论**：MoE dispatch-combine 的 unpermute 快 torch 参考 **12.5×**，来源是融合（四趟 kernel 合成一读一写）；但索引构建本身 0.27ms 已超过两次搬运之和——这正是 vLLM 要为它专门写 CUDA kernel 的实证理由。

## 0. 元信息
| 日期 | 2026-08-24 | 环境 | v0.25.1-venv， RTX 4090 | 状态 | 完成 |
|---|---|---|---|---|---|
关联：阶段二"MoE Permute/Unpermute(Triton)"；对照 DeepEP 讲 dispatch-combine。

## 1. 目的与假设
实现按专家排序的 permute 与加权收回的 unpermute。假设：Triton 数据搬运 ≥ torch 参考；往返一致性 = bf16 噪声级；无原子实现可行。

## 2. 环境与配置
src/moe_permute.py：索引构建 torch argsort(stable)+ bincount； permute=行 gather；unpermute=**gather 式**每 token 收 topk 行加权求和（无原子、求和顺序确定）。形状 T4096/D2048/E60/topk4（Qwen1.5-MoE 族）。

## 3. 步骤
scripts/test_moe_permute.py：往返一致性 + 专家分段单调性断言 + 三段 bench。

## 4. 原始数据
data/raw/EXP-T07/moe_permute_bench.json。

## 5. 结果
| 项 | Triton | torch 参考 | 加速 |
|---|---|---|---|
| permute | 0.081 ms | 0.146 | 1.8× |
| unpermute | 0.084 ms | 1.054 | **12.5×** |
| 索引构建（torch） | 0.266 ms |— | 未专用化 |
往返一致 7.6e-3（bf16 加权和噪声）；计数守恒 16384=T·topk；分段单调 PASS。

## 6. 分析与结论
- unpermute 12.5× 的来源=融合（torch 路径四趟 kernel/中间量 vs 单 kernel 一读一写）——theory/03 第三层的又一实例。
- gather 式换掉 scatter-add：无原子、数值顺序确定（与 vllm/experiments#EXP-017《D5 EPLB gate》的重排数值教训呼应：能选顺序确定的实现就选）。
- **索引构建成大头（0.27ms > 两次搬运之和）**——这就是 vLLM 给它写专用 CUDA kernel(moe_align_block_size)的实证理由；专用化 backlog。

## 7. 异常、偏差与开放问题
kernel 级 bench 未存逐轮 raw（json 存均值）；未做 padding 对齐 grouped GEMM 的 block 边界（接真实 fused_moe 时需要）；跨卡版（all-to-all）不做——阶段三读 DeepEP 的切入问题已在 theory/07 列出。

- backlog（2026-08-24 审计）：本记录/README 引用的关键数字为**单轮** bench，待 GPU 空闲补 ≥3 轮 stability（mean/std 落 stability 文件）。

## 8. 下游影响
阶段二 MoE Permute 项闭环；与 vllm/experiments#EXP-014《D1 MoE decode 分解》(fused_moe 56.4%)拼成完整 "MoE 层内时间去哪了"图景。

- **backlog 闭环（2026-08-24 晚）**:≥3 轮 stability 已补——unpermute 1.053±0.002 / 0.0845±0.0001 ms = 12.46×（12.5× 口径维持）。 raw = data/raw/EXP-T07/*_stability_r{1,2,3}.json，聚合 = data/derived/exp-t07_stability_3rounds.csv。

---

## 8. 上游 PR 前景：查重结论 = **不建议投**（2026-09-15 复核）

本记录 §5 的 unpermute（gather 式无原子、每 token 收 topk 行加权）曾被登记为「MoE unpermute/moe_align 上游 PR」的候选。**按"先查重"纪律复核后否决**，理由两条，都是实测/实读代码得到的：

**(1) vLLM 已有同设计的专用 CUDA kernel —— 本实现是"重新发明"，不是"填补空白"。**
`csrc/libtorch_stable/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.inl`：
- `expandInputRowsKernel`（permute）+ `finalizeMoeRoutingKernel`（unpermute/reduce）；
- permute 侧源码注释**逐字写着本实现的立论**：
  > "I need the reverse map for that reduction to allow each threadblock to do 1 k-way reduce **without atomics** later in MoE. 1 thread block will be responsible for all k summations."
- 另一份现代路径 `csrc/moe/moe_permute_unpermute_op.cu` 同样存在。

即：**「gather 式、无原子、每 block 负责一个 token 的全部 k 次求和」正是 vLLM 现网采用的设计**。§5 的 12.5× 是对 **torch 参考**（argsort + scatter-add）的，不是对 vLLM 现有 kernel —— 把它当"上游可改进项"是口径错位。

**(2) moe_align / token-alignment 这一侧已被三个 open PR 占满。**

| PR | 状态 | 内容 |
|---|---|---|
| [#56383](https://github.com/vllm-project/vllm/pull/56383) | open（2026-09-11） | `batched_moe_align_block_size` 去掉 1-block 限制、格点并行、合并写、免 O(N) 哨兵初始化（+196/−15） |
| [#55225](https://github.com/vllm-project/vllm/pull/55225) | open（2026-09-03） | 小 batch decode 的 token-alignment 旁路（`M·topk·4 ≤ E`）、block size 钳到 16、细粒度专家自适应 tile（+338/−23，报 1.50×） |
| [#53911](https://github.com/vllm-project/vllm/pull/53911) | open（2026-08-26） | DeepEP-v2 `globalize + align + count_and_sort` 三 kernel 融合成一次 launch（报 1.23–2.01×） |

另有一条**警示**：[#43932](https://github.com/vllm-project/vllm/pull/43932) 是对 "[Perf] Optimize moe permute by pre-allocate buffer" 的 **revert** —— 该区域的优化历史上被回滚过，说明收益/复杂度权衡比较严格。

**结论**：本仓 EXP-T07 的成果**留在本仓作为"独立复现了一个工业级设计"的证据**（这对面试是加分项：说明设计判断与库作者一致），**但不作为上游 PR 立项**。§7 原有的"索引构建 0.27ms 超过两次搬运之和"的观察仍有价值 —— 若要找上游增量，方向应是**索引构建（argsort/bincount）的 kernel 化**而非搬运本身；但该方向也需先查是否有 PR 覆盖。

**对简历的影响（已核对）**：简历写的是「为 MoE unpermute 设计 gather 式无原子方案替代 scatter-add，快 **torch 参考** 12.5×」—— 口径本身正确（明确写了对照物是 torch 参考），不构成夸大。但**面试需主动说明**「该设计与 vLLM 现网 CUDA 实现一致，我是独立复现而非首创」，避免被追问时的被动。
