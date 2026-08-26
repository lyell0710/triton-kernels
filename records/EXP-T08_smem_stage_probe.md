# EXP-T08 · num_stages 与 shared memory 份数的映射:编译期资源探针

> **一句话结论**：假设被证伪：`num_stages=2` 并没有开出第二份 shared memory，实测映射是**份数 = max(1, num_stages − 1)**。因此 EXP-T02《流水线 GEMM》「2 级只 +1%」的正确读法不是「双缓冲只值 1%」，而是「这一档还不是双缓冲」。

## 0. 元信息
| 日期 | 2026-08-25 | 环境 | py312(triton3.6/torch2.11), RTX 4090, driver 610.57.04 | 状态 | 完成 |
|---|---|---|---|---|---|
关联：讲义 01/02 深化（docs/lectures/）；解释 EXP-T02《流水线 GEMM》§6「2 级只 +1%」与 EXP-T01《Triton FA2 forward》「BN=128 OOM(160KB)」两条结论的机理。

## 1. 目的与假设
**跑前假设（判定阈值先锁）**：Triton 的 `num_stages=N` 会为喂给 `tl.dot` 的 load 分配 **N 份**片上缓冲，故编译产物的 `metadata.shared` 应满足 `smem(N) = N × 单份 tile 字节`。 **证伪条件**：若 `smem(2) == smem(1)`，则「num_stages 即缓冲份数」被证伪。

## 2. 环境与配置
scripts/probe_smem_regs.py；只做编译 + 一次极小 launch，读 `CompiledKernel.metadata.shared` / `.n_regs` / `.n_spills`，不产生时延数字。被测：src/gemm_pipelined.py 的 `_gemm_kernel`(BM128/BN128/BK64/w8， 单份 tile =(128+128)×64×2B = 32 KB)与 src/fa2_fwd.py 的 `_fa2_fwd_kernel`（bench 形状 B1·H32/8·S4096·D128 fp16，BM=128；单份 K/V tile = BN×128×2B×2）。

## 3. 步骤
```
python scripts/probe_smem_regs.py data/raw/EXP-T08/<UTC>_smem_regs_probe.json "<provenance>"
```

## 4. 原始数据
data/raw/EXP-T08/20260825T230651_smem_regs_probe.json（manifest.txt 记 sha256）。

## 5. 结果

GEMM（单份 tile = 32 KB）:

| num_stages | metadata.shared | 份数 = shared/32KB | n_regs | n_spills |
|---|---|---|---|---|
| 1 | 32768 (32 KB) | 1 | 162 | 0 |
| 2 | 32768 (32 KB) | **1** | 136 | 0 |
| 3 | 65536 (64 KB) | 2 | 170 | 0 |
| 4 | 98304 (96 KB) | 3 | 178 | 0 |

FA2（Q tile 32 KB 常驻 + K/V 缓冲）：

| BLOCK_N | num_stages | metadata.shared | n_regs |
|---|---|---|---|
| 32 | 2 | 49152 (48 KB) | 192 |
| 32 | 3 | 65536 (64 KB) | 200 |
| 64 | 2 | 65536 (64 KB) | 213 |
| 64 | 3 | 98304 (96 KB) | 198 |
| 128 | 2 | 98304 (96 KB) | 255 |
| 128 | 3 | **OutOfResources: Required 163840, Hardware limit 101376** | — |

## 6. 分析与结论
- **假设被证伪**：`smem(2) == smem(1) == 32 KB`。实测映射是 **份数 = max(1， num_stages − 1)**，两个 kernel 一致： GEMM `shared = max(1,N−1)×32KB`；FA2 `shared = 32KB(Q) + max(1,N−1)×(BN×128×2B×2)`， 六格逐格吻合（48/64/64/96/96/160 KB）。
- **对 EXP-T02 结论的机理补充**：`num_stages=2` 这一档**并没有开出第二份缓冲**， 所以「2 级只 +1%」的正确读法不是"双缓冲只值 1%"，而是"这一档还不是双缓冲"； 真正的双缓冲是 `num_stages=3`（64 KB = 2 份），它带来 +20%。EXP-T02 的**数字不变**， 变的是对它的解释。
- **对 EXP-T01 结论的机理补充**：BN=128 的 OOM 数字 160 KB 被逐字复现（`Required: 163840`），且 `Hardware limit: 101376` = 99 KB，与 NVIDIA Ada Tuning Guide §1.4.1.1「maximum shared memory per thread block is 99 KB」一致。 OOM 只在 `num_stages=3` 出现；BN=128 + stages=2 可编译（96 KB）但寄存器打到 255（Ada 每线程上限），说明这一档换成了寄存器压力。
- **交叉验证**：GEMM stages=3 的 `n_regs=170`、FA2 BN=64/stages=2 的 `n_regs=213` 与 kperf 卡片（终端级证据，登记于 EXP-T06《FP8 GEMM》§7）逐字相同；由此 occupancy = 8 warps ÷ 48 warps/SM = 16.7% ≈ 17% 可由「64K regs/SM ÷（regs/线程 × 256 线程） < 2 ⇒ 每 SM 仅 1 个 CTA」直接推出，不再需要观测。

## 7. 异常、偏差与开放问题
- 本探针只读编译产物，不含运行时行为；「stages=4 在 qwen8b 形状上快于 stages=3」的原因（3 份缓冲 vs 2 份 + 调度差异）未隔离，留作开放项。
- 未 dump TTGIR/PTX 验证 `cp.async` 的 commit/wait 分组数量与缓冲份数的对应关系； 本记录只到「份数」这一层。
- 未扫 num_warps 维度；寄存器数随 stages 非单调（162/136/170/178）未解释。

## 8. 下游影响
- docs/lectures/02 §3.3/§3.4 的流水深度模型按本记录改写（份数 = N−1）； docs/lectures/01 §3.6 的 BN=128 OOM 账目由「推断」升为「实测复现」。
- EXP-T01/T02 的**性能数字与限定语不变**；仅机理解释更新。
