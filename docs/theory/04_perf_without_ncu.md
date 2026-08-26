---
topic: 无 NCU 环境的 kernel 性能观测
status: 完成(工具=scripts/kperf.py)
---

# 04 · 没有性能计数器,怎么看 kernel 性能

## 1. 一句话结论

本容器宿主驱动 `RmProfilingAdminOnly=1` 封死了 NCU/CUPTI 计数器(容器内
无解,需宿主开 `NVreg_RestrictProfilingToAdminUsers=0`)——但 NCU 四大件
里三件有不依赖计数器的平替:**计时(CUDA event)、SOL(手算 roofline)、
occupancy(编译产物资源占用+公式)**;唯独 stall 细分没有直接平替,用
**变体对照 + nsys 时间线**近似归因。`scripts/kperf.py` 一键出观测卡。

## 2. 方法(NCU 四大件 ↔ 平替)

| NCU 给的 | 平替 | 精度损失 |
|---|---|---|
| Duration | CUDA event / wall+sync(≥100 iters + warmup) | 无 |
| SOL(带宽/算力 %峰值) | bytes、flops 自己算,除以实测时间,对 1008GB/s / 165TFLOPS | 无(前提:bytes 算对——含读+写,别忘广播/复用) |
| Occupancy | Triton 编译缓存的 n_regs/n_spills/shared + Ada 资源上限(65536 regs、100KB smem、1536 thr/SM)套公式,并报**限制因子** | 理论值,非实测 achieved |
| Warp stall 细分 | ①变体对照(控制变量:去掉某优化测差值);②nsys 时间线看 kernel 间隙/并发;③in-kernel clock64() 分段 | 定性近似 |

**判读口诀**(kperf 卡片的用法):
- 带宽 % 高、算力 % 低 → memory-bound,优化方向=访存(向量化/合并/复用);
- 算力 % 高 → compute-bound,看是否已贴 tensor core 峰值;
- **occupancy 低≠坏**:本仓 GEMM occ 17%(regs 限制,170/线程)却打出
  98% 峰值算力——tensor core kernel 用寄存器堆 ILP,比高 occupancy 更值钱;
  occupancy 只在"延迟藏不住"(两个 % 都低)时才是嫌疑人。
- n_spills>0 是红灯(寄存器溢出到 local memory),先于一切优化处理。

## 3. 本项目实证(kperf 三卡,4090)

| kernel | 带宽 | 算力 | occ(限制因子) | 判读 |
|---|---|---|---|---|
| softmax 8192² | **91%** | 0% | 67%(regs) | memory-bound,已贴 roofline,没油水 |
| FA2 S=4K | 7% | **74%** | 17%(regs 213) | compute-bound;与 SDPA 差的 13% 在 tile/布局层 |
| GEMM 4096³ | 12% | **98%** | 17%(regs 170) | 到顶;occ 低是设计而非缺陷 |

三卡数字为**终端级证据**(kperf.py 单轮终端输出,未存 raw),
登记锚点=EXP-T06《FP8 GEMM》§7;引用时以该登记为准。

## 4. 面试追问 Q&A

- **Q: 没有 NCU 你怎么定位瓶颈?** 先 kperf 卡片分 memory/compute-bound;
  再变体对照归因(本仓 mask 假设证伪、launch 三点法都是实例);nsys 看
  时间线级问题(launch 间隙、kernel 并发、graph)。
- **Q: 理论 occupancy 和 achieved 差在哪?** 理论=资源上限允许的驻留;
  achieved 受 tail effect/负载不均影响,只能计数器测——引用时注明是理论值。
- **Q: 为什么 GEMM 低 occupancy 反而好?** 每线程多寄存器 → 更大
  per-thread tile → 复用更多、指令级并行更深;隐藏延迟的手段从"多 warp
  切换"换成"单 warp 内多路在飞"。NCU 的 occupancy section 也会提示
  "not a limiter"。
- **Q: 这个限制在生产环境常见吗?** 常见(共享集群默认关计数器);
  所以"无计数器方法论"本身是工程能力,不只是权宜。

## 5. 延伸(锚点)

CUDA Occupancy Calculator 公式;Volkov《Better performance at lower
occupancy》;nsys 的 cuda-graph-trace=node 教训(vllm/experiments#EXP-014《D1 MoE decode 分解》);
本仓 kperf.py;Laptop 时代 NCU 数据(Kernel_Optimazation/artifacts/)作
stall 细分的历史参照。
