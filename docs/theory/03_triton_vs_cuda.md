---
topic: Triton vs CUDA 性能差的真实来源
status: 完成(实证=EXP-T03)
---

# 03 · Triton vs CUDA:差距不在 kernel,在 launch 与融合

> 8/24 增补第四层:**CUDA Graph 把 launch 归零**——同一 1024² softmax,
> eager 36.2µs/调用(3 轮) → graph 重放 **3.11µs(11.6×,3 轮:36.16±0.11 → 3.11±0.00)**,反超 torch eager;
> "Triton 小核慢"的正解是上 Graph,不是换 CUDA(EXP-T05)。

## 1. 一句话结论

同一行核(softmax/quantize)在**带宽主导尺寸下 Triton 与 torch/CUDA 同速**
(8192²:917 vs 922 GB/s,双双贴 4090 roofline 91%);小尺寸的 4-16× "差距"
**全部来自主机侧**:Triton Python 分发 ~30µs > torch C++ ~8µs > 裸 CUDA
~5µs。而端到端还有第三层反转:Triton 单 kernel 融合(52µs)反超
"更快的 CUDA kernel + 3 次 torch 前置 launch"(65µs)。

## 2. 机制(排障过程即讲解,EXP-T03 §7 的三步)

1. **现象**:softmax 1024² Triton 36µs vs torch 8µs(4.4×);int8 quantize
   更差。CUDA v4 同尺寸 5.9µs(EXP-K01)。
2. **假设一(证伪)**:掩码 load 阻断向量化 → 加"整除走无 mask 快路径",
   数字纹丝不动——**猜测要交给对照实验,哪怕猜错也要记录**。
3. **假设二(坐实)**:尺寸三点法。8×8(纯开销)37.4µs ≈ 1024² 的
   37.6µs → 时间根本不在 kernel 里;8192²(带宽主导)917 vs 922 GB/s
   打平 → 设备侧没有差距。开销拆解:Triton 每次调用过 Python 包装
   (参数处理/JIT 缓存查找/grid 计算)+ wrapper 里的 empty 分配。

**结论怎么用**(选型准则):
- 大 kernel / 长序列 / 融合机会多 → Triton 白给(FA2 87% 效率
  (EXP-T01),GEMM 打平(EXP-T02));
- 微 kernel 高频调用 → 裸 CUDA/C++ 扩展,或 CUDA Graph 把 launch 摊平
  (vLLM 正是用 CUDA Graph 吃掉 decode 的 launch 海——与
  vllm/experiments#EXP-014 的 graph-trace 陷阱同根);
- torch 集成层:**融合数(launch 数)比单 kernel 快慢更重要**——
  CUDA v4 kernel 快 10×,套上 3 个 scale 前置 launch 后端到端反输。

## 3. 本项目实证(EXP-T03,4090)

| 场景 | Triton | 对照 | 读法 |
|---|---|---|---|
| softmax 8×8(纯开销) | 37.4 µs | torch 8.0 | launch 开销差 |
| softmax 1024² | 37.6 µs | torch 8.1 / CUDA v4 7.8 | 被开销遮蔽 |
| softmax 8192² | 917 GB/s | torch 922 GB/s | **设备侧同速** |
| int8q 端到端(torch 层) | **51.7 µs(1 launch 融合)** | ext-CUDA v4 65.1(4 launch) | 融合 > 单核快慢 |
| int8q 裸 bench | — | CUDA v4 5.9 µs | 不含封装的下限 |

RMSNorm 顺带:vs pytorch_eager 1.8×/6.3×(2048×1024/4096)——eager 多
kernel 多中间量,融合收益同一原理。

## 4. 面试追问 Q&A

- **Q: 所以"Triton 比 CUDA 慢"对吗?** 在本仓测的行核与 GEMM 上,设备侧
  不对;成立的是"Triton 调用一次比 CUDA 扩展贵 ~25-60µs"。说清测的是
  kernel 还是调用链,是这题的全部。
- **Q: 生产里怎么消 launch 开销?** CUDA Graph 录制整个 decode step
  (launch 全部变 graph 节点重放);torch.compile 的 kernel 融合;
  或把小算子拼进邻居(RMSNorm 融进 GEMM 的 epilogue)。
- **Q: 你的 CUDA v4 为什么裸跑 5.9µs 而 torch 层 65µs?** 65 = kernel 5.9
  + scale 的 3 个 torch kernel + 4 次分发;裸 bench 的 scale 是预先算好
  传入的(EXP-K01 口径注明)——对照物口径差异本身要摆上台面。

## 5. 延伸(锚点)

Triton dispatch 源码(runtime/jit.py 的 launch 路径);CUDA Graph 文档;
vLLM cudagraph_mode 配置(vllm/experiments#EXP-014 附带教训:profiling
时 graph 内 kernel 需 node 级 trace);本仓 `scripts/test_ew_gemm.py`。
