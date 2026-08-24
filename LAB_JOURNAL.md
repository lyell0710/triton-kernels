# LAB_JOURNAL — triton-kernels

## §1 建仓一日:FA2 + 流水线 GEMM + 三件套 + 绑定(2026-08-23,EXP-T01~03)

- **做了什么**:①FA2 forward 从零(causal+GQA+非整除,6 形状全过)+ 7 配置
  tile 扫描定优配;②流水线 GEMM(grouped launch)num_stages 1→4 扫描;
  ③RMSNorm/Softmax/INT8 移植 + "性能差"排障(一个假设证伪、一个坐实);
  ④CUDA v4 quantize 绑 torch extension(逐位一致)。
- **为什么**:阶段一清单核心 + 解 llm-engine D15/D16 阻塞。
- **关键数字**:FA2 88% of SDPA@4K;GEMM 162 TFLOPS 追平 cuBLAS
  (双缓冲 2 级仅 +1%,3 级 +19%——深度按延迟/计算比配);launch 三层:
  设备同速 917/922GB/s、分发 37/8/5.9µs、端到端融合反超(52 vs 65µs)。
- **方法论**:排障三点法(纯开销尺寸/遮蔽尺寸/带宽尺寸)一次定位 launch
  开销;mask 假设证伪照记(猜错也入档)。
- **产物**:src 4 件、scripts 2 件、records T01-03、theory 01-03(五节全
  实证)、raw 3 组、README 红线表。
- **下一步**:llm-engine EXP-D15/D16 接入(本仓 fa2_forward 与
  gemm_pipelined.linear 已按其 attention_impl/linear 契约留口)。
## §2 fp32 通道 + 引擎接入协同(2026-08-23 深夜)

- **做了什么**:为 llm-engine 的 FP32 复跑 gate 打通 fp32 通道——dtype
  自适应 tile(fp32 字节翻倍,BM128 超 Ada 100KB shared 上限,两次 OOM
  实测后定 BM32/stages1)+ IEEE dot 开关(TF32 的 3e-3 → 6e-7);linear
  加小 M 自适应与混合策略支撑 D16。
- **关键数字**:FA2 单层 fp32-IEEE 误差 6e-7(算法精确性证明);引擎级
  FP32 probe 5.3e-5/8.0e-5 双 PASS;引擎 bench:FA2 TTFT -9.4%(0.6B)
  /-6.5%(8B);triton-linear 0.6B regime 负结果 → 甜区边界闭环(theory/03)。
- **事故**:一次 cwd 漂移把 D15/D16 记录写进本仓(已移回 llm-engine,
  本仓自检归零)——批量脚本首行显式 cd,教训第 N 次。
- **产物**:fa2_fwd/gemm_pipelined 的 fp32 路径;llm-engine#EXP-D15/D16
  由本仓依赖支撑。

## §3 阶段二冲刺:FP8 GEMM + MoE permute + kperf(2026-08-24)

- **做了什么**:①kperf(NCU 平替四件套:计时/roofline/occupancy/对照归因,
  三 kernel 观测卡);②FP8 per-block GEMM(e4m3,BLOCK_K=缩放组硬对齐,
  反量化融合累加);③MoE permute/unpermute(gather 式无原子);
  ④flash-decoding 与 CUDA Graph(见 §2 与 T04/T05)。
- **关键数字**:FP8 **227.7/235.7 TFLOPS = 1.5× fp16 cuBLAS**(kernel 精确性
  1.9e-4);MoE unpermute **12.5×**;kperf 揭示 GEMM occ 17% 却 98% 峰值
  (occupancy 不是目的);索引构建 0.27ms > 搬运之和(moe_align 专用
  kernel 的存在理由)。
- **产物**:src/{fp8_gemm,moe_permute,flash_decode}.py、kperf.py、
  records T04~T07、theory 04~07、raw 四组。
- **下一步**:阶段二清单四项全闭环(TP=2 在 llm-engine#EXP-D22);
  阶段三仅剩"读 DeepEP"(纯阅读)与真实 PR(用户动作)。
