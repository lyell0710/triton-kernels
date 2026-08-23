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