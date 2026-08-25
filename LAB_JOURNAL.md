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

## §4 2026-08-24 · 审计收尾批次

- **做了什么**:逐条闭环外部审计 findings——①theory/02 全文数字以存盘
  raw 改写(旧版移 docs/archive/ 标 superseded);②theory/01/03 的
  88%→87%、慢 12%→13%(4K 严格值 87.45%);③6 组 raw 目录补
  manifest.txt(sha256+provenance 勘注,不动文件本体不重命名);
  ④kperf 三卡登记为终端级证据(EXP-T06 §7,theory/04 §3 指去);
  ⑤docs/talk/ 首版讲稿(逐句过红线表);⑥README 数字加"单轮"限定+
  T01~T07 §7 补 stability backlog;⑦theory/05-07 补第 5 节"延伸"、
  Q&A 节名统一;⑧README 结构节更新、红线表补 cuBLAS 口径句。
- **为什么**:审计确认 theory 层数字与存盘 raw 漂移(首轮未存盘数字
  残留),关键数字缺 ≥3 轮 stability;GPU 被另一实验占用禁复测,按
  铁律 6 走"措辞降级 + backlog 登记",数字修订一律以仓内 raw 为准。
- **关键数字**:theory/02 现行=131.9/133.5/**160.5**/157.1 vs cuBLAS
  159.8(最优 stages=3,打平差 0.4% 内);qwen8b 154.4@s4 反超 147.3
  4.8%;FA2 4K=**87%**(87.45% 不进位)、慢 13%。全部指
  data/raw/EXP-T01,T02 存盘值。
- **产物路径**:docs/theory/01-07、docs/archive/02_double_buffering_20260823.md、
  docs/talk/triton_kernels_talk.md、data/raw/EXP-T0{1,2,3,5,6,7}/manifest.txt、
  records/EXP-T01~T07 §7、README.md、LAB_JOURNAL.md 本节。
- **下一步**:GPU 空闲后按各 record §7 backlog 补 ≥3 轮 stability
  (mean/std 落 stability 文件),解锁 README"单轮"限定;本仓仍无远端,
  待用户建 GitHub repo 后 push(README「远程」节)。

## §5 2026-08-24 晚 · stability backlog 闭环

- **做了什么**:五 bench(T01/02/05/06/07)各 3 轮 UTC 前缀落盘 + 通用聚合器
  (list 下钻 + 非有限值守卫)出 derived;README/红线表解除单轮限定。
- **关键数字**:FA2 87.2%(3 轮)、GEMM 打平复现、**T05 修正 11.8×→11.6×**
  (36.16±0.11/3.11)、fp8 228.1±1.3、moe 12.46×。
- **产物**:data/raw/EXP-T0{1,2,5,6,7}/*_stability_r*.json + data/derived/*_3rounds.csv。
- **下一步**:待用户建远端推送(仍无 remote)。

## §6 2026-08-25 · README 门面升级

- **做了什么**:README 重排为门面结构(headline 表→图表区→代码导览→
  Quickstart→结构→台账→红线表+方法论→相关仓);新增
  scripts/plot_readme_figures.py 从 data/derived/*_3rounds.csv 生成
  figures/fig1(FA2 vs SDPA 按 S)/fig2(GEMM stages 扫描)/fig3
  (launch 四口径对数轴),误差条=3 轮 std,固定配色。
- **为什么**:对外可读性(30 秒扫读拿到数据/图/代码/方法论);数字全部
  沿用现行文档与 derived,不新造;台账与红线表按铁律 1/6 原样保留,
  诚实度文化(provenance/≥3轮/证伪保留/勘误留痕)作为差异化如实写出。
- **关键数字**:无新增测量;图内数字=stability csv(fig1 87%@4K,
  fig2 159.4±1.2 vs cublas 160.0±0.7,fig3 36.16±0.11→3.11µs)。
- **产物路径**:README.md、scripts/plot_readme_figures.py、
  figures/fig{1,2,3}_*.png、LAB_JOURNAL.md 本节。
- **下一步**:待用户建远端后 push(仍无 remote)。
