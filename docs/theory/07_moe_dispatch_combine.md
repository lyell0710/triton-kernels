---
topic: MoE Permute/Unpermute 与 DeepEP 的 dispatch-combine
status: 完成(实证=EXP-T07)
---

# 07 · MoE 的搬运问题:单卡 permute ↔ 跨卡 all-to-all

## 1. 一句话结论
grouped GEMM 要求"同专家 token 连续",于是 MoE 每层两次搬运:permute
(按专家排序展开)与 unpermute(按 router 权重收回)。单卡内它是内存
gather;跨卡 EP 时同一代数变成 all-to-all——DeepEP 的 dispatch/combine
就是它的 NVLink/RDMA 版。本仓 Triton 实现:permute 1.8×、**unpermute
12.5×** 于 torch 参考,且 gather 式设计天然无原子加。

## 2. 机制
- **索引与搬运分离**:排序/直方图(确定每行去哪)用 torch argsort
  (vLLM 用专用 CUDA kernel moe_align_block_size 干这件事,本机实测
  索引构建 0.27ms 反而是大头——这就是它值得专用 kernel 的证据);
  大块数据搬运交给 Triton 行 gather。
- **unpermute 的反直觉设计**:直觉是 scatter-add(每个专家输出行加回
  原 token 位)→ 必须原子加;改成 **gather 式**(每个 token 自己去收
  topk 行加权求和)→ 无竞争、顺序确定(数值可复现!)。EPLB 一致性
  实验(vllm/experiments#EXP-017)里"求和顺序改变→输出漂移"的教训,
  在这里反向应用:选顺序确定的实现。
- **对照 DeepEP**:dispatch = permute 的跨卡版(token 按目标专家所在
  rank 分桶 all-to-all);combine = unpermute 的跨卡版(部分结果收回
  加权)。DeepEP 的增值在通信与计算重叠(SM 收发分工、hook 式流水),
  单卡版没有"重叠"可言——这条差异就是阶段三读 DeepEP 的切入问题。

## 3. 本项目实证(EXP-T07,T=4096 D=2048 E=60 topk=4,bf16)
| 项 | Triton | torch 参考 | 加速 |
|---|---|---|---|
| permute | 0.081 ms | 0.146 | 1.8× |
| unpermute | 0.084 ms | 1.054 | **12.5×** |
| 索引构建(torch argsort) | 0.266 ms | — | 待专用化 |
往返一致性 7.6e-3(bf16 加权和噪声级);专家分段单调性断言通过。

## 4. 面试追问 Q&A
- **Q: 为什么 unpermute 差距那么大?** torch 路径 = 反索引构建 + 高级
  索引 gather + 广播乘 + sum 四趟 kernel/中间量;Triton 单 kernel 一趟
  读一趟写——又是融合>单核(theory/03 第三层)的实例。
- **Q: 索引构建怎么专用化?** 直方图+前缀和+桶内序三个小 kernel
  (即 moe_align 的结构);收益上限 = 0.27ms → ~0.03ms,backlog。

## 5. 延伸(锚点)

DeepEP README(dispatch/combine、SM 收发分工与 hook 式通信-计算重叠
——阶段三阅读的切入问题见 §2);vLLM 的 moe_align_block_size 专用
kernel(vllm csrc/moe/);vllm/experiments#EXP-014(fused_moe 56.4%)
与 #EXP-017(求和顺序数值教训);本仓 `src/moe_permute.py` 与
`scripts/test_moe_permute.py`。
