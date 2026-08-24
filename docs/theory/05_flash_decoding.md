---
topic: Flash-Decoding
status: 完成(实证=EXP-T04)
---

# 05 · Flash-Decoding:decode 的并行度从哪来

## 1. 一句话结论
decode 时 Sq=1,FA2 的 Q 行块并行完全失效(B·H 个 program 喂不饱 128 SM);
Flash-Decoding 把并行度改从 **KV 序列维**取:分段算部分 softmax 统计量
(m_p, l_p, acc_p),再一次 online 归并——本仓实测 32K 上下文 vs naive
**2.39×**,短上下文受两次 launch 地板(~90µs)限制持平。

## 2. 机制(三步)
1. **为什么 FA2 不行**:grid=(S_q/BM, B·H),S_q=1 时第一维=1;0.6B 的
   B·H=16 → 16 个 program 对 128 SM,7/8 的卡在闲着。
2. **split-K**:KV 切 num_splits 段,grid=(splits, B·H)。每段独立跑
   与 FA 相同的 online softmax,输出三个部分量(不除 l!)。
3. **归并的代数**(softmax 可归并性第二次使用):
   m=max(m_p);l=Σ l_p·e^{m_p−m};out=Σ acc_p·e^{m_p−m} / l。
   空段 m_p=−inf 自动零权,数值天然安全。
splits 选择 = 填满 SM(B·H·splits ≳ 2×SM)与段长 ≥ BLOCK_N 的折中。

## 3. 本项目实证(EXP-T04)
kernel 级 1.21×(512)→ **2.39×**(32K);GQA 原生(不 repeat kv,带宽
减半的另一半收益);引擎级 512 上下文 TPOT 持平(attention 占比 <10%,
合理)且 fp32 probe PASS。接线教训:胶水层一个 .contiguous() 每步整拷
KV cache,曾把收益变负——kernel 快 ≠ 引擎快,拷贝藏在调用约定里。

## 4. 面试追问 Q&A
- **Q: 和 paged attention 什么关系?** 正交:paged 解决 KV 的**寻址**
  (block table 间接层),flash-decoding 解决**并行度**;vLLM 的 decode
  kernel 二者兼有(paged 读 + split 归并)。
- **Q: 为什么不用 atomic 归并?** 部分量之间要按 m 重标定后才能加,
  原子加做不了非线性归并;两段式(或单 kernel + 跨 CTA 同步)是标准解。
- **Q: 短上下文为什么不赢?** 两次 kernel launch 的固定成本(本机 ~90µs)
  > 计算差;正解是 CUDA Graph(EXP-T05:launch 塌缩 11.6×)或 persistent
  kernel。

## 5. 延伸(锚点)

Flash-Decoding 官方博客(Dao 等,2023,PyTorch blog);FA2 论文
(2307.08691)§3(split 归并与 online softmax 同一代数);vLLM decode
attention kernel(paged 读 + split 归并的合体;paged 衔接见
vllm/experiments 白板图);本仓 `src/flash_decode.py`(partial/combine
两 kernel)与 llm-engine 接线(llm-engine#EXP-D11/D14)。
