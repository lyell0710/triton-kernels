# 白板推导卡 · FA2 在线 softmax 与 split-K 归并


**推导目标**：证明分块计算 softmax(QKᵀ)V 与一次算完等价。
1. 定义部分统计量：第 p 块 K/V 给出（m_p， l_p， acc_p），其中 m_p=max_j s_j，l_p=Σ_j e^{s_j−m_p}，acc_p=Σ_j e^{s_j−m_p} v_j。
2. 在线合并（FA2 主循环，块内）：m'=max(m， m_p)； l' = l·e^{m−m'} + l_p·e^{m_p−m'}；acc' = acc·e^{m−m'} + acc_p·e^{m_p−m'}。 **关键句**：e^{s−m'} = e^{s−m}·e^{m−m'}，旧量整体乘一个因子即可换基准。
3. 块间归并（flash-decoding，一次成型）：m=max_p m_p； l=Σ l_p e^{m_p−m}；out=Σ acc_p e^{m_p−m} / l。 **面试点**：同一可归并性用了两次——块内 online、块间 reduce； 数值稳定性来自"减当前最大值"，不是近似。
4. 边界追问预案：为什么最终除 l 而不是每块先除？（每块先除就不可归并了， 除法必须最后做——llm-engine 曾靠 fp32 复跑验证该实现，EXP-D11《KV Cache 正确性》。）

