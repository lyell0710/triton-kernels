# SPDX-License-Identifier: MIT
"""EXP-T05: CUDA Graph 消 launch 开销实测。

场景 = launch 主导的小 kernel 高频调用(EXP-T03（三件套移植 + torch 绑定）的 1024² softmax,
eager 每调用 ~37µs 其中 ~30µs 是 Python 侧分发)。把 N 次调用录成一张
graph,重放时 launch 全部变 graph 节点——每调用成本应塌缩到 kernel 本体。
"""
import json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from elementwise_kernels import softmax

def wall(fn, it=50, wu=10):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it*1e3

N = 100
x = torch.randn(1024, 1024, device="cuda")
out = {}

# eager 循环:N 次 triton softmax
out["eager_loop_ms"] = wall(lambda: [softmax(x) for _ in range(N)])

# graph 捕获同样的 N 次调用
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    for _ in range(3): softmax(x)          # 预热(JIT+分配器)
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g):
    for _ in range(N): y = softmax(x)
out["graph_replay_ms"] = wall(lambda: g.replay())

# torch.softmax 对照(C++ 分发)
out["torch_loop_ms"] = wall(lambda: [torch.softmax(x, -1) for _ in range(N)])
g2 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g2):
    for _ in range(N): y2 = torch.softmax(x, -1)
out["torch_graph_ms"] = wall(lambda: g2.replay())

per = {k: round(v*1000/N, 2) for k, v in out.items()}   # µs/次
print("每次调用成本 (µs):", json.dumps(per, indent=1))
print(f"triton: eager {per['eager_loop_ms']} → graph {per['graph_replay_ms']} "
      f"(消掉 {per['eager_loop_ms']-per['graph_replay_ms']:.1f} µs/调用, "
      f"{per['eager_loop_ms']/per['graph_replay_ms']:.1f}x)")
if len(sys.argv) > 1:
    p = Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"provenance": sys.argv[2] if len(sys.argv)>2 else "",
               "N": N, "shape": "1024x1024 fp32",
               "total_ms": out, "per_call_us": per}, open(p,"w"), indent=1)
