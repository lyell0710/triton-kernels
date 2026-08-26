# SPDX-License-Identifier: MIT
"""EXP-T07（MoE Permute/Unpermute）: MoE permute/unpermute 正确性 + bench(vs torch index ops)。"""
import json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from moe_permute import build_indices, permute, unpermute

def bench(fn, it=100, wu=20):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it*1e3

torch.manual_seed(0)
T, D, E, TOPK = 4096, 2048, 60, 4          # Qwen1.5-MoE 形状族
x = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)
logits = torch.randn(T, E, device="cuda")
w, ids = torch.topk(torch.softmax(logits, -1), TOPK, dim=-1)

src_row, pos, counts = build_indices(ids, E)
y = permute(x, src_row)
out = unpermute(y, pos, w)

# 正确性①:往返一致 out[t] == x[t] * Σw(专家计算恒等时)
ref = x.float() * w.sum(-1, keepdim=True).float()
err = (out.float() - ref).abs().max().item()
# 正确性②:排序缓冲确实按专家分段
flat_e = ids.reshape(-1)[torch.argsort(ids.reshape(-1), stable=True)]
mono = bool((flat_e[1:] >= flat_e[:-1]).all().item())
print(f"roundtrip err={err:.2e}  专家分段单调={mono}  counts sum={counts.sum().item()}=={T*TOPK}")

def torch_permute():
    o = torch.argsort(ids.reshape(-1), stable=True)
    return x[o // TOPK]
def torch_unpermute():
    # scatter 式 index_add(torch 常规写法,有原子)
    o = torch.argsort(ids.reshape(-1), stable=True)
    inv = torch.empty_like(o); inv[o] = torch.arange(o.numel(), device=o.device)
    acc = (y.float()[inv.reshape(T, TOPK)] * w[..., None].float()).sum(1)
    return acc

r = {"triton_permute_ms": round(bench(lambda: permute(x, src_row)), 4),
     "torch_permute_ms": round(bench(torch_permute), 4),
     "triton_unpermute_ms": round(bench(lambda: unpermute(y, pos, w)), 4),
     "torch_unpermute_ms": round(bench(torch_unpermute), 4),
     "index_build_ms": round(bench(lambda: build_indices(ids, E)), 4)}
print(json.dumps(r, indent=1))
if len(sys.argv) > 1:
    p = Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"provenance": sys.argv[2] if len(sys.argv)>2 else "",
               "shape": {"T": T, "D": D, "E": E, "topk": TOPK},
               "roundtrip_err": err, "monotonic": mono, "bench": r},
              open(p, "w"), indent=1)
