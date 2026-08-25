"""EXP-T04 backlog 补测:flash-decoding 正确性 gate + 3 轮可存盘 bench。
协议=EXP-T04 原协议:B=1,Hq=16,Hkv=8,D=128,bf16,Skv ∈ {512,2048,8192,32768};
对照 = naive decode attention(torch,fp16,GQA repeat)。正确性 vs fp32 精算。
用法:python test_flash_decode.py <out.json> "<provenance>"
"""
import json, sys, time
sys.path.insert(0, "/root/projects/triton-kernels/src")
import torch
from flash_decode import flash_decode

def naive_math(q, kr, vr, scale):
    """attention 数学本体(kv 已 repeat)。"""
    s = (q @ kr.transpose(-1, -2)) * scale
    return torch.softmax(s, dim=-1) @ vr

def naive_incl_repeat(q, k, v, scale):
    """含 repeat 成本口径:GQA 原生 kernel 免掉的正是这份实体化拷贝。"""
    g = q.shape[1] // k.shape[1]
    return naive_math(q, k.repeat_interleave(g, dim=1),
                      v.repeat_interleave(g, dim=1), scale)

def ref_fp32(q, k, v, scale):
    g = q.shape[1] // k.shape[1]
    kr = k.repeat_interleave(g, dim=1).float(); vr = v.repeat_interleave(g, dim=1).float()
    s = (q.float() @ kr.transpose(-1, -2)) * scale
    return (torch.softmax(s, dim=-1) @ vr).to(q.dtype)

def bench(fn, iters=200, warmup=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3

out = {"correctness": {}, "bench": {}}
torch.manual_seed(0)
B, Hq, Hkv, D = 1, 16, 8, 128
scale = D ** -0.5
for Skv in (512, 2048, 8192, 32768):
    q = torch.randn(B, Hq, 1, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, Skv, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, Skv, D, device="cuda", dtype=torch.bfloat16)
    o = flash_decode(q, k, v, sm_scale=scale)
    r = ref_fp32(q, k, v, scale)
    err = (o - r).abs().max().item()
    out["correctness"][f"skv{Skv}"] = err
    fd = bench(lambda: flash_decode(q, k, v, sm_scale=scale))
    g = q.shape[1] // k.shape[1]
    kr = k.repeat_interleave(g, dim=1); vr = v.repeat_interleave(g, dim=1)
    nv_pre = bench(lambda: naive_math(q, kr, vr, scale))     # 原口径:repeat 预置
    del kr, vr
    nv_incl = bench(lambda: naive_incl_repeat(q, k, v, scale))  # 完整口径
    out["bench"][f"skv{Skv}"] = {"flash_decode_ms": round(fd, 5),
        "naive_prerepeat_ms": round(nv_pre, 5), "speedup_vs_prerepeat": round(nv_pre / fd, 3),
        "naive_incl_repeat_ms": round(nv_incl, 5), "speedup_vs_incl_repeat": round(nv_incl / fd, 3),
        "pass": err < 2e-2}
    print(Skv, "err", f"{err:.2e}", "fd", round(fd,4),
          "pre", round(nv_pre,4), f"x{nv_pre/fd:.2f}",
          "incl", round(nv_incl,4), f"x{nv_incl/fd:.2f}")
if len(sys.argv) > 1:
    with open(sys.argv[1], "w") as f:
        json.dump({"provenance": sys.argv[2] if len(sys.argv) > 2 else "", **out}, f, indent=1)
