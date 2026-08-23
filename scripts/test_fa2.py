# SPDX-License-Identifier: MIT
"""FA2 forward 正确性 gate + benchmark。

正确性:与 fp32 精算参考(手写 softmax(QK^T)V)比 max abs err,
阈值 2e-2(fp16 输入、fp32 累加的经验界;逐 shape 打印实际值)。
覆盖:MHA/GQA、causal/非causal、seq 非块整除、双 head_dim。
性能:vs torch SDPA(flash 后端)与 naive(物化 S²)——同 dtype 同 shape。
"""

import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fa2_fwd import fa2_forward  # noqa: E402


def ref_attention(q, k, v, causal):
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    kr = k.repeat_interleave(Hq // Hkv, dim=1).float()
    vr = v.repeat_interleave(Hq // Hkv, dim=1).float()
    s = (q.float() @ kr.transpose(-1, -2)) * (D ** -0.5)
    if causal:
        mask = torch.tril(torch.ones(S, S, device=q.device, dtype=torch.bool))
        s = s.masked_fill(~mask, float("-inf"))
    return (torch.softmax(s, dim=-1) @ vr).to(q.dtype)


def check():
    torch.manual_seed(0)
    cases = [
        # (B, Hq, Hkv, S, D, causal)
        (1, 8, 8, 512, 64, True),
        (2, 16, 16, 1024, 128, True),
        (1, 16, 8, 1024, 128, True),    # GQA 2:1
        (1, 16, 4, 777, 128, True),     # GQA 4:1 + 非整除 seq
        (1, 8, 8, 512, 64, False),
        (1, 32, 8, 2048, 128, True),    # Qwen3-8B 形状族
    ]
    print("| B | Hq | Hkv | S | D | causal | max_abs_err | pass |")
    ok = True
    for B, Hq, Hkv, S, D, causal in cases:
        q = torch.randn(B, Hq, S, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.float16)
        out = fa2_forward(q, k, v, causal=causal)
        ref = ref_attention(q, k, v, causal)
        err = (out - ref).abs().max().item()
        p = err < 2e-2
        ok &= p
        print(f"| {B} | {Hq} | {Hkv} | {S} | {D} | {causal} | {err:.2e} | {p} |")
    return ok


def bench_one(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def flops_attn(B, H, S, D, causal):
    f = 4.0 * B * H * S * S * D          # QK^T 与 PV 各 2*S*S*D
    return f / 2 if causal else f


def bench():
    rows = []
    for S in (512, 1024, 2048, 4096):
        B, Hq, Hkv, D, causal = 1, 32, 8, 128, True
        q = torch.randn(B, Hq, S, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.float16)

        ours = bench_one(lambda: fa2_forward(q, k, v, causal=True))
        with torch.nn.attention.sdpa_kernel(
                torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            sdpa = bench_one(lambda: torch.nn.functional.
                             scaled_dot_product_attention(
                                 q, k, v, is_causal=True, enable_gqa=True))
        naive = bench_one(lambda: ref_attention(q, k, v, True)) if S <= 2048 \
            else float("nan")
        tf = flops_attn(B, Hq, S, D, causal) / 1e12
        rows.append({"S": S, "ours_ms": round(ours, 4),
                     "sdpa_flash_ms": round(sdpa, 4),
                     "naive_fp32_ms": round(naive, 4),
                     "ours_tflops": round(tf / (ours / 1e3), 1),
                     "sdpa_tflops": round(tf / (sdpa / 1e3), 1)})
        print(rows[-1])
    return rows


if __name__ == "__main__":
    passed = check()
    print("CORRECTNESS", "PASS" if passed else "FAIL")
    rows = bench()
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        prov = sys.argv[2] if len(sys.argv) > 2 else ""
        json.dump({"provenance": prov, "correctness_pass": passed,
                   "bench": rows}, open(out, "w"), indent=1)
