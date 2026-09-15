# SPDX-License-Identifier: MIT
"""FA2 backward 正确性 gate + benchmark。

正确性:与 torch autograd(朴素 fp32 物化 attention 的反向)逐梯度比 max abs err,
阈值 2e-2 —— 与 forward gate 同口径(fp16 输入、fp32 累加的经验界)。
同时报 forward 重跑的 O 误差,用于区分"forward 本身偏"与"backward 推错"。
覆盖:MHA/GQA、causal/非causal、seq 非块整除、双 head_dim、is_grads_batched=False。

性能:vs torch SDPA(A 的 backward 走 flash 后端)+ vs 朴素物化反向,
同 dtype 同 shape。FLOPs 口径见 flops_bwd()。
"""

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fa2_bwd import fa2_backward  # noqa: E402
from fa2_fwd import fa2_forward  # noqa: E402


def ref_attention_torch(q, k, v, causal):
    """朴素物化 attention(fp32),供 autograd 求参考梯度。"""
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    kr = k.repeat_interleave(Hq // Hkv, dim=1).float().requires_grad_(True)
    vr = v.repeat_interleave(Hq // Hkv, dim=1).float().requires_grad_(True)
    qf = q.float().requires_grad_(True)
    s = (qf @ kr.transpose(-1, -2)) * (D ** -0.5)
    if causal:
        mask = torch.tril(torch.ones(S, S, device=q.device, dtype=torch.bool))
        s = s.masked_fill(~mask, float("-inf"))
    o = torch.softmax(s, dim=-1) @ vr
    return o, qf, kr, vr


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
    print("| B | Hq | Hkv | S | D | causal | fwd_err | dq_err | dk_err | dv_err | pass |")
    ok = True
    for B, Hq, Hkv, S, D, causal in cases:
        q = torch.randn(B, Hq, S, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.float16)

        # ── 我们的 forward+backward ──
        o, lse = fa2_forward(q, k, v, causal=causal, return_lse=True)
        do = torch.randn_like(o)
        dq, dk, dv = fa2_backward(q, k, v, o, do, lse, causal=causal)

        # ── torch autograd 参考(朴素 fp32) ──
        o_ref, qf, kr, vr = ref_attention_torch(q, k, v, causal)
        o_ref.backward(do.float())
        # GQA:kv 侧梯度需把同组 q head 的贡献加回(参考侧 repeat 过,故求和)
        dk_ref = (kr.grad.sum(dim=1, keepdim=True)
                  .expand(-1, Hkv, -1, -1).reshape(B, Hkv, S, D) if False
                  else kr.grad.reshape(B, Hq, S, D).reshape(B, Hkv, Hq // Hkv, S, D).sum(dim=2))
        dv_ref = (vr.grad.reshape(B, Hkv, Hq // Hkv, S, D).sum(dim=2)
                  if False else vr.grad.reshape(B, Hq, S, D)
                  .reshape(B, Hkv, Hq // Hkv, S, D).sum(dim=2))

        e_fwd = (o.float() - o_ref).abs().max().item()
        e_dq = (dq.float() - qf.grad).abs().max().item()
        e_dk = (dk.float() - dk_ref).abs().max().item()
        e_dv = (dv.float() - dv_ref).abs().max().item()
        # 【易错】不能用 max(a, nan) < 2e-2 判:Python 的 max 遇 nan 会静默返回另一个
        # 操作数(nan 的比较恒 False),于是 nan 反而"通过"。这里显式把 nan/非有限
        # 一律判失败——本 gate 初版就因此漏报过 nan。
        errs = (e_fwd, e_dq, e_dk, e_dv)
        finite = all(v == v and abs(v) != float("inf") for v in errs)
        p = finite and max(errs) < 2e-2
        ok &= p
        print(f"| {B} | {Hq} | {Hkv} | {S} | {D} | {causal} | {e_fwd:.2e} | {e_dq:.2e} "
              f"| {e_dk:.2e} | {e_dv:.2e} | {p} |")
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


def flops_bwd(B, H, S, D, causal):
    """backward 的 FLOPs:重算 S 一遍(2 S²D)+ dP/dV(2 S²D)+ dS/dQ/dK(4 S²D)
    = 8 S²D per head;再乘 B·H。causal 减半。与 forward 的 4 S²D 比约 2×。"""
    f = 8.0 * B * H * S * S * D
    return f / 2 if causal else f


def bench():
    rows = []
    for S in (512, 1024, 2048, 4096):
        B, Hq, Hkv, D, causal = 1, 32, 8, 128, True
        q = torch.randn(B, Hq, S, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.float16)

        o, lse = fa2_forward(q, k, v, causal=True, return_lse=True)
        do = torch.randn_like(o)
        ours = bench_one(lambda: fa2_backward(q, k, v, o, do, lse, causal=True))

        # SDPA backward(flash 后端)
        qf = q.detach().clone().requires_grad_(True)
        kf = k.detach().clone().requires_grad_(True)
        vf = v.detach().clone().requires_grad_(True)

        def sdpa_bwd():
            with torch.nn.attention.sdpa_kernel(
                    torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                oo = torch.nn.functional.scaled_dot_product_attention(
                    qf, kf, vf, is_causal=True, enable_gqa=True)
            oo.backward(do, retain_graph=True)
            qf.grad = kf.grad = vf.grad = None

        sdpa = bench_one(sdpa_bwd)

        # 朴素物化反向(仅小 S 测,显存 O(S²))
        if S <= 2048:
            qn = q.detach().clone().requires_grad_(True)
            kn = k.detach().clone().requires_grad_(True)
            vn = v.detach().clone().requires_grad_(True)

            def naive_bwd():
                Bs, Hs, Ss, Ds = qn.shape
                kr = kn.repeat_interleave(Hs // kn.shape[1], dim=1).float()
                vr = vn.repeat_interleave(Hs // vn.shape[1], dim=1).float()
                s = (qn.float() @ kr.transpose(-1, -2)) * (Ds ** -0.5)
                mask = torch.tril(torch.ones(Ss, Ss, device=qn.device, dtype=torch.bool))
                s = s.masked_fill(~mask, float("-inf"))
                (torch.softmax(s, dim=-1) @ vr).backward(do.float())
                qn.grad = kn.grad = vn.grad = None

            naive = bench_one(naive_bwd)
        else:
            naive = float("nan")

        tf = flops_bwd(B, Hq, S, D, causal) / 1e12
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
