# SPDX-License-Identifier: MIT
"""RMSNorm/Softmax/INT8-quantize/GEMM 的正确性 gate + benchmark(EXP-T02/T03)。

对照物命名诚实(CORE 铁律 6):pytorch_eager = 朴素 torch 表达式;
cublas = torch.matmul(fp16);cuda_v4 = Kernel_Optimazation 仓同尺寸实测值
(4090,EXP-K01 提交,fp32 口径,手工引用不重跑)。
GEMM 的 num_stages 扫描单列——"双缓冲的贡献"就是 stages=1→2 的差。
"""

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from elementwise_kernels import rmsnorm, softmax, int8_quantize  # noqa: E402
from gemm_pipelined import gemm  # noqa: E402


def bench(fn, iters=200, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    torch.manual_seed(0)
    out = {"correctness": {}, "bench": {}}

    # ---------- RMSNorm(fp16,llm-engine 形状族) ----------
    for rows, cols in [(2048, 1024), (2048, 4096)]:
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float16)
        w = torch.randn(cols, device="cuda", dtype=torch.float16)
        ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True)
                                       + 1e-6) * w.float()).half()
        err = (rmsnorm(x, w) - ref).abs().max().item()
        out["correctness"][f"rmsnorm_{rows}x{cols}"] = err
        t_tri = bench(lambda: rmsnorm(x, w))
        t_eag = bench(lambda: (x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).half())
        out["bench"][f"rmsnorm_{rows}x{cols}"] = {
            "triton_ms": round(t_tri, 5), "pytorch_eager_ms": round(t_eag, 5)}

    # ---------- Softmax(fp32,与 CUDA 仓同尺寸) ----------
    for rows, cols in [(1024, 1024), (1024, 1500), (8192, 8192)]:
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
        ref = torch.softmax(x, dim=-1)
        err = (softmax(x) - ref).abs().max().item()
        out["correctness"][f"softmax_{rows}x{cols}"] = err
        out["bench"][f"softmax_{rows}x{cols}"] = {
            "triton_ms": round(bench(lambda: softmax(x)), 5),
            "pytorch_ms": round(bench(lambda: torch.softmax(x, dim=-1)), 5)}

    # ---------- INT8 per-channel quantize(fp32,与 CUDA 仓同尺寸) ----------
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    x_big = torch.randn(8192, 8192, device="cuda", dtype=torch.float32)
    out["bench"]["int8q_8192x8192"] = {
        "triton_ms": round(bench(lambda: int8_quantize(x_big)), 5)}
    q, s = int8_quantize(x)
    deq = q.float() * s[:, None]
    out["correctness"]["int8q_1024x1024_maxabs"] = (deq - x).abs().max().item()
    out["correctness"]["int8q_scale_err_vs_ref"] = (
        s - x.abs().amax(1) / 127).abs().max().item()

    def eager_quant():
        sc = (x.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
        return torch.round(x / sc[:, None]).clamp(-127, 127).to(torch.int8), sc
    out["bench"]["int8q_1024x1024"] = {
        "triton_ms": round(bench(lambda: int8_quantize(x)), 5),
        "pytorch_eager_ms": round(bench(eager_quant), 5)}

    # ---------- GEMM:num_stages 扫描(双缓冲贡献)+ vs cuBLAS ----------
    for M, K, N, tag in [(4096, 4096, 4096, "square4k"),
                         (2048, 4096, 12288, "qwen8b_mlp_up")]:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        ref = a @ b
        err = (gemm(a, b) - ref).abs().max().item()
        rel = err / ref.abs().max().item()
        out["correctness"][f"gemm_{tag}_relerr"] = rel
        tf = 2 * M * N * K / 1e12
        row = {"cublas_ms": round(bench(lambda: a @ b), 4)}
        for st in (1, 2, 3, 4):
            ms = bench(lambda: gemm(a, b, num_stages=st))
            row[f"stages{st}_ms"] = round(ms, 4)
            row[f"stages{st}_tflops"] = round(tf / (ms / 1e3), 1)
        row["cublas_tflops"] = round(tf / (row["cublas_ms"] / 1e3), 1)
        out["bench"][f"gemm_{tag}"] = row

    print(json.dumps(out, indent=1))
    if len(sys.argv) > 1:
        p = Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"provenance": sys.argv[2] if len(sys.argv) > 2 else "",
                   **out}, open(p, "w"), indent=1)


if __name__ == "__main__":
    main()
