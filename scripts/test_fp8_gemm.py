# SPDX-License-Identifier: MIT
"""EXP-T06: FP8 per-block GEMM 正确性 + benchmark。

正确性双层:①kernel 精确性 = 与"逐块反量化后 fp32 精算"比(应 ~1e-3 级,
只含累加序差);②量化保真 = 与原 fp16 矩阵乘比(fp8 量化误差本体)。
bench 三对照:fp8 prequant / fp16 triton gemm(EXP-T02)/ cublas fp16。
"""
import json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fp8_gemm import fp8_gemm, fp8_gemm_prequant, quant_fp8_block, quant_fp8_token_group
from gemm_pipelined import gemm

def bench(fn, it=50, wu=15):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it*1e3

torch.manual_seed(0)
out = {"correctness": {}, "bench": {}}
for M, K, N, tag in [(4096, 4096, 4096, "square4k"),
                     (2048, 4096, 12288, "qwen8b_mlp_up")]:
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    a8, sa = quant_fp8_token_group(a); b8, sb = quant_fp8_block(b)
    # ①kernel 精确性:反量化参考
    a_dq = (a8.float().reshape(M, K//128, 128) * sa[:, :, None]).reshape(M, K)
    b_dq = (b8.float().reshape(K//128, 128, N//128, 128)
            * sb[:, None, :, None]).reshape(K, N)
    ref_dq = a_dq @ b_dq
    y = fp8_gemm_prequant(a8, sa, b8, sb, out_dtype=torch.float32)
    rel_kernel = ((y - ref_dq).abs().max() / ref_dq.abs().max()).item()
    # ②量化保真:vs 原 fp16
    ref_fp16 = (a.float() @ b.float())
    rel_quant = ((y - ref_fp16).abs().max() / ref_fp16.abs().max()).item()
    out["correctness"][tag] = {"kernel_relerr": rel_kernel,
                               "quant_relerr_vs_fp16": rel_quant}
    tf = 2*M*N*K/1e12
    r = {"fp8_prequant_ms": round(bench(lambda: fp8_gemm_prequant(a8, sa, b8, sb)), 4),
         "fp8_e2e_online_quant_ms": round(bench(lambda: fp8_gemm(a, b)), 4),
         "fp16_triton_ms": round(bench(lambda: gemm(a, b)), 4),
         "fp16_cublas_ms": round(bench(lambda: a @ b), 4)}
    for k in list(r):
        r[k.replace("_ms", "_tflops")] = round(tf/(r[k]/1e3), 1)
    out["bench"][tag] = r
    print(tag, json.dumps(out["correctness"][tag]), json.dumps(r))
if len(sys.argv) > 1:
    p = Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"provenance": sys.argv[2] if len(sys.argv)>2 else "", **out},
              open(p, "w"), indent=1)
