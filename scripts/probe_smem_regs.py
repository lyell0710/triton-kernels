# SPDX-License-Identifier: MIT
"""EXP-T08（num_stages 与 shared memory 份数的映射）:编译期资源探针——num_stages / BLOCK_N 与 shared memory 份数的映射。

假设(跑前锁定):Triton 的 num_stages=N 会为喂给 tl.dot 的 load 分配 N 份
片上缓冲,故 smem(N) = N × 单份 tile 字节。判定:读编译产物的
metadata.shared,若 smem(2) == smem(1) 则假设被证伪。

只做编译 + 一次极小 launch,不产生任何 benchmark 时延数字(profiler 隔离
不适用:本脚本不写 bench 表)。
用法: python scripts/probe_smem_regs.py [out.json] ["<provenance>"]
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torch
from gemm_pipelined import gemm, _gemm_kernel
from fa2_fwd import fa2_forward, _fa2_fwd_kernel

dev = torch.cuda.current_device()
res = {"env": {"gpu": torch.cuda.get_device_name(0),
               "triton": __import__("triton").__version__,
               "torch": torch.__version__},
       "gemm_stages": [], "fa2_blockn": []}


def drain(jit, tag, extra):
    for k in jit.device_caches[dev][0].values():
        row = {**extra, "shared_bytes": k.metadata.shared, "n_regs": k.n_regs,
               "n_spills": k.n_spills, "num_warps": k.metadata.num_warps}
        res[tag].append(row)
        print(tag, row)


# --- GEMM: BM128/BN128/BK64/w8,单份 tile = (128+128)*64*2B = 32 KB ---
a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
b = torch.randn(512, 512, device="cuda", dtype=torch.float16)
for st in (1, 2, 3, 4):
    _gemm_kernel.device_caches[dev][0].clear()
    gemm(a, b, num_stages=st)
    drain(_gemm_kernel, "gemm_stages", {"num_stages": st})

# --- FA2: bench 形状 B1·H32/8·S4096·D128 fp16,BM=128 ---
q = torch.randn(1, 32, 4096, 128, device="cuda", dtype=torch.float16)
k = torch.randn(1, 8, 4096, 128, device="cuda", dtype=torch.float16)
v = torch.randn(1, 8, 4096, 128, device="cuda", dtype=torch.float16)
for bn in (32, 64, 128):
    for st in (2, 3):
        _fa2_fwd_kernel.device_caches[dev][0].clear()
        try:
            fa2_forward(q, k, v, block_n=bn, num_stages=st)
            drain(_fa2_fwd_kernel, "fa2_blockn",
                  {"block_n": bn, "num_stages": st})
        except Exception as e:
            row = {"block_n": bn, "num_stages": st,
                   "error": type(e).__name__ + ": " + str(e)[:200]}
            res["fa2_blockn"].append(row)
            print("fa2_blockn", row)

if len(sys.argv) > 1:
    p = Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"provenance": sys.argv[2] if len(sys.argv) > 2 else "", **res},
              open(p, "w"), indent=1)
    print("saved", p)
