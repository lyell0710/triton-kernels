# SPDX-License-Identifier: MIT
"""README 门面图 fig1-3:全部从 data/derived/*_stability_3rounds.csv 读数生成。

用法: /root/venvs/kernel-opt/bin/python scripts/plot_readme_figures.py
产物: figures/fig{1,2,3}_*.png(白底 dpi=240;误差条=3 轮 std)
规范: 单图单结论(标题即结论句);图脚注写源数据文件+日期;
      配色固定 我方 #1a6fb8 / 次强调 #0f4c81 / 基线 #c0392b / 中性 #999。
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
DERIVED = ROOT / "data" / "derived"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

# 中文字体
_FONT = "/usr/share/fonts/truetype/arphic/uming.ttc"
font_manager.fontManager.addfont(_FONT)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=_FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

C_OURS = "#1a6fb8"      # 我方
C_OURS2 = "#0f4c81"     # 次强调
C_BASE = "#c0392b"      # 基线/对照
C_NEUT = "#999999"      # 中性
INK = "#333333"

DATA_DATE = "2026-08-24"  # stability 3 轮落盘日期(csv 首行 provenance)


def load_csv(name):
    """读 derived csv → {metric: (mean, std)}。首行 provenance 以 # 开头被跳过。"""
    out = {}
    with open(DERIVED / name) as f:
        for row in csv.DictReader(r for r in f if not r.startswith("#")):
            out[row["metric"]] = (float(row["mean"]), float(row["std"]))
    return out


def style_ax(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=10)
    ax.xaxis.grid(True, color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)


def footnote(fig, src):
    fig.text(0.01, 0.01, f"source: {src} · {DATA_DATE} · RTX 4090 · 误差条=3轮std",
             fontsize=8, color="#777777")


# ---------- fig1: FA2 ours vs SDPA-flash 按序列长 ----------
def fig1():
    d = load_csv("exp-t01_stability_3rounds.csv")
    seqs = [512, 1024, 2048, 4096]
    ours = [d[f"bench[{i}].ours_ms"] for i in range(4)]
    sdpa = [d[f"bench[{i}].sdpa_flash_ms"] for i in range(4)]
    eff = [s[0] / o[0] * 100 for o, s in zip(ours, sdpa)]  # SDPA时间/我方时间

    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor="white")
    ys = range(len(seqs))
    h = 0.38
    ax.barh([y + h / 2 for y in ys], [o[0] for o in ours], height=h,
            color=C_OURS, xerr=[o[1] for o in ours], ecolor=INK,
            error_kw={"lw": 1}, label="FA2 简化版(本仓)")
    ax.barh([y - h / 2 for y in ys], [s[0] for s in sdpa], height=h,
            color=C_BASE, xerr=[s[1] for s in sdpa], ecolor=INK,
            error_kw={"lw": 1}, label="torch SDPA(flash 后端)")
    for y, o, s, e in zip(ys, ours, sdpa, eff):
        ax.text(o[0] * 1.02 + 0.005, y + h / 2, f"{o[0]:.3f} ms({e:.0f}%)",
                va="center", fontsize=9, color=INK)
        ax.text(s[0] * 1.02 + 0.005, y - h / 2, f"{s[0]:.3f} ms",
                va="center", fontsize=9, color=INK)
    ax.set_yticks(list(ys), [f"S={s}" for s in seqs])
    ax.set_xlabel("时延(ms,越短越好)", fontsize=10, color=INK)
    ax.set_xlim(0, max(o[0] for o in ours) * 1.3)
    ax.set_title("80 行简化版 FA2 forward 达 SDPA-flash 的 87%(S=4K,3 轮)",
                 fontsize=12, color=INK, pad=12)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    style_ax(ax)
    footnote(fig, "data/derived/exp-t01_stability_3rounds.csv")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGS / "fig1_fa2_vs_sdpa.png", dpi=240, facecolor="white")
    plt.close(fig)


# ---------- fig2: GEMM num_stages 扫描 vs cuBLAS ----------
def fig2():
    d = load_csv("exp-t02_stability_3rounds.csv")
    rows = [("cuBLAS(torch.matmul)", d["bench.gemm_square4k.cublas_tflops"], C_BASE)]
    for s in (4, 3, 2, 1):
        c = C_OURS2 if s == 3 else C_OURS
        rows.append((f"Triton stages={s}", d[f"bench.gemm_square4k.stages{s}_tflops"], c))

    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor="white")
    ys = range(len(rows))
    ax.barh(list(ys), [r[1][0] for r in rows], height=0.62,
            color=[r[2] for r in rows], xerr=[r[1][1] for r in rows],
            ecolor=INK, error_kw={"lw": 1})
    for y, r in zip(ys, rows):
        ax.text(r[1][0] + 2.5, y, f"{r[1][0]:.1f}", va="center",
                fontsize=10, color=INK)
    ax.set_yticks(list(ys), [r[0] for r in rows])
    ax.set_xlabel("TFLOPS(4096³ fp16,越高越好)", fontsize=10, color=INK)
    ax.set_xlim(0, 185)
    ax.set_title("2 级双缓冲仅 +1%,3 级流水才 +21% 打平 cuBLAS(误差条内)",
                 fontsize=12, color=INK, pad=12)
    style_ax(ax)
    footnote(fig, "data/derived/exp-t02_stability_3rounds.csv")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGS / "fig2_gemm_stages.png", dpi=240, facecolor="white")
    plt.close(fig)


# ---------- fig3: launch 开销四口径(对数轴)----------
def fig3():
    d = load_csv("exp-t05_stability_3rounds.csv")
    rows = [  # 自上而下:eager → graph;颜色跟实体:蓝=Triton,红=torch,灰=torch+graph
        ("Triton eager 循环", d["per_call_us.eager_loop_ms"], C_OURS2),
        ("torch.softmax eager", d["per_call_us.torch_loop_ms"], C_BASE),
        ("torch.softmax + Graph", d["per_call_us.torch_graph_ms"], C_NEUT),
        ("Triton + Graph 重放", d["per_call_us.graph_replay_ms"], C_OURS),
    ]
    fig, ax = plt.subplots(figsize=(8, 3.9), facecolor="white")
    ys = range(len(rows))[::-1]
    ax.barh(list(ys), [r[1][0] for r in rows], height=0.62,
            color=[r[2] for r in rows], xerr=[r[1][1] for r in rows],
            ecolor=INK, error_kw={"lw": 1})
    for y, r in zip(ys, rows):
        ax.text(r[1][0] * 1.06, y, f"{r[1][0]:.2f} µs", va="center",
                fontsize=10, color=INK)
    ax.set_yticks(list(ys), [r[0] for r in rows])
    ax.set_xscale("log")
    ax.set_xlim(1, 90)
    ax.set_xlabel("每次调用成本(µs,对数轴,1024² softmax ×100 调用)",
                  fontsize=10, color=INK)
    ax.set_title("CUDA Graph 把 Triton 小核 launch 塌缩 11.6×(36.2→3.11µs),反超 torch",
                 fontsize=12, color=INK, pad=12)
    style_ax(ax)
    footnote(fig, "data/derived/exp-t05_stability_3rounds.csv")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGS / "fig3_launch_cudagraph.png", dpi=240, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    print("wrote:", *sorted(p.name for p in FIGS.glob("fig*.png")))
