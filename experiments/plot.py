"""Turn results/*.json into report figures (report/figures/*.pdf) and LaTeX
tables (report/figures/tables.tex).   python experiments/plot.py"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "report", "figures")
os.makedirs(FIG, exist_ok=True)

# reference categorical palette, fixed slot order
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "lines.markersize": 3.5, "legend.frameon": False,
    "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})


def load(exp):
    out = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(RES, exp, "*.json"))):
        name = os.path.basename(p)[:-5]
        key = re.sub(r"_s\d+$", "", name)
        with open(p) as f:
            out[key].append(json.load(f))
    return out


def agg(runs, fn):
    v = np.array([fn(r) for r in runs], float)
    return float(np.nanmean(v)), float(np.nanstd(v)) if len(v) > 1 else 0.0


def acc(k):
    return lambda r: r["accuracy"][k]


def lat(k):
    return lambda r: r["latency"][k]


def ovh(k):
    return lambda r: r["overhead"][k]


def per_obj_min(r):
    a = r["accuracy"]
    return a["idsw"] / max(a["gt"] * r["config"]["dt"] / 60.0, 1e-9)


def per_obj_min_online(r):
    a = r["accuracy"]
    return a["idsw_online"] / max(a["gt"] * r["config"]["dt"] / 60.0, 1e-9)


def line(ax, xs, data, color, label, marker="o", ls="-"):
    m = np.array([d[0] for d in data])
    s = np.array([d[1] for d in data])
    ax.plot(xs, m, color=color, marker=marker, ls=ls, label=label)
    if s.any():
        ax.fill_between(xs, m - s, m + s, color=color, alpha=0.15, lw=0)


TEX = []


def tex(s):
    TEX.append(s)


# ------------------------------------------------------------------ scalability
def fig_scale():
    d = load("scale")
    if not d:
        return
    Ms = sorted({int(k.split("_M")[1]) for k in d})
    Md = [M for M in Ms if f"dist_M{M}" in d]
    Mc_all = [M for M in Ms if f"cent_M{M}" in d]
    Mc = [M for M in Mc_all if not np.isnan(agg(d[f"cent_M{M}"], lat("p50"))[0])]
    sat = [M for M in Mc_all if M not in Mc]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.85))
    ax = axs[0]
    line(ax, Md, [agg(d[f"dist_M{M}"], acc("idf1")) for M in Md], BLUE, "distributed (ours)")
    line(ax, Mc, [agg(d[f"cent_M{M}"], acc("idf1")) for M in Mc], ORANGE, "centralized", marker="s")
    ax.set_xscale("log"); ax.set_ylim(0, 1.02); ax.set_xlabel("cameras M"); ax.set_ylabel("IDF1")
    for M in sat:
        ax.annotate("centralized\nsaturated", (M, 0.62), ha="center", fontsize=6, color=ORANGE)
    ax.set_title("(a) identity accuracy"); ax.legend(loc="lower left")
    ax = axs[1]
    line(ax, Md, [agg(d[f"dist_M{M}"], lat("p50")) for M in Md], BLUE, "dist. p50")
    line(ax, Md, [agg(d[f"dist_M{M}"], lat("p99")) for M in Md], BLUE, "dist. p99", ls="--", marker="^")
    line(ax, Mc, [agg(d[f"cent_M{M}"], lat("p50")) for M in Mc], ORANGE, "cent. p50", marker="s")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("cameras M"); ax.set_ylabel("end-to-end latency (s)")
    for M in sat:
        ax.annotate(r"$\infty$", (M, 3.0), ha="center", fontsize=9, color=ORANGE)
    ax.set_title("(b) latency"); ax.legend(loc="upper left")
    ax = axs[2]
    line(ax, Md, [agg(d[f"dist_M{M}"], ovh("center_cpu_max")) for M in Md], BLUE, "busiest center")
    line(ax, Md, [agg(d[f"dist_M{M}"], ovh("agg_cpu")) for M in Md], AQUA, "aggregator", marker="D")
    line(ax, Mc_all, [agg(d[f"cent_M{M}"], ovh("center_cpu_max")) for M in Mc_all], ORANGE, "central node", marker="s")
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_xlabel("cameras M"); ax.set_ylabel("CPU utilization")
    ax.set_title("(c) busiest-node load"); ax.legend(loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "scale.pdf")); fig.savefig(os.path.join(FIG, "scale.png"), dpi=170)
    plt.close(fig)
    # table
    rows = []
    for M in Ms:
        for kind, lab in (("dist", "Distributed"), ("cent", "Centralized")):
            k = f"{kind}_M{M}"
            if k not in d:
                continue
            r = d[k]
            in_kb = agg(r, lambda x: (x["overhead"]["center_in_max_Bps"]) / 1e3)[0]
            if np.isnan(agg(r, lat("p50"))[0]):
                rows.append(f"{M:,} & {lab} & \\multicolumn{{5}}{{c}}{{saturated: no step reported within the run}} & "
                            f"{agg(r, ovh('center_cpu_max'))[0] * 100:.0f}\\% & {in_kb:,.0f} \\\\")
                continue
            rows.append(f"{M:,} & {lab} & {agg(r, acc('mota'))[0]:.3f} & {agg(r, acc('idf1'))[0]:.3f} & "
                        f"{agg(r, per_obj_min)[0]:.2f} & {agg(r, lat('p50'))[0] * 1e3:.0f} & "
                        f"{agg(r, lat('p99'))[0] * 1e3:.0f} & {agg(r, ovh('center_cpu_max'))[0] * 100:.0f}\\% & "
                        f"{in_kb:,.0f} \\\\")
    tex("\\newcommand{\\TabScaleRows}{" + "\n".join(rows) + "}")
    # latency breakdown at M=1000
    if "dist_M1000" in d:
        parts = defaultdict(list)
        for r in d["dist_M1000"]:
            for k, v in r["latency"]["parts"].items():
                parts[k].append(v)
        tex("\\newcommand{\\LatParts}{" + " & ".join(f"{np.mean(v) * 1e3:.1f}" for v in parts.values()) + "}")
        tex("\\newcommand{\\LatPartNames}{" + " & ".join(parts.keys()) + "}")


# ------------------------------------------------------------------ region size
def fig_region():
    d = load("region")
    if not d:
        return
    Ss = sorted(int(k[1:]) for k in d)
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.7))
    ho = [agg(d[f"S{S}"], lambda r: r["overhead"]["counters"].get("handoff_accepts", 0) / r["config"]["duration"])
          for S in Ss]
    line(axs[0], Ss, ho, BLUE, "")
    axs[0].set_ylabel("handoffs / s"); axs[0].set_title("(a) ownership transfers")
    line(axs[1], Ss, [agg(d[f"S{S}"], ovh("center_cpu_max")) for S in Ss], BLUE, "busiest center")
    line(axs[1], Ss, [agg(d[f"S{S}"], ovh("center_cpu_mean")) for S in Ss], AQUA, "mean center", marker="D")
    axs[1].set_ylabel("CPU utilization"); axs[1].set_title("(b) center load"); axs[1].legend()
    line(axs[2], Ss, [agg(d[f"S{S}"], acc("idf1")) for S in Ss], BLUE, "IDF1")
    line(axs[2], Ss, [agg(d[f"S{S}"], acc("mota")) for S in Ss], AQUA, "MOTA", marker="D")
    axs[2].set_title("(c) accuracy"); axs[2].legend()
    for ax in axs:
        ax.set_xscale("log"); ax.set_xticks(Ss); ax.set_xticklabels(Ss); ax.set_xlabel("target region size S")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "region.pdf")); fig.savefig(os.path.join(FIG, "region.png"), dpi=170)
    plt.close(fig)


# ------------------------------------------------------------------ deadline / network / noise
def fig_sweeps():
    dl = load("deadline")
    nw = load("network")
    nz = load("noise")
    if not (dl or nw or nz):
        return
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 1.75))
    if dl:
        Ws = sorted(int(k[1:]) for k in dl if re.fullmatch(r"W\d+", k))
        ax = axs[0]
        line(ax, Ws, [agg(dl[f"W{W}"], lat("p50")) for W in Ws], BLUE, "latency p50 (in s)")
        line(ax, Ws, [agg(dl[f"W{W}"], acc("mota")) for W in Ws], AQUA, "MOTA", marker="D")
        line(ax, Ws, [agg(dl[f"W{W}"], ovh("late_det_rate")) for W in Ws], ORANGE, "late-drop rate", marker="s")
        ax.set_xlabel("center deadline W (ms)"); ax.set_title("(a) deadline trade-off"); ax.legend(loc="center right")
        ax.set_ylim(0, 1.02)
    if nw:
        ps = sorted(int(k[4:]) for k in nw if k.startswith("loss"))
        ax = axs[1]
        line(ax, ps, [agg(nw[f"loss{p}"], acc("mota")) for p in ps], AQUA, "MOTA", marker="D")
        line(ax, ps, [agg(nw[f"loss{p}"], acc("idf1")) for p in ps], BLUE, "IDF1")
        ax.set_xlabel("message loss (%)"); ax.set_title("(b) message loss"); ax.legend(loc="lower left")
        ax.set_ylim(0.5, 1.0)
        th = sorted(int(k[4:]) for k in nw if k.startswith("skew"))
        ax = axs[2]
        line(ax, th, [agg(nw[f"skew{t}"], acc("mota")) for t in th], AQUA, "MOTA", marker="D")
        line(ax, th, [agg(nw[f"skew{t}"], acc("idf1")) for t in th], BLUE, "IDF1")
        ax.set_xlabel(r"max clock offset $\theta_{max}$ (ms)"); ax.set_title("(c) clock skew"); ax.legend(loc="lower left")
        ax.set_ylim(0.5, 1.0)
    if nz:
        las = sorted(int(k[2:]) for k in nz if k.startswith("la"))
        ax = axs[3]
        line(ax, las, [agg(nz[f"la{x}"], acc("idf1")) for x in las], BLUE, "IDF1 vs look-alike %")
        pn = sorted(float(k[2:]) for k in nz if k.startswith("pn"))
        ax2x = [x * 100 / max(pn) * 0.5 for x in pn]
        ax.set_xlabel(r"look-alike fraction $\rho_{la}$ (%)"); ax.set_title("(d) appearance ambiguity")
        ax.set_ylim(0.5, 1.0)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "sweeps.pdf")); fig.savefig(os.path.join(FIG, "sweeps.png"), dpi=170)
    plt.close(fig)
    rows = []
    if nz:
        for x in sorted(float(k[2:]) for k in nz if k.startswith("pn")):
            key = "pn" + (str(int(x)) if x == int(x) else str(x))
            r = nz[key]
            rows.append(f"$\\times{x:g}$ & {agg(r, acc('mota'))[0]:.3f} & {agg(r, acc('idf1'))[0]:.3f} & "
                        f"{agg(r, acc('rmse'))[0]:.2f} \\\\")
    tex("\\newcommand{\\TabPosNoiseRows}{" + "\n".join(rows) + "}")
    if dl:
        Was = sorted(int(k[2:]) for k in dl if k.startswith("Wa"))
        rows = [f"{W} & {agg(dl[f'Wa{W}'], lat('p50'))[0] * 1e3:.0f} & {agg(dl[f'Wa{W}'], lat('p99'))[0] * 1e3:.0f} & "
                f"{agg(dl[f'Wa{W}'], lambda r: r['overhead']['late_reports'] / r['config']['duration'])[0]:.2f} & "
                f"{agg(dl[f'Wa{W}'], acc('mota'))[0]:.3f} \\\\" for W in Was]
        tex("\\newcommand{\\TabWaRows}{" + "\n".join(rows) + "}")


# ------------------------------------------------------------------ faults
def tab_faults():
    c = load("crash")
    if c:
        rows = []
        for tag, lab in (("rep", "Raft ($R{=}3$)"), ("norep", "no replication")):
            r = c.get(f"center_{tag}")
            if not r:
                continue
            rec = agg(r, lambda x: x["continuity"]["recovery_s"])
            kept = agg(r, lambda x: x["continuity"]["kept_frac"])
            kept10 = agg(r, lambda x: x["continuity_10s"]["kept_frac"])
            sw_b = agg(r, lambda x: x["acc_before"]["idsw_online"])
            sw_a = agg(r, lambda x: x["acc_after"]["idsw_online"])
            idf_a = agg(r, lambda x: x["acc_after"]["idf1"])
            rows.append(f"{lab} & {rec[0]:.2f}$\\pm${rec[1]:.2f} & {kept[0] * 100:.0f}\\% & {kept10[0] * 100:.0f}\\% & "
                        f"{sw_b[0]:.1f} & {sw_a[0]:.1f} & {idf_a[0]:.3f} \\\\")
        tex("\\newcommand{\\TabCenterRows}{" + "\n".join(rows) + "}")
        rows = []
        for rate in sorted(int(k[6:]) for k in c if k.startswith("member")):
            r = c[f"member{rate}"]
            ev = agg(r, lambda x: x["events"].get("crash", 0))
            rows.append(f"{rate} & {ev[0]:.0f} & {agg(r, acc('mota'))[0]:.3f} & {agg(r, acc('idf1'))[0]:.3f} & "
                        f"{agg(r, per_obj_min)[0]:.2f} & {agg(r, lambda x: x['events'].get('voter_change', 0))[0]:.1f} \\\\")
        tex("\\newcommand{\\TabMemberRows}{" + "\n".join(rows) + "}")
    a = load("agg")
    if a:
        rows = []
        for key, lab in (("agg_none", "no crash"), ("agg1", "down 1 s"), ("agg5", "down 5 s")):
            r = a.get(key)
            if not r:
                continue
            if key == "agg_none":
                rows.append(f"{lab} & -- & -- & {agg(r, acc('idf1'))[0]:.3f} & {agg(r, per_obj_min)[0]:.2f} \\\\")
                continue
            rb = agg(r, lambda x: x["rebuilds"][0][0] * 1e3 if x["rebuilds"] else np.nan)
            kept = agg(r, lambda x: x["id_preserved"]["frac"])
            rows.append(f"{lab} & {rb[0]:.0f} & {kept[0] * 100:.1f}\\% & {agg(r, acc('idf1'))[0]:.3f} & "
                        f"{agg(r, per_obj_min)[0]:.2f} \\\\")
        tex("\\newcommand{\\TabAggRows}{" + "\n".join(rows) + "}")
    ch = load("churn")
    if ch:
        rows = []
        for x in sorted(int(k[5:]) for k in ch):
            r = ch[f"churn{x}"]
            rows.append(f"{x / 10:g} & {agg(r, lambda q: q['events'].get('join', 0))[0]:.0f} / "
                        f"{agg(r, lambda q: q['events'].get('leave', 0))[0]:.0f} & {agg(r, acc('mota'))[0]:.3f} & "
                        f"{agg(r, acc('idf1'))[0]:.3f} & {agg(r, per_obj_min)[0]:.2f} \\\\")
        tex("\\newcommand{\\TabChurnRows}{" + "\n".join(rows) + "}")


def tab_ablation():
    d = load("ablation")
    if not d:
        return
    rows = []
    for M in (300, 1000):
        for key, lab in (("full", "full protocol"), ("noglobalreid", "$-$ global re-id"),
                         ("nohandoff", "$-$ handoff"), ("nohandoff_noagg", "$-$ handoff, dedup, re-id")):
            r = d.get(f"{key}_M{M}")
            if not r:
                continue
            rows.append(f"{M:,} & {lab} & {agg(r, acc('mota'))[0]:.3f} & {agg(r, acc('idf1'))[0]:.3f} & "
                        f"{agg(r, per_obj_min_online)[0]:.2f} & {agg(r, per_obj_min)[0]:.2f} & "
                        f"{agg(r, lambda x: x['events'].get('merge', 0) / x['config']['duration'])[0]:.2f} \\\\")
        if M == 300:
            rows.append("\\midrule")
    tex("\\newcommand{\\TabAblationRows}{" + "\n".join(rows) + "}")


if __name__ == "__main__":
    fig_scale()
    fig_region()
    fig_sweeps()
    tab_faults()
    tab_ablation()
    with open(os.path.join(FIG, "tables.tex"), "w") as f:
        f.write("% generated by experiments/plot.py\n" + "\n".join(TEX) + "\n")
    print("wrote", FIG)
