"""Publication figures for sections/results.tex.
Usage: python make_figures.py <path-to-CoFormer-folder> <output-dir>
Palette: validated categorical slots 1-4 (light surface); every series also has a
distinct marker shape, and every figure has a corresponding table in the text."""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

R = sys.argv[1] if len(sys.argv) > 1 else "."
OUTD = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUTD, exist_ok=True)
# Libertinus Serif (SIL OFL), converted to TrueType (analysis/fonts) so that the
# PDFs embed a clean Type 42 font; falls back to the TeX Live OTF files.
_here = os.path.dirname(os.path.abspath(__file__))
for stem in ("LibertinusSerif-Regular", "LibertinusSerif-Italic", "LibertinusSerif-Bold"):
    for cand in (os.path.join(_here, "fonts", stem + ".ttf"),
                 "/usr/share/texlive/texmf-dist/fonts/opentype/public/libertinus-fonts/" + stem + ".otf"):
        if os.path.exists(cand):
            fm.fontManager.addfont(cand); break
plt.rcParams.update({
    "font.family": "Libertinus Serif", "mathtext.fontset": "custom",
    "mathtext.rm": "Libertinus Serif", "mathtext.it": "Libertinus Serif:italic",
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6,
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "savefig.dpi": 300, "pdf.fonttype": 42,
})
INK2, INK3 = "#52514e", "#8a8984"
COL = {"frozen": "#2a78d6", "coformer": "#eb6834", "soft_frozen": "#1baf7a", "no_stgs": "#eda100"}
MRK = {"frozen": "o", "coformer": "s", "soft_frozen": "^", "no_stgs": "D"}
# greyscale-safe encodings: line style per condition, filled (ST) vs hollow (soft) markers
LS = {"frozen": "-", "coformer": (0, (5, 2)), "soft_frozen": (0, (1, 1.2)), "no_stgs": (0, (5, 1.5, 1, 1.5))}
HOLLOW = {"frozen": False, "coformer": False, "soft_frozen": True, "no_stgs": True}
LAB = {"frozen": "frozen (ST, detached)", "coformer": "coformer (ST, routed)",
       "soft_frozen": "soft_frozen (soft, detached)", "no_stgs": "no_stgs (soft, routed)"}
FACT = ["frozen", "coformer", "soft_frozen", "no_stgs"]
CK, OUT = f"{R}/checkpoints", f"{R}/outputs"

def mean_ci(v, n=10000, seed=0):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(v, len(v)).mean() for _ in range(n)])
    return v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def save(fig, name):
    fig.savefig(f"{OUTD}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUTD}/{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

# ---------------------------------------------------------------- Figure: OOD strip
fam = {"A": pd.read_csv(f"{CK}/results_probeA_99f69cb71d.csv"),
       "B": pd.read_csv(f"{CK}/results_probeB_99f69cb71d.csv")}
fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.35), sharey=False)
panels = [("A", "csat_norm_paraphrase", "Family A, paraphrase OOD"),
          ("A", "csat_norm_scale", "Family A, scale OOD"),
          ("B", "csat_norm_paraphrase", "Family B, paraphrase OOD"),
          ("B", "csat_norm_scale", "Family B, scale OOD")]
rng = np.random.default_rng(1)
for ax, (f, m, title) in zip(axes, panels):
    df = fam[f]
    for i, c in enumerate(FACT):
        v = df[df.condition == c].sort_values("seed")[m].values
        x = i + rng.uniform(-0.17, 0.17, len(v))
        ax.scatter(x, v, s=11, color=COL[c], alpha=0.7, marker=MRK[c], linewidths=0, zorder=2)
        mu, lo, hi = mean_ci(v)
        ax.errorbar(i, mu, yerr=[[mu - lo], [hi - mu]], fmt=MRK[c], ms=5.5, color=COL[c],
                    mec="#fcfcfb", mew=0.8, elinewidth=1.4, capsize=0, zorder=3)
    ax.set_xticks(range(4)); ax.set_xticklabels(["frozen", "coformer", "soft\\_frozen".replace("\\", ""), "no_stgs"],
                                              rotation=35, ha="right")
    ax.set_title(title, color="#0b0b0b"); ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.6, 3.6)
axes[0].set_ylabel(r"CSat$^{\mathrm{norm}}$ (chance $=0$)")
fig.tight_layout(w_pad=0.9)
save(fig, "fig_ood")

# ---------------------------------------------------------------- Figure: partial supervision
P = pd.read_csv(f"{OUT}/partial_supervision.csv")
A = fam["A"]
POS = {0.0: 0, 0.001: 1, 0.005: 2, 0.02: 3, 1.0: 4}
TICK = ["0", "0.001", "0.005", "0.02", "1"]
ENDPT0 = {"coformer": "no_faith", "no_stgs": "soft_no_faith"}   # f=0 exists only for routed cells
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
off = {"frozen": -0.12, "coformer": -0.04, "soft_frozen": 0.04, "no_stgs": 0.12}
for ax, m, yl in [(axes[0], "faith_excess_id", r"$\mathrm{F}_1^{\mathrm{ex}}$ (ID)"),
                  (axes[1], "csat_norm_id", r"CSat$^{\mathrm{norm}}$ (ID)")]:
    for c in FACT:
        xs, mus, los, his = [], [], [], []
        for f in (0.001, 0.005, 0.02, 1.0):
            if f == 1.0:
                v = A[A.condition == c][m].values
            else:
                v = P[(P.faith_fraction == f) & (P.condition == c)][m].values
            x = POS[f] + off[c]
            ax.scatter(x + rng.uniform(-0.025, 0.025, len(v)), v, s=8, color=COL[c], alpha=0.6,
                       marker=MRK[c], linewidths=0, zorder=2)
            mu, lo, hi = mean_ci(v)
            xs.append(x); mus.append(mu); los.append(mu - lo); his.append(hi - mu)
        ax.errorbar(xs, mus, yerr=[los, his], fmt=MRK[c], ls=LS[c], color=COL[c], ms=5, lw=1.3,
                    mfc="white" if HOLLOW[c] else COL[c], mec=COL[c], mew=1.1,
                    elinewidth=1.0, capsize=0, label=LAB[c], zorder=3)
    # f = 0: only the solver-only ablations exist; drawn as separate, unconnected symbols
    for abl, mk, lab, dx in [("no_faith", "X", "no_faith (ST, routed, f=0)", -0.08),
                             ("soft_no_faith", "P", "soft_no_faith (soft, routed, f=0)", 0.08)]:
        v = A[A.condition == abl][m].values
        ax.scatter(POS[0.0] + dx + rng.uniform(-0.025, 0.025, len(v)), v, s=8, color="#0b0b0b", alpha=0.5,
                   marker=mk, linewidths=0, zorder=2)
        mu, lo, hi = mean_ci(v)
        ax.errorbar([POS[0.0] + dx], [mu], yerr=[[mu - lo], [hi - mu]], fmt=mk, color="#0b0b0b", ms=6,
                    mfc="white" if abl.startswith("soft") else "#0b0b0b", mec="#0b0b0b", mew=1.0,
                    elinewidth=1.0, capsize=0, label=lab, zorder=3)
    ax.axvline(0.5, color=INK3, lw=0.6, zorder=1)
    ax.set_xticks(range(5)); ax.set_xticklabels(TICK)
    ax.set_xlabel(r"fraction $f$ of training instances with per-slot supervision")
    ax.set_ylabel(yl); ax.grid(axis="x", visible=False)
axes[0].axhline(0.30, color=INK3, lw=0.7, ls=(0, (3, 2)), zorder=1)
axes[0].text(4.35, 0.315, "success\nthreshold", color=INK2, fontsize=7, ha="right", va="bottom")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.13),
           handlelength=2.6, columnspacing=1.2)
fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.5)
save(fig, "fig_partial")

# ---------------------------------------------------------------- Figure: refinement depth
S = pd.read_csv(f"{OUT}/hparam_sensitivity.csv")
fig, ax = plt.subplots(figsize=(3.3, 2.3))
# no_stgs first (filled, larger), coformer on top with open markers: the two coincide at T=50 and T=100
for knob, c, lab, ms, fc in [("refine_steps[no_stgs]", "no_stgs", "no_stgs", 7, None),
                             ("refine_steps", "coformer", "coformer", 5, "#fcfcfb")]:
    s = S[S.knob == knob].sort_values("value")
    ax.plot(s.value, s.acc_id, "-" + MRK[c], color=COL[c], ms=ms, lw=1.6 if fc is None else 1.2,
            mfc=fc if fc else COL[c], mec=COL[c] if fc else "#fcfcfb", mew=1.1 if fc else 0.6, label=lab, zorder=3 if fc else 2)
for x, y in [(10, 0.075), (100, 0.427), (200, 0.493)]:
    ax.annotate(f"{y:.3f}", (x, y), xytext=(5, -3), textcoords="offset points", fontsize=7.5, color=INK2)
ax.axvline(50, color=INK3, lw=0.7, ls=(0, (3, 2)))
ax.text(52, 0.04, "production $T$", color=INK2, fontsize=7, rotation=90, va="bottom")
ax.set_xscale("log"); ax.set_xticks([10, 20, 30, 50, 100, 200]); ax.set_xticklabels(["10", "20", "30", "50", "100", "200"])
ax.minorticks_off()
ax.set_xlabel("refinement steps $T$"); ax.set_ylabel("exact solve Acc (ID)")
ax.set_ylim(0, 0.56); ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); save(fig, "fig_refine")

# ---------------------------------------------------------------- Figure: natural-language benchmarks
RC = pd.read_csv(f"{OUT}/real_comparison_full.csv")
SYS = [("SATLM-frozen", "SatLM-style"), ("LogicLM-frozen", "Logic-LM-style"), ("DiLA-frozen", "DiLA-style"),
       ("CoFormer-frozen", "frozen"), ("CoFormer-coformer", "coformer"), ("CoFormer-no_faith", "no_faith"),
       ("CoFormer-no_stgs", "no_stgs"), ("CoFormer-soft_frozen", "soft_frozen")]
GC = {"prompted": "#2a78d6", "trained": "#eb6834"}
GM = {"prompted": "o", "trained": "s"}
floor = RC[RC.system == "majority_floor"].accuracy.values
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
for ax, m, xl in [(axes[0], "accuracy", "three-way accuracy (held-out)"),
                  (axes[1], "abstain_rate", "abstention rate (share of Unknown)")]:
    if m == "accuracy":
        ax.axvspan(floor.min(), floor.max(), color="#e4e3df", zorder=0, lw=0)
        ax.axvline(floor.mean(), color=INK3, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.text(floor.mean(), -0.95, "majority-class floor", fontsize=6.8, color=INK2, ha="center", va="bottom")
    for i, (s, lab) in enumerate(SYS):
        grp = "prompted" if not s.startswith("CoFormer") else "trained"
        v = RC[RC.system == s].sort_values("seed")[m].values
        y = len(SYS) - 1 - i
        ax.scatter(v, y + rng.uniform(-0.12, 0.12, len(v)), s=12, color=GC[grp], alpha=0.65,
                   marker=GM[grp], linewidths=0, zorder=2)
        ax.plot([v.mean()], [y], GM[grp], color=GC[grp], ms=6, mec="#fcfcfb", mew=0.8, zorder=3)
    ax.set_xlabel(xl); ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(len(SYS))); axes[0].set_yticklabels([l for _, l in SYS][::-1])
axes[0].set_xlim(0.2, 0.55); axes[1].set_xlim(0.3, 1.02)
axes[0].set_xticks([0.25, 0.35, 0.45]); axes[1].set_xticks([0.4, 0.6, 0.8, 1.0]); axes[0].set_ylim(-1.05, 7.5)
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], marker="o", ls="", color=GC["prompted"]), Line2D([], [], marker="s", ls="", color=GC["trained"])],
           ["prompted formaliser (frozen backbone)", "trained variant (LoRA adapters + clause projector)"],
           loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.05))
fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.2)
save(fig, "fig_real")
print("figures written to", OUTD)
