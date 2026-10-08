"""Second follow-up study (C1-C3) and family H: every number reported in the manuscript.

Usage: python followup2_famH.py <path-to-CoFormer-folder> [<tables-dir>]
       -> followup2_famH_numbers.json (next to this script), tab_fs2.tex, tab_famh.tex

Rules (fixed before the runs; outputs/followup2.json and outputs/followup_h.json):
  FS2: one-sided exact sign-flip test on 12 per-seed differences in ID excess, Holm over {C1, C2, C3},
       SUPPORTED iff Holm p < 0.05 and mean difference >= 0.03.
  H:   one-sided exact sign-flip tests, Holm over {H_full, H_scarce}, same rule; a member is
       UNTESTABLE (ceiling) if the median frozen share of the maximum excess is >= 0.95.
"""
import sys, os, json, glob, itertools, numpy as np, pandas as pd
from scipy import stats, optimize

R = sys.argv[1]; TD = sys.argv[2] if len(sys.argv) > 2 else None
HERE = os.path.dirname(os.path.abspath(__file__))
FP = "99f69cb71d"
F2 = pd.read_csv(f"{R}/checkpoints/results_fs2_{FP}.csv")
H = pd.read_csv(f"{R}/checkpoints/results_famH_{FP}.csv")
PS = pd.read_csv(f"{R}/checkpoints/results_partial_{FP}.csv")
P3 = pd.read_csv(f"{R}/checkpoints/results_planIII_{FP}.csv")
A = pd.read_csv(f"{R}/checkpoints/results_probeA_{FP}.csv")
p05 = PS[PS.faith_fraction.round(4) == 0.005]
spec2 = json.load(open(f"{R}/outputs/followup2.json"))
specH = json.load(open(f"{R}/outputs/followup_h.json"))
FMAX = 1 - (3 / 16) / 2


def signflip(d, alt="two"):
    d = np.asarray(d, float)
    if np.all(np.abs(d) < 1e-12): return 1.0
    s = np.array(list(itertools.product([-1.0, 1.0], repeat=len(d)))); m = (s * d).mean(1); o = d.mean()
    return float((m >= o - 1e-12).mean()) if alt == "greater" else float((np.abs(m) >= abs(o) - 1e-12).mean())

def tci(d, lev):
    d = np.asarray(d, float); n = len(d); sd = d.std(ddof=1)
    if sd == 0: return [float(d.mean()), float(d.mean())]
    h = stats.t.ppf(.5 + lev / 2, n - 1) * sd / np.sqrt(n)
    return [float(d.mean() - h), float(d.mean() + h)]

def mde(d):
    d = np.asarray(d, float); n = len(d); sd = d.std(ddof=1)
    if sd == 0: return 0.0
    tc = stats.t.ppf(.975, n - 1)
    f = lambda es: (np.nan_to_num(stats.nct.sf(tc, n - 1, es * np.sqrt(n)), nan=1.0)
                    + np.nan_to_num(stats.nct.cdf(-tc, n - 1, es * np.sqrt(n)), nan=0.0) - 0.8)
    return float(optimize.brentq(f, 1e-6, 20) * sd)

def vals(df, cond, seeds, m="faith_excess_id"):
    return df[df.condition == cond].set_index("seed")[m].reindex(seeds)

def compare(xa, xb, extra=True):
    ok = ~(xa.isna() | xb.isna()); x, y = xa[ok].values, xb[ok].values; d = x - y
    sa, sb = x > .30, y > .30; bb, cc = int((sa & ~sb).sum()), int((~sa & sb).sum())
    r = dict(n=int(len(d)), mean_a=float(x.mean()), mean_b=float(y.mean()), diff=float(d.mean()),
             ci95=tci(d, .95), ci90=tci(d, .90), p_one=signflip(d, "greater"), p_two=signflip(d),
             mde80=mde(d), succ_a=int(sa.sum()), succ_b=int(sb.sum()),
             p_mcnemar_one=float(stats.binom.sf(bb - 1, bb + cc, .5)) if bb + cc else 1.0,
             mean_succ_a=float(x[sa].mean()) if sa.any() else None,
             mean_succ_b=float(y[sb].mean()) if sb.any() else None,
             mean_fail_a=float(x[~sa].mean()) if (~sa).any() else None,
             mean_fail_b=float(y[~sb].mean()) if (~sb).any() else None,
             n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()))
    r["benefit_excluded"] = bool(r["ci90"][1] < 0.03)
    return r

def holm(rows, m):
    ps = sorted((r["p_one"], i) for i, r in enumerate(rows)); run = 0.0
    for rank, (p, i) in enumerate(ps):
        run = max(run, min(1.0, (m - rank) * p)); rows[i]["p_holm"] = run

out = {"fs2": {"sha256": spec2["sha256"], "written_utc": spec2["written_utc"], "n_rows": int(len(F2)),
               "torch": sorted(F2.torch_version.astype(str).unique().tolist())},
       "H": {"sha256": specH["sha256"], "written_utc": specH["written_utc"],
             "generator_sha256": specH["plan"]["generator_sha256"], "n_rows": int(len(H))}}

# ------------------------------------------------------------------ follow-up study 2
S12 = list(range(12)); E = lambda e: F2[F2.experiment == e]
a2 = P3[P3.experiment == "A2"]
conf = {"C1": compare(vals(E("C1"), "coformer", S12), vals(E("C1"), "frozen", S12)),
        "C2": compare(vals(E("C2_Q1"), "rl_ans_ps", S12), vals(p05, "frozen", S12)),
        "C3": compare(vals(E("C3"), "ans_st_ps", S12), vals(a2, "ans_st_ps", S12))}
rows = list(conf.values()); holm(rows, 3)
for k, r in conf.items():
    r["verdict"] = "supported" if (r["p_holm"] < .05 and r["diff"] >= .03) else "not supported"
out["fs2"]["confirmatory"] = conf
# C1 against the stored 12-epoch runs (same seeds, f = 0.005): what four times the budget changes
c1x = {}
for c in ("coformer", "frozen"):
    c1x[c] = compare(vals(E("C1"), c, S12), vals(p05, c, S12))
out["fs2"]["C1_vs_12ep"] = c1x
# C1 learning curves
cv = []
for r in E("C1").itertuples():
    for e, v in json.loads(r.learning_curve or "[]"):
        cv.append(dict(condition=r.condition, seed=int(r.seed), epoch=int(e), excess=float(v)))
cv = pd.DataFrame(cv)
piv = cv.pivot_table(index="epoch", columns="condition", values="excess")
out["fs2"]["C1_curve_mean"] = {c: {int(e): float(v) for e, v in piv[c].items()} for c in piv.columns}
g = cv.pivot_table(index=["condition", "seed"], columns="epoch", values="excess")
gain = (g[48] - g[36]).groupby(level=0).mean()
out["fs2"]["C1_gain_36_48"] = {k: float(v) for k, v in gain.items()}
out["fs2"]["C1_converged"] = bool((gain.abs() < 0.02).all())
out["fs2"]["C1_rising_runs_36_48"] = {c: int(((g[48] - g[36]).xs(c) > 0.02).sum()) for c in ("coformer", "frozen")}
# diagnostics
S6 = list(range(6))
out["fs2"]["diag"] = {
    "Q16ps_vs_frozen": compare(vals(E("C2_Q16ps"), "rl_ans_ps", S6), vals(p05, "frozen", S6)),
    "Q16ps_vs_Q1ps": compare(vals(E("C2_Q16ps"), "rl_ans_ps", S6), vals(E("C2_Q1"), "rl_ans_ps", S6)),
    "C3_vs_frozen": compare(vals(E("C3"), "ans_st_ps", S12), vals(p05, "frozen", S12)),
    "C2_vs_C3": compare(vals(E("C2_Q1"), "rl_ans_ps", S12), vals(E("C3"), "ans_st_ps", S12))}
q16 = E("C2_Q16")
a1 = P3[(P3.experiment == "A1") & (P3.condition == "ans_st")]
out["fs2"]["Q16_answers_only"] = dict(n=int(len(q16)), mean=float(q16.faith_excess_id.mean()),
    max=float(q16.faith_excess_id.max()), degenerate=int(q16.degenerate.astype(bool).sum()),
    density=float(q16.pred_density.mean()), a1_mean=float(a1.faith_excess_id.mean()),
    a1_degenerate=int(a1.degenerate.astype(bool).sum()), a1_n=int(len(a1)))
out["fs2"]["degenerate"] = {e: int(E(e).degenerate.astype(bool).sum()) for e in sorted(F2.experiment.unique())}
out["fs2"]["minutes_median"] = {f"{e}|{c}": float(d.train_secs.median() / 60)
                                for (e, c), d in F2.groupby(["experiment", "condition"])}
out["fs2"]["gpu_hours"] = float(F2.train_secs.sum() / 3600)
# secondary: CSat-norm for the confirmatory pairs
sec = {}
for k, (ea, ca, eb, cb) in {"C1": (E("C1"), "coformer", E("C1"), "frozen"),
                            "C2": (E("C2_Q1"), "rl_ans_ps", p05, "frozen"),
                            "C3": (E("C3"), "ans_st_ps", a2, "ans_st_ps")}.items():
    sec[k] = compare(vals(ea, ca, S12, "csat_norm_id"), vals(eb, cb, S12, "csat_norm_id"))
out["fs2"]["csat_norm_id"] = sec

# ------------------------------------------------------------------ family H
lv = json.load(open(f"{R}/outputs/famH_level.json"))
out["H"]["pilot"] = lv["pilot"]; out["H"]["level"] = lv["selected_level"]
pil = H[H.experiment.str.startswith("H_pilot")]
out["H"]["pilot_runs"] = [dict(level=int(r.level), excess=float(r.faith_excess_id),
                               para=float(r.faith_excess_paraphrase), minutes=float(r.train_secs / 60))
                          for r in pil.sort_values("level").itertuples()]
st = {}
for f in sorted(glob.glob(f"{R}/data_cache/dataset_famH_*_n20000-2000-2000_L*.npz")):
    z = np.load(f); st[f[-6:-4]] = json.loads(str(z["stats"]))
out["H"]["generation"] = st
main = H[H.experiment.isin(["H_full", "H_scarce"])]
out["H"]["at_max"] = {f"{e}|{c}": dict(n=int(len(d)), n_ge_095=int((d.faith_excess_id >= .95 * FMAX).sum()),
                                        n_ge_0999=int((d.faith_excess_id >= .999 * FMAX).sum()),
                                        min=float(d.faith_excess_id.min()), max=float(d.faith_excess_id.max()))
                      for (e, c), d in main.groupby(["experiment", "condition"])}
out["H"]["degenerate"] = int(main.degenerate.astype(bool).sum())
confH, secH = {}, {}
for h, n in (("H_full", 16), ("H_scarce", 12)):
    d = main[main.experiment == h]; S = list(range(n))
    r = compare(vals(d, "coformer", S), vals(d, "frozen", S))
    r["share_b_median"] = float(np.median(vals(d, "frozen", S).values)) / FMAX
    r["untestable"] = bool(r["share_b_median"] >= .95)
    confH[h] = r
    for m in ("faith_excess_paraphrase", "csat_norm_paraphrase", "ood_retention_para", "csat_norm_id"):
        secH[f"{h}|{m}"] = compare(vals(d, "coformer", S, m), vals(d, "frozen", S, m))
testable = [r for r in confH.values() if not r["untestable"]]
holm(testable, 2)
for r in confH.values():
    r["verdict"] = ("untestable (ceiling)" if r["untestable"] else
                    "supported" if (r["p_holm"] < .05 and r["diff"] >= .03) else "not supported")
out["H"]["confirmatory"] = confH; out["H"]["secondary"] = secH
cvh = []
for r in main.itertuples():
    for e, v in json.loads(r.learning_curve or "[]"):
        cvh.append(dict(experiment=r.experiment, condition=r.condition, seed=int(r.seed), epoch=int(e), excess=float(v)))
cvh = pd.DataFrame(cvh)
pv = cvh.pivot_table(index="epoch", columns=["experiment", "condition"], values="excess")
out["H"]["curve_mean"] = {f"{e}|{c}": {int(k): float(v) for k, v in pv[(e, c)].items()} for e, c in pv.columns}
out["H"]["minutes_median"] = float(main.train_secs.median() / 60)
out["H"]["gpu_hours"] = float(H.train_secs.sum() / 3600)
# family-A reference for the paraphrase split (stored main matrix, full supervision)
out["H"]["familyA_ref"] = {c: dict(para=float(A[A.condition == c].faith_excess_paraphrase.mean()),
                                   csat_para=float(A[A.condition == c].csat_norm_paraphrase.mean()),
                                   excess_id=float(A[A.condition == c].faith_excess_id.mean()), n=int((A.condition == c).sum()))
                           for c in ("frozen", "coformer")}
out["fmax"] = FMAX

# seeds needed for 80% power (two-sided paired t test) at the observed effect and SD of the open contrasts
def _sd(r):
    return (r["ci95"][1] - r["ci95"][0]) / 2 / stats.t.ppf(.975, r["n"] - 1) * np.sqrt(r["n"])
def _nseeds(delta, sd):
    for n in range(3, 1000):
        tc = stats.t.ppf(.975, n - 1); nc = delta / sd * np.sqrt(n)
        if stats.nct.sf(tc, n - 1, nc) + stats.nct.cdf(-tc, n - 1, nc) >= .8: return n
_p3 = json.load(open(os.path.join(HERE, "plan3_numbers.json")))["diag"]
out["power"] = {k: dict(diff=r["diff"], sd=_sd(r), n_obs=r["n"], seeds_80=_nseeds(abs(r["diff"]), _sd(r)))
                for k, r in {"K_ST": _p3["K_ST"], "K_soft": _p3["K_soft"],
                             "H_scarce_csat_para": secH["H_scarce|csat_norm_paraphrase"],
                             "H_scarce_retention": secH["H_scarce|ood_retention_para"]}.items()}
json.dump(out, open(os.path.join(HERE, "followup2_famH_numbers.json"), "w"), indent=1)

# ------------------------------------------------------------------ tables (bodies, as tab_plan3)
def sg(x): return ("$-$" if x < 0 else "$+$") + f"{abs(x):.3f}"
def pv(p): return "$<$0.001" if p < 0.001 else f"{p:.3f}"
def row(label, r, succ=True):
    s = (f"{label} & {r['n']} & {r['mean_a']:.3f} & {r['mean_b']:.3f} & {sg(r['diff'])} & "
         f"[{sg(r['ci95'][0])}, {sg(r['ci95'][1])}] & {pv(r['p_two'])}")
    return s + (f" & {r['succ_a']}/{r['succ_b']} \\\\" if succ else " \\\\")
if TD:
    os.makedirs(TD, exist_ok=True)
    d = out["fs2"]["diag"]; x = out["fs2"]["C1_vs_12ep"]
    L = [row(r"C1: \cond{coformer} vs.\ \cond{frozen}, 48 epochs", conf["C1"]),
         row(r"C2: RLOO, $Q=1$, vs.\ \cond{frozen}", conf["C2"]),
         row(r"C3: A2 shuffled vs.\ A2 real labels", conf["C3"]),
         r"\midrule",
         row(r"C2 diag.: RLOO, $Q=16$, vs.\ \cond{frozen}", d["Q16ps_vs_frozen"]),
         row(r"C2 diag.: RLOO, $Q=16$ vs.\ $Q=1$", d["Q16ps_vs_Q1ps"]),
         row(r"C3 diag.: A2 shuffled vs.\ \cond{frozen}", d["C3_vs_frozen"]),
         row(r"C1 diag.: \cond{coformer}, 48 vs.\ 12 epochs", x["coformer"]),
         row(r"C1 diag.: \cond{frozen}, 48 vs.\ 12 epochs", x["frozen"])]
    open(os.path.join(TD, "tab_fs2.tex"), "w").write("\n".join(L) + "\n")
    names = [("faith_excess_paraphrase", r"$\Fex$, paraphrase"), ("csat_norm_paraphrase", r"$\csatn$, paraphrase"),
             ("ood_retention_para", "paraphrase retention")]
    L = []
    for h, nm in (("H_full", r"H$_\text{full}$"), ("H_scarce", r"H$_\text{scarce}$")):
        L.append(row(nm + r": $\Fex$, ID (primary)", confH[h], succ=False))
        for m, mn in names:
            L.append(row(nm + ": " + mn, secH[f"{h}|{m}"], succ=False))
        if h == "H_full": L.append(r"\midrule")
    open(os.path.join(TD, "tab_famh.tex"), "w").write("\n".join(L) + "\n")

print(json.dumps({k: {kk: (vv if not isinstance(vv, dict) else "...") for kk, vv in v.items()} if isinstance(v, dict) else v
                  for k, v in out.items()}, indent=1)[:3000])
