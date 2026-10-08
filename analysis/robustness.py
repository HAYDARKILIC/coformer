"""Small-sample robustness analyses added after review (all post hoc).
Usage: python robustness.py <path-to-CoFormer-folder>  -> writes robustness_numbers.json

For every seed-paired contrast of the confirmatory families and for the post hoc OOD contrasts:
  * exact two-sided sign-flip permutation p-value (all 2^S sign patterns; statistic = mean difference)
  * t-based 90% interval of the mean difference and a one-sided non-superiority check at delta_min
    (the predicted-direction benefit is excluded at alpha = 0.05 iff the 90% bound lies inside -delta_min / +delta_min)
  * two-sided minimum detectable effect at 80% power from the observed SD of the differences (paired t)
  * 95% noncentral-t interval for d_z (Steiger & Fouladi inversion)
  * Clopper-Pearson interval for the H6 per-seed shares
  * H3 recomputed on paraphrase retention only (1 - ret_para), which removes the slot-tiled scale split
"""
import sys, os, json, itertools, numpy as np, pandas as pd
from scipy import stats, optimize

R = sys.argv[1] if len(sys.argv) > 1 else "."
CK, OUT = f"{R}/checkpoints", f"{R}/outputs"
A = pd.read_csv(f"{CK}/results_probeA_99f69cb71d.csv")
B = pd.read_csv(f"{CK}/results_probeB_99f69cb71d.csv")
EC = pd.read_csv(f"{OUT}/results_estimator_confirmatory.csv")
P = pd.read_csv(f"{OUT}/partial_supervision.csv")
RC = pd.read_csv(f"{OUT}/real_comparison_full.csv")


def vec(df, c, m, key="condition"):
    return df[df[key] == c].sort_values("seed")[m].values.astype(float)


def signflip_p(d):
    d = np.asarray(d, float)
    n = len(d)
    if n == 0 or np.all(np.abs(d) < 1e-12):
        return 1.0
    obs = abs(d.mean())
    signs = np.array(list(itertools.product([-1, 1], repeat=n)), float)
    means = np.abs((signs * d).mean(axis=1))
    return float((means >= obs - 1e-12).mean())


def t_ci(d, level):
    d = np.asarray(d, float); n = len(d); sd = d.std(ddof=1)
    if sd < 1e-12:
        return [float(d.mean())] * 2
    h = stats.t.ppf(0.5 + level / 2, n - 1) * sd / np.sqrt(n)
    return [float(d.mean() - h), float(d.mean() + h)]


def mde(d, power=0.8, alpha=0.05):
    d = np.asarray(d, float); n = len(d); sd = d.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    df = n - 1; tc = stats.t.ppf(1 - alpha / 2, df)
    def f(es):
        nc = es * np.sqrt(n)
        up = np.nan_to_num(stats.nct.sf(tc, df, nc), nan=1.0)
        lo = np.nan_to_num(stats.nct.cdf(-tc, df, nc), nan=0.0)
        return up + lo - power
    es = optimize.brentq(f, 1e-6, 20)
    return float(es * sd)


def dz_ci(d, level=0.95):
    d = np.asarray(d, float); n = len(d); sd = d.std(ddof=1)
    if sd < 1e-12:
        return None
    dz = d.mean() / sd; t_obs = dz * np.sqrt(n); df = n - 1
    F = lambda nc: float(np.nan_to_num(stats.nct.cdf(t_obs, df, nc), nan=0.0 if nc > t_obs else 1.0))
    B = abs(t_obs) + 40
    lo = optimize.brentq(lambda nc: F(nc) - (0.5 + level / 2), -B, B)
    hi = optimize.brentq(lambda nc: F(nc) - (0.5 - level / 2), -B, B)
    return [float(lo / np.sqrt(n)), float(hi / np.sqrt(n))], float(dz)


def contrast(a, b, direction, dmin):
    ok = ~(np.isnan(a) | np.isnan(b)); d = a[ok] - b[ok]
    ci90 = t_ci(d, 0.90)
    # non-superiority: is a benefit of at least dmin (in the predicted direction) excluded?
    excl = (ci90[1] < dmin) if direction == "greater" else (ci90[0] > -dmin)
    z = dz_ci(d)
    return {"n": int(len(d)), "mean": float(d.mean()), "p_signflip": signflip_p(d),
            "min_attainable_p": float(2 / 2 ** len(d)), "ci90_t": ci90, "ci95_t": t_ci(d, 0.95),
            "benefit_ge_dmin_excluded": bool(excl), "mde80_two_sided": mde(d),
            "dz": None if z is None else z[1], "dz_ci95_nct": None if z is None else z[0]}


out = {}
# H3 (planned, testable) and the paraphrase-only sensitivity version
a, b = vec(A, "coformer", "ood_rel_drop"), vec(A, "frozen", "ood_rel_drop")
out["H3"] = contrast(a, b, "less", 0.03)
a, b = 1 - vec(A, "coformer", "ood_retention_para"), 1 - vec(A, "frozen", "ood_retention_para")
out["H3_paraphrase_only"] = contrast(a, b, "less", 0.03)
# E family
for h, (m, c1, c2, dr) in {"E1": ("csat_norm_paraphrase", "grs_routed", "coformer", "greater"),
                           "E2": ("csat_norm_paraphrase", "grs_frozen", "frozen", "greater"),
                           "E3": ("ood_rel_drop", "grs_routed", "coformer", "less")}.items():
    out[h] = contrast(vec(EC, c1, m), vec(EC, c2, m), dr, 0.03)
# P family
for f in sorted(P.faith_fraction.unique()):
    sub = P[P.faith_fraction == f]
    out[f"P_ST@{f}"] = contrast(vec(sub, "coformer", "faith_excess_id"), vec(sub, "frozen", "faith_excess_id"), "greater", 0.03)
    out[f"P_soft@{f}"] = contrast(vec(sub, "no_stgs", "faith_excess_id"), vec(sub, "soft_frozen", "faith_excess_id"), "greater", 0.03)
# R family (seed level, S = 4)
for h, (m, s1, s2) in {"R0": ("accuracy", "CoFormer-no_stgs", "CoFormer-coformer"),
                       "R1": ("accuracy", "CoFormer-coformer", "DiLA-frozen"),
                       "R2": ("accuracy", "CoFormer-coformer", "CoFormer-frozen"),
                       "R3": ("acc_proofwriter", "CoFormer-coformer", "CoFormer-frozen")}.items():
    out[h] = contrast(vec(RC, s1, m, "system"), vec(RC, s2, m, "system"), "greater", 0.05)
# post hoc OOD factorial effects and pairwise contrasts, both families
cells = ["frozen", "coformer", "soft_frozen", "no_stgs"]
for fam, df in (("A", A), ("B", B)):
    for m in ["faith_excess_paraphrase", "faith_excess_scale", "csat_norm_paraphrase", "csat_norm_scale", "ood_rel_drop"]:
        std_, str_, sd_, sr_ = (vec(df, c, m) for c in cells)
        for k, e in {"bridge": 0.5 * ((sd_ - std_) + (sr_ - str_)), "solver": 0.5 * ((str_ - std_) + (sr_ - sd_)),
                     "interaction": (sr_ - sd_) - (str_ - std_)}.items():
            e = e[~np.isnan(e)]
            out[f"fac_{fam}_{m}_{k}"] = {"mean": float(e.mean()), "p_signflip": signflip_p(e), "n": int(len(e))}
        a, b = vec(df, "coformer", m), vec(df, "frozen", m); ok = ~(np.isnan(a) | np.isnan(b))
        out[f"pair_{fam}_{m}"] = {"mean": float((a[ok] - b[ok]).mean()), "p_signflip": signflip_p(a[ok] - b[ok])}
# P family at f = 0.005: Cochran's Q for the four success indicators on the same seeds,
# exact p-value by within-seed permutation (200,000 draws, seed 0)
sub = P[P.faith_fraction == 0.005]
Msucc = np.array([(vec(sub, c, "faith_excess_id") > 0.30).astype(int)
                  for c in ["frozen", "coformer", "soft_frozen", "no_stgs"]]).T


def cochran_q(M):
    k = M.shape[1]; C = M.sum(0); Rw = M.sum(1); N = M.sum()
    den = k * N - (Rw ** 2).sum()
    return 0.0 if den == 0 else (k - 1) * (k * (C ** 2).sum() - N ** 2) / den


q_obs = cochran_q(Msucc); rng_q = np.random.default_rng(0)
hits = sum(cochran_q(np.array([rng_q.permutation(r) for r in Msucc])) >= q_obs - 1e-12 for _ in range(200000))
out["P_success_f0.005_cochranQ"] = {"successes": Msucc.sum(0).tolist(), "Q": float(q_obs),
                                    "p_exact_perm": hits / 200000.0,
                                    "p_chi2": float(1 - stats.chi2.cdf(q_obs, Msucc.shape[1] - 1))}
# H6 shares: Clopper-Pearson for 12/12
k, n = 12, 12
out["H6_clopper_pearson_12_of_12"] = [float(stats.beta.ppf(0.025, k, n - k + 1)), 1.0]
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "robustness_numbers.json"), "w"), indent=1)
print("written robustness_numbers.json")

# ------------------------------------------------------------------ LaTeX table (Appendix)
planned = {"H3": 0.1334, "H3_paraphrase_only": None}
cp = pd.read_csv(f"{OUT}/confirmatory_plan_ii.csv")
for _, r in cp.iterrows():
    key = r.hypothesis.split(":")[0].replace("P_SOFT", "P_soft").replace(" @ f=", "@")
    planned[key] = float(r.p)
cr = pd.read_csv(f"{OUT}/confirmatory_real_data.csv")
for _, r in cr.iterrows():
    planned[r.hyp] = (float(r.p_raw), str(r.testability))


def fmt_p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def num(x, d=3):
    if abs(x) < 0.5 * 10 ** (-d):
        return f"{0:.{d}f}"
    s = f"{x:+.{d}f}"
    return s.replace("+", "$+$").replace("-", "$-$")


LAB = {"H3": r"H3 ($\Drop$)", "H3_paraphrase_only": r"H3, paraphrase only$^\dagger$",
       "E1": "E1", "E2": "E2", "E3": "E3"}
rows, order = [], ["H3", "H3_paraphrase_only", "E1", "E2", "E3"] + \
    [f"P_{b}@{f}" for f in (0.001, 0.005, 0.02) for b in ("ST", "soft")] + ["R0", "R1", "R2", "R3"]
for k in order:
    o = out[k]
    lab = LAB.get(k)
    if lab is None and k.startswith("P_"):
        b, f = k[2:].split("@")
        lab = rf"P\textsubscript{{{b}}} at $f={f}$"
    if lab is None:
        lab = k
    pl = planned.get(k)
    if isinstance(pl, tuple):
        pl_s = fmt_p(pl[0]) + ("" if pl[1] == "testable" else f" ({pl[1]})")
    elif pl is None:
        pl_s = "--"
    else:
        pl_s = fmt_p(pl)
    ci = o["ci90_t"]
    rows.append(f"{lab} & {o['n']} & {num(o['mean'])} & {pl_s} & {fmt_p(o['p_signflip'])} & "
                f"[{num(ci[0])}, {num(ci[1])}] & {'yes' if o['benefit_ge_dmin_excluded'] else 'no'} & "
                f"{o['mde80_two_sided']:.3f} \\\\")
    if k in ("H3_paraphrase_only", "E3", "P_soft@0.02"):
        rows.append(r"\midrule")
os.makedirs(sys.argv[2] if len(sys.argv) > 2 else "tables", exist_ok=True)
open(os.path.join(sys.argv[2] if len(sys.argv) > 2 else "tables", "tab_robust.tex"), "w").write("\n".join(rows) + "\n")
print("written tab_robust.tex")
