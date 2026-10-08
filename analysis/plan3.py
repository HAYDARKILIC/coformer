"""Plan III analysis (follow-up experiments specified before their runs; outputs/plan_III.json).
Usage: python plan3.py <path-to-CoFormer-folder> <tables-dir>  -> plan3_numbers.json, tab_plan3.tex
Confirmatory (A2): one-sided exact sign-flip test, Holm over {A2_ST, A2_soft}, delta_min 0.03.
Diagnostic (A1, K, W, B): estimates, t intervals, exact two-sided sign-flip p, MDE; no verdicts."""
import sys, os, json, itertools, numpy as np, pandas as pd
from scipy import stats, optimize
R = sys.argv[1]; TD = sys.argv[2] if len(sys.argv) > 2 else "tables"
HERE = os.path.dirname(os.path.abspath(__file__))
P3 = pd.read_csv(f"{R}/checkpoints/results_planIII_99f69cb71d.csv")
PS = pd.read_csv(f"{R}/checkpoints/results_partial_99f69cb71d.csv")
A = pd.read_csv(f"{R}/checkpoints/results_probeA_99f69cb71d.csv")
p05 = PS[PS.faith_fraction.round(4) == 0.005]
plan = json.load(open(f"{R}/outputs/plan_III.json"))

def signflip(d, alt="two"):
    d = np.asarray(d, float)
    if np.all(np.abs(d) < 1e-12): return 1.0
    s = np.array(list(itertools.product([-1.0, 1.0], repeat=len(d)))); m = (s * d).mean(1); o = d.mean()
    return float((m >= o - 1e-12).mean()) if alt == "greater" else float((np.abs(m) >= abs(o) - 1e-12).mean())
def tci(d, lev):
    d = np.asarray(d, float); n = len(d); h = stats.t.ppf(.5 + lev / 2, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return [float(d.mean() - h), float(d.mean() + h)]
def mde(d):
    d = np.asarray(d, float); n = len(d); sd = d.std(ddof=1); tc = stats.t.ppf(.975, n - 1)
    f = lambda es: (np.nan_to_num(stats.nct.sf(tc, n - 1, es * np.sqrt(n)), nan=1.0)
                    + np.nan_to_num(stats.nct.cdf(-tc, n - 1, es * np.sqrt(n)), nan=0.0) - 0.8)
    return float(optimize.brentq(f, 1e-6, 20) * sd)
def vals(df, cond, seeds, m="faith_excess_id"):
    s = df[df.condition == cond].set_index("seed")[m]; return s.reindex(seeds)

out = {"plan_sha256": plan["sha256"], "plan_written_utc": plan["written_utc"],
       "n_rows": int(len(P3)), "torch": sorted(P3.torch_version.unique().tolist())}
# ---------- A2 confirmatory
live = []
for h, a, b in [("A2_ST", "ans_st_ps", "frozen"), ("A2_soft", "ans_soft_ps", "soft_frozen")]:
    S = list(range(12)); x = vals(P3[P3.experiment == "A2"], a, S); y = vals(p05, b, S)
    ok = ~(x.isna() | y.isna()); x, y = x[ok].values, y[ok].values; d = x - y
    xc = vals(P3[P3.experiment == "A2"], a, S, "csat_norm_id")[ok].values; yc = vals(p05, b, S, "csat_norm_id")[ok].values
    sa, sb = x > .30, y > .30; bb, cc = int((sa & ~sb).sum()), int((~sa & sb).sum())
    r = dict(id=h, n=int(len(d)), mean_a=float(x.mean()), mean_b=float(y.mean()), diff=float(d.mean()),
             ci95=tci(d, .95), ci90=tci(d, .90), p_one=signflip(d, "greater"), p_two=signflip(d),
             diff_csat=float((xc - yc).mean()), ci95_csat=tci(xc - yc, .95), p_two_csat=signflip(xc - yc),
             succ_a=int(sa.sum()), succ_b=int(sb.sum()),
             p_mcnemar_one=float(stats.binom.sf(bb - 1, bb + cc, .5)) if bb + cc else 1.0,
             mde80=mde(d), max_a=float(x.max()))
    r["benefit_excluded"] = bool(r["ci90"][1] < 0.03); live.append(r)
ps = [r["p_one"] for r in live]; order = np.argsort(ps); k = len(ps); adj = [0.0] * k; run = 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (k - rank) * ps[i])); adj[i] = run
for r, pa in zip(live, adj):
    r["p_holm"] = pa; r["verdict"] = "supported" if (pa < .05 and r["diff"] >= .03) else "not supported"
out["A2"] = {r["id"]: r for r in live}
# ---------- A1 diagnostic
for h, a, ref in [("A1_ST", "ans_st", "no_faith"), ("A1_soft", "ans_soft", "soft_no_faith")]:
    s = P3[(P3.experiment == "A1") & (P3.condition == a)]
    out.setdefault("A1", {})[h] = dict(n=int(len(s)), mean=float(s.faith_excess_id.mean()),
        max=float(s.faith_excess_id.max()), degenerate=int(s.degenerate.astype(bool).sum()),
        density=float(s.pred_density.mean()), soft_density=float(s.pred_soft_density.mean()),
        ref_mean=float(A[A.condition == ref].faith_excess_id.mean()))
# ---------- K, W, B diagnostic
diag = {}
for h, exp, a, bsrc, b in [("K_ST", "K", "coformer", "p05", "frozen"), ("K_soft", "K", "no_stgs", "p05", "soft_frozen"),
                           ("W_ST", "W", "coformer", "p05", "frozen"), ("W_soft", "W", "no_stgs", "p05", "soft_frozen"),
                           ("B_ST", "B", "coformer", "B", "frozen"),
                           ("K_vs_K10_ST", "K", "coformer", "p05", "coformer"), ("K_vs_K10_soft", "K", "no_stgs", "p05", "no_stgs"),
                           ("W_vs_w1_ST", "W", "coformer", "p05", "coformer"), ("W_vs_w1_soft", "W", "no_stgs", "p05", "no_stgs"),
                           ("B_coformer_vs_12ep", "B", "coformer", "p05", "coformer"), ("B_frozen_vs_12ep", "B", "frozen", "p05", "frozen")]:
    S = list(range(6)); bdf = p05 if bsrc == "p05" else P3[P3.experiment == "B"]
    x = vals(P3[P3.experiment == exp], a, S); y = vals(bdf, b, S); ok = ~(x.isna() | y.isna())
    if ok.sum() < 2: continue
    x, y = x[ok].values, y[ok].values; d = x - y
    diag[h] = dict(n=int(len(d)), mean_a=float(x.mean()), mean_b=float(y.mean()), diff=float(d.mean()),
                   ci95=tci(d, .95), p_two=signflip(d), mde80=mde(d), succ_a=int((x > .3).sum()), succ_b=int((y > .3).sum()))
out["diag"] = diag
# learning curves of the budget stage: share of runs still rising after epoch 12
B = P3[P3.experiment == "B"]; lc = []
for r in B.itertuples():
    c = dict((int(e), float(v)) for e, v in json.loads(r.learning_curve or "[]"))
    if 12 in c and 24 in c: lc.append(dict(condition=r.condition, seed=int(r.seed), e12=c[12], e24=c[24]))
lc = pd.DataFrame(lc)
if len(lc):
    out["B_curves"] = dict(n=int(len(lc)), mean_gain_12_to_24=float((lc.e24 - lc.e12).mean()),
                           runs_gain_gt_0_05=int(((lc.e24 - lc.e12) > .05).sum()),
                           max_gain=float((lc.e24 - lc.e12).max()))
# ---------- natural-language family D (descriptive)
RC = pd.read_csv(f"{R}/checkpoints/real_comparison.csv")
dsys = [s for s in ("Direct-frozen", "CoT-frozen") if s in set(RC.system)]
out["D"] = {s: {m: float(RC[RC.system == s][m].mean()) for m in ("accuracy", "acc_proofwriter", "acc_folio", "abstain_rate", "fallback_rate")
                if m in RC.columns} | {"n_seeds": int((RC.system == s).sum())} for s in dsys}
json.dump(out, open(os.path.join(HERE, "plan3_numbers.json"), "w"), indent=1)

# ---------- LaTeX table body
def f(x, d=3, sign=False):
    if abs(x) < 0.0005: return f"{0:.{d}f}"
    return ("$-$" if x < 0 else ("$+$" if sign else "")) + f"{abs(x):.{d}f}"
def ci(c): return f"[{f(c[0], sign=True)}, {f(c[1], sign=True)}]"
def p(x): return "$<$0.001" if x < 0.001 else f"{x:.3f}"
L = {"A2_ST": r"A2, ST: \cond{ans\_st\_ps} vs.\ \cond{frozen}", "A2_soft": r"A2, soft: \cond{ans\_soft\_ps} vs.\ \cond{soft\_frozen}",
     "K_ST": r"K, ST: $K=T$ vs.\ \cond{frozen}", "K_soft": r"K, soft: $K=T$ vs.\ \cond{soft\_frozen}",
     "W_ST": r"W, ST: task weight 5 vs.\ \cond{frozen}", "W_soft": r"W, soft: task weight 5 vs.\ \cond{soft\_frozen}",
     "B_ST": r"B: \cond{coformer} vs.\ \cond{frozen}, 24 epochs"}
rows = []
for h in ("A2_ST", "A2_soft"):
    r = out["A2"][h]
    rows.append(f"{L[h]} & {r['n']} & {f(r['mean_a'])} & {f(r['mean_b'])} & {f(r['diff'], sign=True)} & {ci(r['ci95'])} & {p(r['p_two'])} & {r['succ_a']}/{r['succ_b']} \\\\")
rows.append(r"\midrule")
for h in ("K_ST", "K_soft", "W_ST", "W_soft", "B_ST"):
    if h not in diag: continue
    r = diag[h]
    rows.append(f"{L[h]} & {r['n']} & {f(r['mean_a'])} & {f(r['mean_b'])} & {f(r['diff'], sign=True)} & {ci(r['ci95'])} & {p(r['p_two'])} & {r['succ_a']}/{r['succ_b']} \\\\")
os.makedirs(TD, exist_ok=True)
open(os.path.join(TD, "tab_plan3.tex"), "w").write("\n".join(rows) + "\n")
print(json.dumps({k: out[k] for k in ("n_rows", "A2")}, indent=1)[:1500])
print(json.dumps(out["diag"], indent=0)[:1500]); print(out.get("B_curves"), out["A1"], out["D"])
