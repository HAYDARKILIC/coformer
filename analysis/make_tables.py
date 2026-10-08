"""Generate the LaTeX tables of sections/results.tex directly from the run files.
Usage: python make_tables.py <path-to-CoFormer-folder> <output-dir>
No number in the tables is transcribed by hand."""
import sys, os, json, numpy as np, pandas as pd
R = sys.argv[1] if len(sys.argv) > 1 else "."
OD = sys.argv[2] if len(sys.argv) > 2 else "tables"
os.makedirs(OD, exist_ok=True)
CK, OUT = f"{R}/checkpoints", f"{R}/outputs"
FAM = {"A": pd.read_csv(f"{CK}/results_probeA_99f69cb71d.csv"),
       "B": pd.read_csv(f"{CK}/results_probeB_99f69cb71d.csv")}
CONDS = ["frozen", "coformer", "soft_frozen", "no_stgs", "no_faith", "soft_no_faith"]
TT = {c: r"\cond{" + c.replace("_", r"\_") + "}" for c in CONDS + ["grs_frozen", "grs_routed"]}

def f(x, d=3, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)): return "--"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    if float(s) == 0: s = s.replace("-", "").replace("+", "") if not sign else s.replace("-", "+")
    return s.replace("-", "$-$") if not sign else s.replace("-", "$-$").replace("+", "$+$")
def ms(v, d=3):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) == 0: return "--"
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return f(v.mean(), d) + (r"\,{\scriptsize(" + f(sd, d) + ")}" if sd >= 0.5 * 10 ** (-d) else "")
def pv(p):
    if p is None or not np.isfinite(p): return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"
def write(name, s):
    open(f"{OD}/{name}.tex", "w").write(s)

# ------------------------------------------------------------------ ID results
rows = []
for fam, df in FAM.items():
    rows.append(r"\midrule" + "\n" + rf"\multicolumn{{9}}{{@{{}}l}}{{\textit{{Family {fam}}} ({'3-SAT, $n=16$, $m=64$' if fam=='A' else '4-SAT, $n=12$, $m=108$'})}} \\")
    for c in CONDS:
        s = df[df.condition == c]
        deg = int(s.degenerate.sum())
        rows.append(" & ".join([TT[c], ms(s.faith_id), ms(s.faith_excess_id), ms(s.csat_norm_id), ms(s.acc_id),
                                ms(s.refine_steps, 2), ms(s.error_wrong_translation), ms(s.pred_density),
                                f"{deg}/{len(s)}"]) + r" \\")
write("tab_id", "\n".join(rows) + "\n")

# ------------------------------------------------------------------ OOD results
E = pd.read_csv(f"{OUT}/results_estimator.csv")
rows = []
for fam, df in FAM.items():
    rows.append(r"\midrule" + "\n" + rf"\multicolumn{{6}}{{@{{}}l}}{{\textit{{Family {fam}}}}} \\")
    src = [(c, df) for c in ["frozen", "coformer", "soft_frozen", "no_stgs"]]
    if fam == "A":
        src += [("grs_frozen", E), ("grs_routed", E)]
    for c, d in src:
        s = d[d.condition == c]
        drop = s.ood_rel_drop
        rows.append(" & ".join([TT[c] + (r"$^\dagger$" if c.startswith("grs") else ""),
                                ms(s.faith_excess_paraphrase), ms(s.faith_excess_scale),
                                ms(s.csat_norm_paraphrase), ms(s.csat_norm_scale), ms(drop)]) + r" \\")
write("tab_ood", "\n".join(rows) + "\n")

# ------------------------------------------------------------------ confirmatory H1-H9
C = pd.read_csv(f"{OUT}/confirmatory_holm.csv")
MET = {"faith_excess_id": r"$\Fex$ (ID)", "csat_norm_id": r"$\csatn$ (ID)", "ood_rel_drop": r"$\Drop$",
       "refine_steps": r"mean $t^{\ast}$", "error_wrong_translation": "wrong-transl.\\ share"}
CMP = {"H1": ("coformer", "frozen"), "H2": ("coformer", "frozen"), "H3": ("coformer", "frozen"),
       "H4": ("coformer", "frozen"), "H5": ("coformer", "frozen"), "H7": ("no_stgs", "coformer"),
       "H8": ("soft_frozen", "frozen"), "H9": ("no_stgs", "soft_frozen")}
rows = []
for _, r in C.iterrows():
    h = r.hypothesis; a, b = CMP[h]
    ver = str(r.verdict)
    ver = ("not testable (vacuous)" if "vacuous" in ver else "not testable (floored)" if "floored" in ver
           else "not supported" if ver.startswith("not supp") else "supported" if ver.startswith("SUPP") else ver)
    fam = "exploratory" if h in ("H7", "H8", "H9") else "confirmatory"
    rows.append(" & ".join([h, MET[r.metric], TT[a] + " -- " + TT[b], f(r.mean_diff, 4, True), pv(r.p_raw),
                            pv(r.p_holm) if np.isfinite(r.p_holm) else "--", f(r.cohen_d, 2, True), ver]) + r" \\")
write("tab_conf", "\n".join(rows) + "\n")

# ------------------------------------------------------------------ factorial (post hoc, OOD)
NUM = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_numbers.json")))
FM = [("csat_norm_paraphrase", r"$\csatn$ (paraphrase)"), ("csat_norm_scale", r"$\csatn$ (scale)"),
      ("faith_excess_paraphrase", r"$\Fex$ (paraphrase)"), ("faith_excess_scale", r"$\Fex$ (scale)"),
      ("ood_rel_drop", r"$\Drop$")]
rows = []
for fam in "AB":
    rows.append(r"\midrule" + "\n" + rf"\multicolumn{{4}}{{@{{}}l}}{{\textit{{Family {fam}}}}} \\")
    for m, lab in FM:
        cell = NUM[f"factorial_{fam}"][m]
        rows.append(" & ".join([lab] + [f(cell[k]["mean"], 4, True) + r"\,{\scriptsize(" + pv(cell[k]["p"]) + ")}"
                                         for k in ("bridge", "solver", "interaction")]) + r" \\")
write("tab_factorial", "\n".join(rows) + "\n")

# ------------------------------------------------------------------ plan II
P2 = pd.read_csv(f"{OUT}/confirmatory_plan_ii.csv")
rows, last = [], None
for _, r in P2.iterrows():
    famname = "E" if r.family.startswith("E") else "P"
    if famname != last:
        rows.append(r"\midrule" + "\n" + rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{'Estimator replication, seeds 12--23' if famname=='E' else 'Partial supervision, seeds 0--11'}}}}} \\")
        last = famname
    lab = r.hypothesis.split(":")[0].replace("P_ST", r"P\textsubscript{ST}").replace("P_SOFT", r"P\textsubscript{soft}")
    lab = lab.replace(" @ f=", r"\,@\,$f{=}$").replace("0.02", "0.02")
    rows.append(" & ".join([lab, f(r.mean_diff, 4, True), "[" + f(r.ci_lo, 4, True) + ", " + f(r.ci_hi, 4, True) + "]",
                            pv(r.p), pv(r.p_holm), f(r.dz, 2, True), "not supported" if r.verdict.startswith("not") else r.verdict]) + r" \\")
write("tab_planii", "\n".join(rows) + "\n")

# ------------------------------------------------------------------ partial supervision
P = pd.read_csv(f"{OUT}/partial_supervision.csv")
rows = []
for fr in sorted(P.faith_fraction.unique()):
    k = int(round(fr * 20000))
    rows.append(r"\midrule" + "\n" + rf"\multicolumn{{6}}{{@{{}}l}}{{$f={fr:g}$ ({k} supervised instances)}} \\")
    for c in ["frozen", "coformer", "soft_frozen", "no_stgs"]:
        s = P[(P.faith_fraction == fr) & (P.condition == c)]
        rows.append(" & ".join([TT[c], ms(s.faith_excess_id), ms(s.csat_norm_id), ms(s.acc_id),
                                f"{int((s.faith_excess_id > 0.30).sum())}/{len(s)}", f"{int(s.degenerate.sum())}/{len(s)}"]) + r" \\")
write("tab_partial", "\n".join(rows) + "\n")

# ------------------------------------------------------------------ natural-language benchmarks
RC = pd.read_csv(f"{OUT}/real_comparison_full.csv")
SYS = [("majority_floor", "Majority-class floor"), ("SATLM-frozen", r"\sys{SatLM}-style"),
       ("LogicLM-frozen", r"\sys{Logic-LM}-style"), ("DiLA-frozen", r"\sys{DiLA}-style"),
       ("CoFormer-frozen", TT["frozen"]), ("CoFormer-coformer", TT["coformer"]),
       ("CoFormer-no_faith", TT["no_faith"]), ("CoFormer-no_stgs", TT["no_stgs"]),
       ("CoFormer-soft_frozen", TT["soft_frozen"])]
rows = []
for s, lab in SYS:
    d = RC[RC.system == s]
    if s == "CoFormer-frozen": rows.append(r"\midrule")
    if s == "SATLM-frozen": rows.append(r"\midrule")
    g = lambda m, dd=3: ms(d[m], dd) if m in d and d[m].notna().any() else "--"
    rows.append(" & ".join([lab, g("accuracy"), g("acc_proofwriter"), g("acc_folio"), g("abstain_rate"),
                            g("fallback_rate"), g("mean_refine"), g("lora_param_delta", 2)]) + r" \\")
write("tab_real", "\n".join(rows) + "\n")

CR = pd.read_csv(f"{OUT}/confirmatory_real_data.csv")
RCMP = {"R0": ("no_stgs", "coformer"), "R1": ("coformer", r"\sys{DiLA}-style"), "R2": ("coformer", "frozen"),
        "R3": ("coformer", "frozen"), "R0b": ("no_stgs", "coformer"), "R4": ("coformer", "frozen")}
RMET = {"accuracy": "accuracy", "acc_proofwriter": "acc.\\ (ProofWriter)", "acc_folio": "acc.\\ (FOLIO)"}
rows = []
for _, r in CR.iterrows():
    a, b = RCMP[r.hyp]
    a = TT.get(a, a); b = TT.get(b, b)
    st = str(r.testability)
    ver = ("not testable (floored)" if st == "floored" else "not testable (vacuous)" if st == "vacuous"
           else ("not supported" if str(r.verdict).startswith("not") else str(r.verdict)))
    fam = "exploratory" if r.hyp in ("R0b", "R4") else "confirmatory"
    rows.append(" & ".join([r.hyp, RMET[r.metric], a + " -- " + b, f(r.mean_diff, 4, True), pv(r.p_raw),
                            pv(r.p_holm) if np.isfinite(r.p_holm) else "--", f(r.cohen_d, 2, True), ver]) + r" \\")
write("tab_realtests", "\n".join(rows) + "\n")

# cluster bootstrap (real)
rows = []
LBL = {"CoFormer-no_stgs": TT["no_stgs"], "CoFormer-coformer": TT["coformer"], "CoFormer-frozen": TT["frozen"], "DiLA-frozen": r"\sys{DiLA}-style"}
for line in open(f"{OUT}/example_bootstrap_real.txt"):
    if "_vs_" not in line: continue
    lhs, rest = line.split(":", 1)
    a, b = lhs.strip().split("_vs_")
    kv = dict(x.split("=") for x in rest.split())
    lo, hi = kv["CI"].strip("[]").split(",")
    rows.append(" & ".join([LBL[a] + " -- " + LBL[b], f(float(kv["diff"]), 4, True),
                            "[" + f(float(lo), 4, True) + ", " + f(float(hi), 4, True) + "]", pv(float(kv["p"]))]) + r" \\")
write("tab_realcluster", "\n".join(rows) + "\n")

# ------------------------------------------------------------------ sweeps + negative control
rows = []
L = pd.read_csv(f"{OUT}/lambda_sweep.csv"); D = pd.read_csv(f"{OUT}/density_sweep.csv")
RS = pd.read_csv(f"{OUT}/ratio_sweep.csv"); NC = pd.read_csv(f"{OUT}/negative_control.csv")
def srow(lab, r):
    return " & ".join([lab, f(r.faith_excess_id), f(r.csat_norm_id), f(r.csat_norm_paraphrase), f(r.csat_norm_scale),
                       f(r.acc_id), f(r.pred_density), "yes" if bool(r.degenerate) else "no"]) + r" \\"
rows.append(r"\midrule" + "\n" + r"\multicolumn{8}{@{}l}{\textit{Faithfulness weight} $\lambda_{\mathrm f}$ (\cond{coformer})} \\")
for _, r in L.sort_values("lambda_faith").iterrows(): rows.append(srow(rf"$\lambda_{{\mathrm f}}={r.lambda_faith:g}$", r))
rows.append(r"\midrule" + "\n" + r"\multicolumn{8}{@{}l}{\textit{Density-hinge weight} $\lambda_{\mathrm d}$ (\cond{coformer})} \\")
for _, r in D.sort_values("lambda_density").iterrows(): rows.append(srow(rf"$\lambda_{{\mathrm d}}={r.lambda_density:g}$", r))
rows.append(r"\midrule" + "\n" + r"\multicolumn{8}{@{}l}{\textit{Clause-to-variable ratio} $\alpha$} \\")
for _, r in RS.sort_values(["sweep_ratio", "condition"], ascending=[True, False]).iterrows():
    rows.append(srow(rf"$\alpha={r.sweep_ratio:g}$, " + TT[r.condition], r))
rows.append(r"\midrule" + "\n" + r"\multicolumn{8}{@{}l}{\textit{Shuffled-label negative control} (production budget)} \\")
for _, r in NC.iterrows(): rows.append(srow(TT[r.base_condition] + ", shuffled", r))
write("tab_sweeps", "\n".join(rows) + "\n")

# ------------------------------------------------------------------ objective anchoring
A = pd.read_csv(f"{OUT}/objective_anchoring.csv")
NAME = lambda s: (s.replace("random d=", r"random, $\rho=").replace("% kept", r"\%$ of slots kept") if "kept" in s
                  else s.replace("random d=", r"random, $\rho=") + ("$" if "random" in s else ""))
rows = []
for _, r in A.iterrows():
    nm = r.cnf
    if nm.startswith("random"): lab = r"random, $\rho=" + nm.split("=")[1] + "$"
    elif "kept" in nm: lab = "gold, " + nm.split()[1].replace("%", r"\%") + " of slots kept"
    else: lab = "gold"
    rows.append(" & ".join([lab, f(r.density), f(r.clause_f1), f(r.loss_self, 4), f(r.loss_anchored, 4)]) + r" \\")
write("tab_anchor", "\n".join(rows) + "\n")
print("tables written to", OD)
