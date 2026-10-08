"""Post hoc analyses requested by the referees (all exploratory; none enters the confirmatory families).
Usage: python3 extra_analyses.py <path-to-CoFormer-folder> [<path-to-notebook.ipynb>]
       -> writes extra_analyses.json next to this script.

1. Learning curves of the doubled-budget runs (outputs/plan_iii_learning_curves.csv):
   per-seed normalised AUC (trapezoid over epochs 3-24 / 21) and time-to-threshold
   (first recorded epoch with excess > 0.30, censored at 24); paired coformer - frozen
   differences with exact two-sided sign-flip p (all 2^6 patterns).
2. Two-part (hurdle) analysis of partial supervision at f = 0.005 and f = 0.02
   (outputs/partial_supervision.csv): success = faith_excess_id > 0.30; exact McNemar,
   conditional levels, exact within-seed permutation test of the success-rate difference,
   phi coefficient of the paired success indicators.
3. Natural-language example-level analysis: the held-out split of every seed is rebuilt with the
   notebook's own split logic (as in cot_rescore.py), per-example correctness from
   checkpoints/per_example_real.json is aligned and verified against the reported per-seed
   accuracies (overall, ProofWriter and FOLIO) and, for CoT/Direct, against a re-parse of the
   cached generations. Contrasts are then analysed with an example-cluster bootstrap and an
   example-cluster sign-flip permutation test (examples recur across seeds' held-out sets).
"""
import sys, os, json, itertools, hashlib, re
import numpy as np, pandas as pd
from scipy import stats

R = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
NB = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks", "as_run", "CoFormer_runall_v14_auto.ipynb")
CK, OUT = f"{R}/checkpoints", f"{R}/outputs"
THR = 0.30
out = {"_meta": dict(folder=R, notebook=NB, threshold=THR,
                     note="post hoc, exploratory; no multiplicity correction")}


def signflip_p(d, two_sided=True):
    """Exact sign-flip p over all 2^n patterns, statistic = mean difference."""
    d = np.asarray(d, float); n = len(d)
    if n == 0 or np.all(np.abs(d) < 1e-12):
        return 1.0
    signs = np.array(list(itertools.product([-1, 1], repeat=n)), float)
    m = (signs * d).mean(axis=1); obs = d.mean()
    if two_sided:
        return float((np.abs(m) >= abs(obs) - 1e-12).mean())
    return float((m >= obs - 1e-12).mean())


# ---------------------------------------------------------------------------------------------
# 1. Learning curves
# ---------------------------------------------------------------------------------------------
LC = pd.read_csv(f"{OUT}/plan_iii_learning_curves.csv")
LC = LC[(LC.epoch >= 3) & (LC.epoch <= 24)]
lc = {}
per = {}
for c in ["coformer", "frozen"]:
    per[c] = {}
    for s, g in LC[LC.condition == c].groupby("seed"):
        g = g.sort_values("epoch"); e = g.epoch.values.astype(float); x = g.excess.values
        auc = float(float(np.sum((x[1:] + x[:-1]) * np.diff(e)) / 2) / (e[-1] - e[0]))
        hit = e[x > THR]
        ttt = float(hit[0]) if len(hit) else 24.0
        per[c][int(s)] = dict(auc=auc, ttt=ttt, reached=bool(len(hit)), final=float(x[-1]),
                              epochs=e.tolist())
seeds = sorted(set(per["coformer"]) & set(per["frozen"]))
for m in ["auc", "ttt", "final"]:
    a = np.array([per["coformer"][s][m] for s in seeds]); b = np.array([per["frozen"][s][m] for s in seeds])
    d = a - b
    lc[m] = dict(coformer=a.tolist(), frozen=b.tolist(), diff=d.tolist(),
                 mean_coformer=float(a.mean()), mean_frozen=float(b.mean()),
                 mean_diff=float(d.mean()), sd_diff=float(d.std(ddof=1)),
                 n_pos=int((d > 1e-12).sum()), n_neg=int((d < -1e-12).sum()),
                 p_signflip_two_sided=signflip_p(d))
lc["ttt"]["reached_coformer"] = int(sum(per["coformer"][s]["reached"] for s in seeds))
lc["ttt"]["reached_frozen"] = int(sum(per["frozen"][s]["reached"] for s in seeds))
lc["ttt"]["note"] = "censored at epoch 24 when the threshold is never exceeded at a recorded epoch"
lc["seeds"] = seeds
lc["recorded_epochs"] = sorted(LC.epoch.unique().tolist())
out["learning_curves"] = lc

# ---------------------------------------------------------------------------------------------
# 2. Hurdle analysis of partial supervision
# ---------------------------------------------------------------------------------------------
P = pd.read_csv(f"{OUT}/partial_supervision.csv")
PAIRS = [("coformer", "frozen"), ("no_stgs", "soft_frozen")]
hur = {}
for f in [0.005, 0.02]:
    Pf = P[np.isclose(P.faith_fraction, f)]
    hur[str(f)] = {}
    for rt, dt in PAIRS:
        A = Pf[Pf.condition == rt].set_index("seed").faith_excess_id
        B = Pf[Pf.condition == dt].set_index("seed").faith_excess_id
        sd = sorted(set(A.index) & set(B.index)); A = A.loc[sd].values; B = B.loc[sd].values
        sa = (A > THR).astype(int); sb = (B > THR).astype(int)
        b01 = int(((sa == 1) & (sb == 0)).sum())      # routed success, detached failure
        c10 = int(((sa == 0) & (sb == 1)).sum())
        nd = b01 + c10
        p1 = float(stats.binom.sf(b01 - 1, nd, 0.5)) if nd else 1.0
        p2 = float(min(1.0, 2 * min(stats.binom.sf(b01 - 1, nd, 0.5),
                                    stats.binom.cdf(b01, nd, 0.5)))) if nd else 1.0
        dd = (sa - sb).astype(float)
        if sa.std() > 0 and sb.std() > 0:
            phi = float(np.corrcoef(sa, sb)[0, 1])
        else:
            phi = None

        def cond(x, s):
            return dict(n_success=int(s.sum()), n_fail=int((1 - s).sum()),
                        mean_success=float(x[s == 1].mean()) if s.sum() else None,
                        mean_fail=float(x[s == 0].mean()) if (1 - s).sum() else None,
                        sd_success=float(x[s == 1].std(ddof=1)) if s.sum() > 1 else None,
                        sd_fail=float(x[s == 0].std(ddof=1)) if (1 - s).sum() > 1 else None,
                        mean_all=float(x.mean()))
        hur[str(f)][f"{rt}_vs_{dt}"] = dict(
            n_seeds=len(sd), success_routed=int(sa.sum()), success_detached=int(sb.sum()),
            both=int(((sa == 1) & (sb == 1)).sum()), neither=int(((sa == 0) & (sb == 0)).sum()),
            routed_only=b01, detached_only=c10,
            mcnemar_exact_one_sided_p=p1, mcnemar_exact_two_sided_p=p2,
            success_rate_diff=float(dd.mean()),
            perm_within_seed_two_sided_p=signflip_p(dd),
            perm_within_seed_one_sided_p=signflip_p(dd, two_sided=False),
            phi=phi, conditional={rt: cond(A, sa), dt: cond(B, sb)},
            success_routed_by_seed=sa.tolist(), success_detached_by_seed=sb.tolist())
out["hurdle_partial_supervision"] = hur

# ---------------------------------------------------------------------------------------------
# 3. Natural-language example-level analysis
# ---------------------------------------------------------------------------------------------
nb = json.load(open(NB))
src = "\n".join("".join(c["source"]) for c in nb["cells"])


def getfunc(name):
    i = src.index("def " + name + "(")
    lines = src[i:].split("\n"); o = [lines[0]]
    for l in lines[1:]:
        if l and not l[0].isspace():
            break
        o.append(l)
    return "\n".join(o)


g = {}
exec("import re, numpy as np\n" + getfunc("resolve_option_answer") + "\n" + getfunc("normalize_label"), g)
i = src.index("_ANSWER_RE = "); exec("import re as _re_d\n" + src[i:src.index("\n", i)], g)
MODEL = re.search(r'"(Qwen/Qwen2\.5-7B-Instruct)"', src).group(1)
cache = json.load(open(f"{CK}/llm_generations.json"))
misses = [0]


def llm_generate(prompt, max_new_tokens=512, temperature=0.0):
    k = hashlib.md5(f"{MODEL}|{max_new_tokens}|{prompt}".encode()).hexdigest()
    if k not in cache:
        misses[0] += 1; return ""
    return cache[k]


g["llm_generate"] = llm_generate
exec(getfunc("baseline_cot") + "\n" + getfunc("baseline_direct"), g)
real = json.load(open(f"{R}/data_cache/real_benchmarks_v4.json"))


def uid(e):
    if e["dataset"] == "ProofWriter" and e.get("id"):
        return "PW:" + str(e["id"])
    h = hashlib.md5((str(e.get("context")) + "||" + str(e.get("question")) + "||" +
                     str(e.get("label"))).encode()).hexdigest()[:12]
    return e["dataset"][:2] + ":" + h


def holdout(seed):
    pool = []
    for raw in real["proofwriter"]:
        e = dict(raw); e["dataset"] = "ProofWriter"; pool.append((e, g["normalize_label"](e.get("label"), e.get("options"))))
    for raw in real["folio"]:
        if raw.get("label") is None: continue
        fol = raw.get("premises-FOL", raw.get("fol", ""))
        if isinstance(fol, str): fol = [fol] if fol else []
        if any(isinstance(x, str) and x.count("(") != x.count(")") for x in fol): continue
        e = dict(raw); e["dataset"] = "FOLIO"; pool.append((e, g["normalize_label"](e.get("label"))))
    rng = np.random.default_rng(seed); rng.shuffle(pool)
    n = len(pool); nh = min(max(40, int(0.2 * n)), max(1, int(0.75 * n)))
    return pool[:nh], n


PE = json.load(open(f"{CK}/per_example_real.json"))
RC = pd.read_csv(f"{OUT}/real_comparison_full.csv")
SEEDS = sorted({int(k.split("|")[1]) for k in PE})
H = {}; pool_n = {}
for s in SEEDS:
    H[s], pool_n[s] = holdout(s)

# alignment check
align = {"checks": [], "all_ok": True}
systems = ["SATLM-frozen", "LogicLM-frozen", "DiLA-frozen", "CoFormer-coformer", "CoFormer-frozen",
           "Direct-frozen", "CoT-frozen"]
rows = []           # long table: seed, uid, dataset, system -> correct
cot_strict = {}; cot_rebuilt = {}; direct_rebuilt = {}
for s in SEEDS:
    hs = H[s]
    ds = np.array([e["dataset"] for e, _ in hs])
    ids = [uid(e) for e, _ in hs]
    assert len(set(ids)) == len(ids), f"duplicate example ids in held-out set of seed {s}"
    cr, st, dr = [], [], []
    for e, lab in hs:
        r = g["baseline_cot"](e); d = g["baseline_direct"](e)
        cr.append(int(r["pred"] == lab)); st.append(int(r["pred"] == lab and not r["used_fallback"]))
        dr.append(int(d["pred"] == lab))
    cot_rebuilt[s] = np.array(cr); cot_strict[s] = np.array(st); direct_rebuilt[s] = np.array(dr)
    for sysn in systems:
        k = f"{sysn}|{s}"
        if k not in PE:
            align["checks"].append(dict(system=sysn, seed=s, present=False)); continue
        v = np.array(PE[k], int)
        rr = RC[(RC.system == sysn) & (RC.seed == s)]
        chk = dict(system=sysn, seed=s, present=True, n_list=len(v), n_holdout=len(hs))
        ok = len(v) == len(hs)
        if len(rr):
            rr = rr.iloc[0]
            chk.update(acc_list=float(v.mean()), acc_reported=float(rr.accuracy),
                       pw_list=float(v[ds == "ProofWriter"].mean()), pw_reported=float(rr.acc_proofwriter),
                       folio_list=float(v[ds == "FOLIO"].mean()), folio_reported=float(rr.acc_folio))
            ok &= abs(chk["acc_list"] - chk["acc_reported"]) < 1e-9
            ok &= abs(chk["pw_list"] - chk["pw_reported"]) < 1e-9
            ok &= abs(chk["folio_list"] - chk["folio_reported"]) < 1e-9
        else:
            chk["acc_reported"] = None; ok = False
        if sysn == "CoT-frozen":
            chk["identical_to_reparse"] = bool(np.array_equal(v, cot_rebuilt[s])); ok &= chk["identical_to_reparse"]
        if sysn == "Direct-frozen":
            chk["identical_to_reparse"] = bool(np.array_equal(v, direct_rebuilt[s])); ok &= chk["identical_to_reparse"]
        chk["ok"] = bool(ok); align["all_ok"] &= bool(ok)
        align["checks"].append(chk)
        for j in range(len(v)):
            rows.append((s, ids[j], ds[j], sysn, int(v[j])))
    for j in range(len(hs)):
        rows.append((s, ids[j], ds[j], "CoT-strict", int(cot_strict[s][j])))
align["generation_cache_misses"] = misses[0]
align["cot_strict_acc_by_seed"] = {s: float(cot_strict[s].mean()) for s in SEEDS}
L = pd.DataFrame(rows, columns=["seed", "uid", "dataset", "system", "correct"])

# overlap structure
occ = L[L.system == "SATLM-frozen"].groupby("uid").seed.nunique()
overlap = dict(n_observations_per_system=int(len(L[L.system == "SATLM-frozen"])),
               n_distinct_examples=int(len(occ)),
               n_examples_in_more_than_one_seed=int((occ > 1).sum()),
               distribution_seeds_per_example={int(k): int(v) for k, v in occ.value_counts().sort_index().items()},
               pool_size=int(pool_n[SEEDS[0]]), holdout_size=int(len(H[SEEDS[0]])))


def contrast(a, b, B=10000, seed=0):
    """Mean over all (seed, example) observations of correct_a - correct_b (= mean of per-seed
    accuracy differences, since every seed has the same held-out size). Inference clusters by
    example: bootstrap resamples distinct examples with replacement (keeping all their seeds);
    permutation flips the sign of each example's summed difference jointly (swap of system labels
    within example, applied to all seeds that contain it)."""
    A = L[L.system == a].set_index(["seed", "uid"]).correct
    Bv = L[L.system == b].set_index(["seed", "uid"]).correct
    idx = A.index.intersection(Bv.index)
    d = (A.loc[idx] - Bv.loc[idx]).astype(float)
    dfd = d.reset_index(); dfd.columns = ["seed", "uid", "d"]
    cl = dfd.groupby("uid").d.agg(["sum", "count"])
    S = cl["sum"].values; C = cl["count"].values; N = C.sum()
    obs = S.sum() / N
    rng = np.random.default_rng(seed)
    k = len(S)
    boots = np.empty(B)
    for t in range(B):
        ii = rng.integers(0, k, k)
        boots[t] = S[ii].sum() / C[ii].sum()
    ci = np.percentile(boots, [2.5, 97.5])
    # crossed bootstrap (examples x seeds) as a secondary interval
    seeds_ = sorted(dfd.seed.unique())
    piv = dfd.pivot(index="uid", columns="seed", values="d")          # NaN where absent
    M = piv.values; present = ~np.isnan(M); M0 = np.nan_to_num(M)
    boots2 = np.empty(B)
    for t in range(B):
        ii = rng.integers(0, k, k); jj = rng.integers(0, len(seeds_), len(seeds_))
        sub = M0[np.ix_(ii, jj)]; pr = present[np.ix_(ii, jj)]
        boots2[t] = sub.sum() / max(1, pr.sum())
    ci2 = np.percentile(boots2, [2.5, 97.5])
    # permutation: sign flip per example cluster
    nz = S[np.abs(S) > 0]
    if len(nz) <= 20:
        signs = np.array(list(itertools.product([-1, 1], repeat=len(nz))), float)
        perm = (signs * nz).sum(axis=1) / N
        p = float((np.abs(perm) >= abs(obs) - 1e-12).mean()); ptype = "exact"
        npm = len(perm)
    else:
        rng2 = np.random.default_rng(seed + 1); npm = 100000
        sg = rng2.choice([-1.0, 1.0], size=(npm, len(nz)))
        perm = (sg @ nz) / N
        p = float(((np.abs(perm) >= abs(obs) - 1e-12).sum() + 1) / (npm + 1)); ptype = "monte_carlo"
    per_seed = {int(s): float(dfd[dfd.seed == s].d.mean()) for s in seeds_}
    return dict(a=a, b=b, mean_a=float(A.loc[idx].mean()), mean_b=float(Bv.loc[idx].mean()),
                mean_diff=float(obs), ci95_example_cluster_bootstrap=[float(ci[0]), float(ci[1])],
                ci95_crossed_bootstrap=[float(ci2[0]), float(ci2[1])],
                p_example_permutation_two_sided=p, permutation=ptype, n_permutations=int(npm),
                n_observations=int(N), n_distinct_examples=int(k),
                n_examples_with_nonzero_cluster_diff=int(len(nz)),
                n_examples_in_more_than_one_seed=int((C > 1).sum()),
                per_seed_diff=per_seed, n_bootstrap=B, bootstrap_seed=seed)


CONTRASTS = [("R1_coformer_vs_DiLA", "CoFormer-coformer", "DiLA-frozen"),
             ("CoT_lenient_vs_SATLM", "CoT-frozen", "SATLM-frozen"),
             ("CoT_lenient_vs_LogicLM", "CoT-frozen", "LogicLM-frozen"),
             ("CoT_strict_vs_SATLM", "CoT-strict", "SATLM-frozen"),
             ("CoT_strict_vs_LogicLM", "CoT-strict", "LogicLM-frozen")]
nl = dict(alignment=align, overlap=overlap, contrasts={})
for name, a, b in CONTRASTS:
    if a in set(L.system) and b in set(L.system):
        nl["contrasts"][name] = contrast(a, b)
out["nl_example_level"] = nl

# ---------------------------------------------------------------------------------------------
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extra_analyses.json")
json.dump(out, open(dst, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

# summary
f3 = lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)
print("== 1. Learning curves (coformer - frozen, 6 seeds)")
for m in ["auc", "ttt", "final"]:
    r = lc[m]
    print(f"  {m}: cof {f3(r['mean_coformer'])} frz {f3(r['mean_frozen'])} diff {f3(r['mean_diff'])} "
          f"(+{r['n_pos']}/-{r['n_neg']}) p={f3(r['p_signflip_two_sided'])}")
print(f"  ttt per seed cof {lc['ttt']['coformer']} frz {lc['ttt']['frozen']}; reached "
      f"cof {lc['ttt']['reached_coformer']}/6 frz {lc['ttt']['reached_frozen']}/6")
print("== 2. Hurdle (success = excess_id > 0.30, 12 seeds)")
for f, dct in hur.items():
    for k, r in dct.items():
        rt, dt = k.split("_vs_")
        c = r["conditional"]
        print(f"  f={f} {k}: succ {r['success_routed']} vs {r['success_detached']} "
              f"(disc {r['routed_only']}/{r['detached_only']}) McN1={f3(r['mcnemar_exact_one_sided_p'])} "
              f"McN2={f3(r['mcnemar_exact_two_sided_p'])} perm2={f3(r['perm_within_seed_two_sided_p'])} phi={r['phi'] if r['phi'] is None else round(r['phi'],3)}"
              f" | succ-mean {f3(c[rt]['mean_success'])}/{f3(c[dt]['mean_success'])} fail-mean {f3(c[rt]['mean_fail'])}/{f3(c[dt]['mean_fail'])}")
print("== 3. NL example level; alignment all_ok =", align["all_ok"], "cache misses", misses[0])
for ch in align["checks"]:
    if not ch.get("ok", False):
        print("   ALIGNMENT PROBLEM:", ch)
print("  overlap:", overlap)
for name, r in nl["contrasts"].items():
    print(f"  {name}: {f3(r['mean_a'])} vs {f3(r['mean_b'])} diff {f3(r['mean_diff'])} "
          f"CI[{f3(r['ci95_example_cluster_bootstrap'][0])},{f3(r['ci95_example_cluster_bootstrap'][1])}] "
          f"crossed[{f3(r['ci95_crossed_bootstrap'][0])},{f3(r['ci95_crossed_bootstrap'][1])}] "
          f"p={r['p_example_permutation_two_sided']:.2g} ({r['permutation']}) k={r['n_distinct_examples']}")
print("wrote", dst)
