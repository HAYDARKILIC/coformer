"""Recompute every statistic cited in sections/results.tex from the raw run files.
Usage: python compute_results.py <path-to-CoFormer-folder>   -> writes results_numbers.json
Estimators replicate the notebook (Cells 3c, 11b, 11f, 13b): seed-paired bootstrap
(10,000 resamples, seed 0, two-sided p), paired d_z, testability screen."""
import sys, json, numpy as np, pandas as pd
R = sys.argv[1] if len(sys.argv) > 1 else "."
CK, OUT = f"{R}/checkpoints", f"{R}/outputs"
A = pd.read_csv(f"{CK}/results_probeA_99f69cb71d.csv")
B = pd.read_csv(f"{CK}/results_probeB_99f69cb71d.csv")

def pboot(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(a, float) - np.asarray(b, float)
    bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n)])
    return float(d.mean()), float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
def dz(a, b):
    d = np.asarray(a, float) - np.asarray(b, float); sd = d.std(ddof=1)
    return 0.0 if sd < 1e-9 and abs(d.mean()) < 1e-9 else (float("nan") if sd < 1e-9 else float(d.mean() / sd))
def ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d, float)
    bs = np.array([rng.choice(d, len(d)).mean() for _ in range(n)])
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
def vec(df, c, m): return df[df.condition == c].sort_values("seed")[m].values
def chance(Y, draws=256, seed=0):
    rng = np.random.default_rng(seed); pres = Y != 0; vals = []
    for _ in range(draws):
        a = np.where(rng.standard_normal((Y.shape[0], Y.shape[2])) >= 0, 1.0, -1.0)
        lit = Y * a[:, None, :]
        vals.append(((((lit > 0) & pres).sum(-1) > 0).mean(1)).mean())
    return float(np.mean(vals))

out = {}
M = ["faith_id", "faith_excess_id", "faith_excess_paraphrase", "faith_excess_scale",
     "csat_raw_id", "csat_norm_id", "csat_norm_paraphrase", "csat_norm_scale",
     "acc_id", "acc_paraphrase", "acc_scale", "refine_steps", "ood_rel_drop",
     "ood_retention_para", "ood_retention_scale", "error_wrong_translation",
     "error_refine_failure", "pred_density", "pred_soft_density", "pred_entropy",
     "degenerate", "train_secs", "grad_steps", "collapse_step"]
for fam, df in (("A", A), ("B", B)):
    g = df.groupby("condition")
    out[f"agg_{fam}"] = {c: {m: {"mean": float(g.get_group(c)[m].mean()),
                                 "sd": float(g.get_group(c)[m].std()),
                                 "n": int(g.get_group(c)[m].notna().sum())}
                             for m in M if m in df.columns} for c in g.groups}
    # ID identity of the four ceiling cells
    cells = ["frozen", "coformer", "soft_frozen", "no_stgs"]
    out[f"id_identical_{fam}"] = {m: bool(np.allclose(np.vstack([vec(df, c, m) for c in cells]),
                                                     vec(df, "frozen", m)[None, :], atol=1e-9))
                                  for m in ["faith_id", "csat_norm_id", "acc_id", "refine_steps",
                                            "error_wrong_translation"]}
    # factorial decomposition (pre-registered on ID; OOD metrics descriptive)
    fac = {}
    for m in ["faith_excess_id", "csat_norm_id", "faith_excess_paraphrase", "faith_excess_scale",
              "csat_norm_paraphrase", "csat_norm_scale", "ood_rel_drop"]:
        std_, str_, sd_, sr_ = (vec(df, c, m) for c in cells)
        eff = {"bridge": 0.5 * ((sd_ - std_) + (sr_ - str_)),
               "solver": 0.5 * ((str_ - std_) + (sr_ - sd_)),
               "interaction": (sr_ - sd_) - (str_ - std_)}
        fac[m] = {}
        for k, e in eff.items():
            e = e[~np.isnan(e)]
            mu, p = pboot(e, np.zeros_like(e))
            fac[m][k] = {"mean": mu, "p": p, "dz": dz(e, np.zeros_like(e)), "ci": ci(e), "n": int(len(e))}
    out[f"factorial_{fam}"] = fac
    # pairwise contrasts coformer vs frozen on OOD metrics (descriptive)
    pc = {}
    for m in ["faith_excess_paraphrase", "faith_excess_scale", "csat_norm_paraphrase",
              "csat_norm_scale", "ood_rel_drop"]:
        a, b = vec(df, "coformer", m), vec(df, "frozen", m); ok = ~(np.isnan(a) | np.isnan(b))
        mu, p = pboot(a[ok], b[ok]); pc[m] = {"mean": mu, "p": p, "dz": dz(a[ok], b[ok]),
                                             "ci": ci(a[ok] - b[ok]), "n": int(ok.sum())}
    out[f"coformer_vs_frozen_ood_{fam}"] = pc
    # chance levels, measured on the cached splits
    fp = {"A": "de40daa0", "B": "ba9180e9"}[fam]
    z = np.load(f"{R}/data_cache/dataset_{fp}.npz")
    out[f"data_{fam}"] = {"chance": {s: chance(z[k]) for s, k in (("id", "Yid"), ("paraphrase", "Ypa"), ("scale", "Ysc"))},
                          "gold_density": {s: float((z[k] != 0).mean()) for s, k in (("id", "Yid"), ("paraphrase", "Ypa"), ("scale", "Ysc"))},
                          "stats": json.loads(str(z["stats"]))}

# confirmatory recomputation (family A)
out["confirmatory_recomputed"] = {}
for h, (m, c1, c2) in {"H1": ("faith_excess_id", "coformer", "frozen"), "H2": ("csat_norm_id", "coformer", "frozen"),
                       "H3": ("ood_rel_drop", "coformer", "frozen"), "H4": ("refine_steps", "coformer", "frozen"),
                       "H5": ("error_wrong_translation", "coformer", "frozen"), "H7": ("faith_excess_id", "no_stgs", "coformer"),
                       "H8": ("faith_excess_id", "soft_frozen", "frozen"), "H9": ("faith_excess_id", "no_stgs", "soft_frozen")}.items():
    a, b = vec(A, c1, m), vec(A, c2, m); ok = ~(np.isnan(a) | np.isnan(b)); a, b = a[ok], b[ok]
    d = a - b; mu, p = pboot(a, b)
    out["confirmatory_recomputed"][h] = {"mean": mu, "p": p, "dz": dz(a, b), "all_zero": bool(np.all(np.abs(d) < 1e-12)), "n": int(len(d))}

# estimator + partial supervision
E = pd.read_csv(f"{OUT}/results_estimator.csv"); EC = pd.read_csv(f"{OUT}/results_estimator_confirmatory.csv")
P = pd.read_csv(f"{OUT}/partial_supervision.csv")
for name, df in (("estimator", E), ("estimator_conf", EC)):
    g = df.groupby("condition")
    out[name] = {c: {m: {"mean": float(g.get_group(c)[m].mean()), "sd": float(g.get_group(c)[m].std())}
                     for m in ["faith_excess_id", "faith_excess_paraphrase", "faith_excess_scale", "csat_norm_id",
                               "csat_norm_paraphrase", "csat_norm_scale", "acc_id", "ood_rel_drop"]} for c in g.groups}
out["partial"] = {}
for (f, c), sub in P.groupby(["faith_fraction", "condition"]):
    ex = sub.sort_values("seed")["faith_excess_id"].values
    out["partial"][f"{f}|{c}"] = {m: {"mean": float(sub[m].mean()), "sd": float(sub[m].std())}
                                 for m in ["faith_id", "faith_excess_id", "csat_norm_id", "acc_id", "pred_density", "degenerate"]}
    out["partial"][f"{f}|{c}"]["success_n"] = int((ex > 0.30).sum())
    out["partial"][f"{f}|{c}"]["min_max_median"] = [float(ex.min()), float(ex.max()), float(np.median(ex))]

# natural-language stage
RC = pd.read_csv(f"{OUT}/real_comparison_full.csv")
g = RC.groupby("system")
out["real"] = {s: {m: {"mean": float(g.get_group(s)[m].mean()), "sd": float(g.get_group(s)[m].std())}
                   for m in ["accuracy", "acc_proofwriter", "acc_folio", "abstain_rate", "fallback_rate", "mean_refine",
                             "lora_param_delta", "pct_label_signal", "n_gold_cnf_applied", "n_label_signal", "train_secs", "opt_steps"]
                   if m in RC.columns and g.get_group(s)[m].notna().any()} for s in g.groups}
sat, dila = RC[RC.system == "SATLM-frozen"].sort_values("seed"), RC[RC.system == "DiLA-frozen"].sort_values("seed")
out["satlm_equals_dila"] = bool(np.allclose(sat[["accuracy", "abstain_rate", "fallback_rate"]].values,
                                            dila[["accuracy", "abstain_rate", "fallback_rate"]].values))
pe = json.load(open(f"{CK}/per_example_real.json"))
out["satlm_equals_dila_per_example"] = all(pe.get(f"SATLM-frozen|{s}") == pe.get(f"DiLA-frozen|{s}") for s in range(4))
out["holdout_sizes"] = {s: len(pe.get(f"SATLM-frozen|{s}", [])) for s in range(4)}
json.dump(out, open("results_numbers.json", "w"), indent=1, default=float)
print("written results_numbers.json")
