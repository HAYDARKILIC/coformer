"""Post hoc solver-side analyses that need no training (CPU only).
Usage: python solver_checks.py <path-to-CoFormer-folder>  -> writes solver_checks.json

The relaxed MaxSAT layer and the metrics are copied from the notebook (Cells 6-8) so that
the reproduction check (1) can confirm the re-implementation before it is used.
  (1) Reproduction: gold formulas of the ID split through the layer must give the reported
      raw CSat, CSat-norm, exact solve and mean refinement count of the factorial conditions.
  (2) Tiled-oracle reference for the scale split: a formula that is exact within the n-wide window
      (slot u carries the gold literal of a variable v with v mod n = u), slot-tiled to full width and
      scored like the models. Slot collisions are resolved (a) in favour of the lower-indexed variable,
      (b) of the higher-indexed one, and (c) per instance by whichever of (a) and (b) satisfies more gold
      clauses. It is a reference point, not an upper bound.
  (3) Entailment-surrogate check: |Delta_e| of the relaxed layer on 56 x 32 formulas with an
      entailed or a non-entailed conclusion literal, compared with the margin mu_e/2 = 0.025.
"""
import sys, os, json, numpy as np, torch

torch.set_num_threads(2)
R = sys.argv[1] if len(sys.argv) > 1 else "."
HERE = os.path.dirname(os.path.abspath(__file__))
FAM = {"A": "de40daa0", "B": "ba9180e9"}


# ------------------------------------------------------------------ notebook re-implementation
class SoftMaxSATLayer:
    def __init__(self, refine_steps=50, step_size=2.0, init_seed=1234):
        self.T, self.eta, self.seed = refine_steps, step_size, init_seed

    @staticmethod
    def soft_satisfaction(clause_mat, assign, present=None):
        lit = clause_mat * assign.unsqueeze(1)
        lit_true = torch.sigmoid(4.0 * lit)
        if present is None:
            present = (clause_mat.abs() > 0).float()
        p_false = (1.0 - lit_true * present).clamp(1e-7, 1.0)
        return 1.0 - torch.prod(p_false, dim=-1)

    def __call__(self, clause_mat, present=None):
        B, Cc, V = clause_mat.shape
        steps_used = torch.full((B,), float(self.T))
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        assign = 0.01 * torch.randn(B, V, generator=g)
        prev = None
        for s in range(self.T):
            assign = assign.detach().requires_grad_(True)
            with torch.enable_grad():
                sat = self.soft_satisfaction(clause_mat, assign, present).mean(dim=1)
                grad = torch.autograd.grad(sat.sum(), assign)[0]
            assign = torch.tanh(assign + self.eta * grad).detach()
            cur = self.soft_satisfaction(clause_mat, assign, present).mean(dim=1)
            if prev is not None:
                improved = (cur - prev) > 1e-4
                steps_used = torch.where(improved, steps_used,
                                         torch.minimum(steps_used, torch.full_like(steps_used, float(s))))
            prev = cur
        return assign, self.soft_satisfaction(clause_mat, assign, present), steps_used


def clause_satisfaction(assign, gold):
    hard = torch.sign(assign); hard[hard == 0] = 1.0
    lit = gold * hard.unsqueeze(1); pres = (gold.abs() > 0).float()
    return (((lit > 0).float() * pres).sum(-1) > 0).float().mean(dim=1)


def solved(assign, gold):
    hard = torch.sign(assign); hard[hard == 0] = 1.0
    lit = gold * hard.unsqueeze(1); pres = (gold.abs() > 0).float()
    return ((((lit > 0).float() * pres).sum(-1) > 0).all(dim=1)).float()


def chance(gold_np, n_draws=256, seed=0):
    g = torch.from_numpy(gold_np).float(); gen = torch.Generator().manual_seed(seed); vals = []
    for _ in range(n_draws):
        a = torch.sign(torch.randn(g.shape[0], g.shape[2], generator=gen)); a[a == 0] = 1.0
        vals.append(float(clause_satisfaction(a, g).mean()))
    return float(np.mean(vals))


def tile(cm, width):
    V = cm.shape[2]
    reps = (width + V - 1) // V
    return cm.repeat(1, 1, reps)[:, :, :width]


def evaluate(pred_np, gold_np, chance_level, layer, bs=64):
    """Mirror of the notebook's evaluate(): unshuffled batches of 64, batch means."""
    tp = npred = ngold = 0; pnz = gnz = slots = 0.0; cs, acc, st = [], [], []
    for i in range(0, gold_np.shape[0], bs):
        cm = torch.from_numpy(pred_np[i:i + bs]).float(); y = torch.from_numpy(gold_np[i:i + bs]).float()
        if cm.shape[2] != y.shape[2]:
            cm = tile(cm, y.shape[2])
        a, _, su = layer(cm)
        p, g = cm.abs() > 0, y.abs() > 0
        tp += int((p & g & (cm == y)).sum()); npred += int(p.sum()); ngold += int(g.sum())
        pnz += float(p.sum()); gnz += float(g.sum()); slots += float(cm.numel())
        cs.append(float(clause_satisfaction(a, y).mean())); acc.append(float(solved(a, y).mean()))
        st.append(float(su.mean()))
    prec, rec = tp / max(npred, 1), tp / max(ngold, 1)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    d, gd = pnz / slots, gnz / slots
    f10 = 0.0 if d <= 0 else gd * d / (gd + d)
    raw = float(np.mean(cs))
    return {"f1": f1, "excess": f1 - f10, "csat_raw": raw,
            "csat_norm": (raw - chance_level) / (1 - chance_level),
            "acc": float(np.mean(acc)), "steps": float(np.mean(st)), "density": d}


def window_oracle(gold_np, n, lower_wins=True):
    """n-wide emission exact within the window; collisions resolved by variable index."""
    B, m, W = gold_np.shape
    out = np.zeros((B, m, n), dtype=gold_np.dtype)
    order = range(W - 1, -1, -1) if lower_wins else range(W)   # the variable written last wins
    for v in order:
        col = gold_np[:, :, v]
        mask = col != 0
        out[:, :, v % n][mask] = col[mask]
    return out


def per_instance_csat(pred_np, gold_np, layer, bs=64):
    vals = []
    for i in range(0, gold_np.shape[0], bs):
        cm = torch.from_numpy(pred_np[i:i + bs]).float(); y = torch.from_numpy(gold_np[i:i + bs]).float()
        if cm.shape[2] != y.shape[2]:
            cm = tile(cm, y.shape[2])
        a, _, _ = layer(cm)
        vals.append(clause_satisfaction(a, y).numpy())
    return np.concatenate(vals)


out = {}
layer = SoftMaxSATLayer()
for fam, fp in FAM.items():
    z = np.load(f"{R}/data_cache/dataset_{fp}.npz")
    Yid, Ysc = z["Yid"].astype(np.float32), z["Ysc"].astype(np.float32)
    n = Yid.shape[2]
    ch_id, ch_sc = chance(Yid), chance(Ysc)
    # (1) reproduction on ID with the gold formula
    out[f"{fam}_id_gold"] = evaluate(Yid, Yid, ch_id, layer)
    # (2) tiled-oracle ceiling on the scale split
    lo, hi = window_oracle(Ysc, n, True), window_oracle(Ysc, n, False)
    out[f"{fam}_scale_tiled_oracle"] = evaluate(lo, Ysc, ch_sc, layer)
    out[f"{fam}_scale_tiled_oracle_higher_wins"] = evaluate(hi, Ysc, ch_sc, layer)
    # per-instance choice: each instance's assignment depends only on its own formula and batch position
    best = np.where((per_instance_csat(hi, Ysc, layer) > per_instance_csat(lo, Ysc, layer))[:, None, None], hi, lo)
    out[f"{fam}_scale_tiled_oracle_best_of_two"] = evaluate(best, Ysc, ch_sc, layer)
    out[f"{fam}_scale_gold_fullwidth"] = evaluate(Ysc, Ysc, ch_sc, layer)
    # share of scale-split clauses in which two variables collide in one slot
    coll = 0; tot = 0
    for v in range(n, Ysc.shape[2]):
        coll += int(((Ysc[:, :, v] != 0) & (Ysc[:, :, v - n] != 0)).sum())
    out[f"{fam}_scale_slot_collisions_per_clause"] = coll / float(Ysc.shape[0] * Ysc.shape[1])
    out[f"{fam}_chance"] = {"id": ch_id, "scale": ch_sc}
    print(fam, json.dumps({k: out[k] for k in out if k.startswith(fam)}, indent=None)[:600])

# (4) entailment surrogate on NL-sized formulas (m_C = 56 rows, n_C = 32 atoms)
rng = np.random.default_rng(0)
mC, nC, k = 56, 32, 3
layer_nl = SoftMaxSATLayer()
recs = {"entailed": [], "not_entailed": []}
for m_used in (16, 32, 48):
    for rep in range(64):
        F = np.zeros((mC + 1, nC), np.float32)
        x = rng.choice([-1.0, 1.0], nC)                 # planted assignment keeps premises satisfiable
        for j in range(m_used - 1):
            vs = rng.choice(nC, k, replace=False)
            sg = rng.choice([-1.0, 1.0], k)
            if not np.any(sg == x[vs]):
                sg[0] = x[vs[0]]
            F[j, vs] = sg
        c = int(rng.integers(nC)); pol = x[c]
        ent = rep % 2 == 0
        if ent:                                         # add the conclusion as a premise fact: entailed
            F[m_used - 1, c] = pol
        Fp, Fm = F.copy(), F.copy()
        Fp[mC, c] = pol; Fm[mC, c] = -pol
        # separate calls, as in the notebook: both queries start from the same initial assignment
        _, s_with, _ = layer_nl(torch.from_numpy(Fp[None]))
        _, s_against, _ = layer_nl(torch.from_numpy(Fm[None]))
        d = float(s_with[0].mean() - s_against[0].mean())
        recs["entailed" if ent else "not_entailed"].append(d)
for k_, v in recs.items():
    v = np.abs(np.array(v))
    out[f"surrogate_{k_}"] = {"n": int(len(v)), "max_abs_delta": float(v.max()),
                              "mean_abs_delta": float(v.mean()),
                              "share_below_0.025": float((v < 0.025).mean())}
out["surrogate_bound_1_over_57"] = 1 / 57
json.dump(out, open(os.path.join(HERE, "solver_checks.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k.startswith("surrogate")}, indent=1))
