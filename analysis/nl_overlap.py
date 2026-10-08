"""Post hoc check: how many held-out natural-language examples share their premise set with a
training example of the same seed (same ProofWriter theory / FOLIO story, different question).
Usage: python nl_overlap.py <path-to-CoFormer-folder>  -> prints and writes nl_overlap.json
The split reproduces build_pools() of the notebook (Cell C5): same filters, numpy default_rng(seed)
shuffle of the pooled list, n_hold = min(max(40, int(0.2 n)), int(0.75 n)).
"""
import sys, os, json, numpy as np
R = sys.argv[1]
real = json.load(open(os.path.join(R, "data_cache", "real_benchmarks_v4.json")))
pool = []
for raw in real["proofwriter"]:
    pool.append(("ProofWriter", raw["context"]))
for raw in real["folio"]:
    if raw.get("label") is None:
        continue
    fol = raw.get("premises-FOL", raw.get("fol", ""))
    if isinstance(fol, str):
        fol = [fol] if fol else []
    if any(isinstance(f, str) and f.count("(") != f.count(")") for f in fol):
        continue
    pool.append(("FOLIO", raw["context"]))
out = {}
for seed in range(4):
    p = list(pool); np.random.default_rng(seed).shuffle(p)
    n = len(p); nh = min(max(40, int(0.2 * n)), max(1, int(0.75 * n)))
    hold, tr = p[:nh], p[nh:]
    trc = set(tr)
    shared = [e for e in hold if e in trc]
    out[seed] = dict(n_hold=nh, shared=len(shared),
                     shared_proofwriter=sum(d == "ProofWriter" for d, _ in shared),
                     shared_folio=sum(d == "FOLIO" for d, _ in shared))
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "nl_overlap.json"), "w"), indent=1)
