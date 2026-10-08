"""Verify the SHA-256 time stamps of the pre-specified follow-up studies.

Each specification file stores the plan and the hash that the notebook computed and e-mailed
when the plan was written. The hash is SHA-256 over json.dumps(plan, sort_keys=True, indent=1).
For family H, the hash of the text-generator source is checked as well.

Usage: python specs/verify_specs.py
"""
import hashlib, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ok = True
for name in ("plan_III.json", "followup2.json", "followup_h.json"):
    d = json.load(open(os.path.join(HERE, name)))
    h = hashlib.sha256(json.dumps(d["plan"], sort_keys=True, indent=1).encode()).hexdigest()
    good = h == d["sha256"]
    ok &= good
    print(f"{name:16s} written {d['written_utc']}  sha256 {h[:16]}...  {'OK' if good else 'MISMATCH'}")
    if "generator_source" in d:
        g = hashlib.sha256(d["generator_source"].encode()).hexdigest()
        stored = d["plan"].get("generator_sha256")
        good = (stored is None) or (g == stored)
        ok &= good
        print(f"{'':16s} generator source sha256 {g[:16]}...  {'OK' if good else 'MISMATCH'}")
raise SystemExit(0 if ok else 1)
