#!/bin/sh
# Recompute every number, table and figure of the article from the stored per-run files.
# Runs on CPU in a few minutes. Afterwards, `git status` / `git diff` shows whether anything
# differs from the committed outputs (it should not).
set -e
cd "$(dirname "$0")"
STATE="$(pwd)/results/CoFormer"
NB="$(pwd)/notebooks/as_run/CoFormer_runall_v14_auto.ipynb"
cd analysis
python3 compute_results.py  "$STATE"
python3 robustness.py       "$STATE" ../tables
python3 solver_checks.py    "$STATE"
python3 nl_overlap.py       "$STATE" > /dev/null
python3 extra_analyses.py   "$STATE" "$NB"
python3 cot_rescore.py      "$STATE" "$NB"
python3 plan3.py            "$STATE" ../tables > /dev/null
python3 followup2_famH.py   "$STATE" ../tables > /dev/null
python3 make_tables.py      "$STATE" ../tables
python3 make_figures.py     "$STATE" ../figures
cd ..
python3 specs/verify_specs.py
echo "Done. Outputs: analysis/*.json, tables/*.tex, figures/*.pdf|png"
