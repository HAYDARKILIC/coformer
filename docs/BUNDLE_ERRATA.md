# Errata for the auto-generated output bundle (`outputs/paper_bundle/`)

The notebook writes `MANIFEST.md`, `contamination_probe.txt` and `run_health.txt` automatically, and its markdown cells contain design notes.
Some of their captions and verdict strings were written by earlier revisions of the reporting
code, and several contradict the final analysis. The article and the scripts in `analysis/` of this repository,
which recompute every reported number from the per-run files, take precedence.
The data files themselves (CSV and JSON) are unaffected.

| File / entry | Superseded text | Correct statement (article section) |
|---|---|---|
| MANIFEST: `confirmatory_holm.csv` | "Pre-registered confirmatory tests (H1-H5)" | Planned tests fixed in a code-embedded analysis plan without an external time stamp. H1, H2, H4 and H5 are not testable (vacuous); H3 is not supported. |
| MANIFEST: `confirmatory_real_data.csv` | "Real-benchmark confirmatory tests (R1-R3)" | The confirmatory family is R0-R3. R0, R2 and R3 are floored. R1 is opposite to the prediction and not significant under exact inference at four seeds. |
| MANIFEST: `st_bias_probe.csv` | "the estimator is anti-parallel to the descent direction" | The task-only straight-through gradient is partly aligned with the soft-relaxation gradient (cosine 0.489-0.725); for the combined objective the cosine is 1.000. |
| MANIFEST: `cases_no_stgs.csv` | "the opposite of the straight-through conditions" | frozen, coformer and no_stgs give identical outcomes on all six cases. |
| MANIFEST: `hparam_sensitivity.csv` | "Refinement depth has no effect; lower temperature helps monotonically" | Exact solve rises with T (0.075 at T=10 to 0.427 at T=100; 0.493 at T=200 for no_stgs). The final temperature changes no metric. |
| MANIFEST: `density_sweep.csv` | "lambda=0 reproduces the collapse" | No setting of lambda_d, including 0, produced a collapse. |
| MANIFEST: `negative_control.csv` | "the advantage must collapse here" | Both shuffled runs emitted empty formulas; the control passes. |
| Notebook design note (Cell U, `NOTES_PREREG`) | "essentially never signals unsatisfiability" | The relaxation flagged 81% of certified-UNSAT instances (threshold 0.5, sensitivity only). Z3 decides because it is sound. |
| `contamination_probe.txt`: verdict | "gains from co-training are attributable to the method" | Low leakage risk. No gains from co-training were observed. |
| MANIFEST header | "health=HEALTHY" | The health check's verdict is "interpretable with scoped limitations". |
| `run_health.txt`: verdict | "HEALTHY WITH SCOPED LIMITATIONS -- claims stand" | No confirmatory hypothesis is supported. Seven synthetic contrasts are not testable, and the claims that stand are the negative and diagnostic ones summarised in the article's claims table. |
| File modification times | (local file times) | The file times are 10 h earlier than the notebook's own UTC time stamps (e.g. the run marker and the superseded-file suffix). The article's timeline is corrected for this offset. |
| `requirements.txt` | transformers 5.16.1, peft 0.20.0, accelerate 1.14.0 (written by the first Plan III session, whose natural-language part did not run) | The reported natural-language results and the Plan III family D were produced with transformers 4.46.2, peft 0.13.2 and accelerate 1.1.1; see `requirements.session-20260927T151205Z.txt`. The released `requirements.txt` restores these versions. |
| `partc_skip_reason.txt` | "backbone failed to load: ModuleNotFoundError ..." | Stale record of the first Plan III session. The second session loaded the backbone with the pinned versions and completed family D; no stored natural-language result was changed. |
