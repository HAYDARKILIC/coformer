# CoFormer -- publication artifact bundle

Generated 2026-10-04T16:58:20 on NVIDIA A100-SXM4-40GB
FULL_SCALE=True | n_vars=16 | n_clauses=64 | ratio=4.0 | seeds=12 | health=INTERPRETABLE (see run_health.txt)

Every table and figure below is regenerated end-to-end by a single Run All of the notebook; `raw_results_matrix.csv` contains the per-(condition, seed) rows from which all aggregates are derived.

## results_table.tex
Table 1. Main ablation results, mean over seeds. `Faith excess' is clause-F1 minus the closed-form score of an information-free CNF at the density the model emitted, g*d/(g+d); raw F1 alone is confounded by density. `CSat-norm' is gold-clause satisfaction rescaled so that a random assignment scores 0 and a perfect one 1. All conditions are trained and differ in exactly one mechanism.

## coformer_results.pdf
Figure 1. (a) In-distribution faithfulness and end-task quality by condition, with across-seed standard deviations. (b) CSat-norm on the three evaluation sets; the sets are unordered categories, not a trajectory. (c) Emitted literal density against the gold density; red marks conditions flagged as constant or information-free predictors.

## confirmatory_holm.csv
Table 2. Planned confirmatory tests (H1-H5; plan fixed in code, no external time stamp), paired over seeds, Holm-corrected within the synthetic family after the testability screen. Effect sizes are paired Cohen's d with bootstrap CIs.

## h6_ablation_dominance.csv
Table 3. H6 ablation dominance: per-seed share of contrasts in which the full model is not beaten by an ablation, with binomial CIs. Reported outside the Holm family.

## confirmatory_real_data.csv
Table 4. Real-benchmark confirmatory tests (R0-R3) on FOLIO and ProofWriter, Holm-corrected within the real family; R4 is exploratory and excluded from the correction.

## st_bias_probe.csv
Table 2. Straight-through gradient bias. Cosine between the straight-through gradient and the soft-relaxation gradient at identical Gumbel noise, and the ratio of their norms, swept over the annealing temperature, at initialisation. For the task loss alone the estimator is partly aligned with the relaxed gradient; with the faithfulness term included it is aligned.

## cases_no_stgs.csv
Qualitative cases (first six ID instances, seed 0). frozen, coformer and no_stgs give the same outcome on every case; the residual errors are refinement failures, not translation failures.

## hparam_sensitivity.csv
Sensitivity of the straight-through conditions to the Gumbel temperature and the number of refinement steps, single seed. Exact solve rises with the number of refinement steps at a translation held at the ceiling; the final Gumbel temperature changes no metric.

## objective_anchoring.csv
Objective-anchoring diagnostic. Clause-F1 and two solver losses for gold, corrupted and random clause matrices: the self-referential loss (satisfy your own clauses) and the anchored loss (satisfy the gold clauses). The sign of the correlation between fidelity and loss is the quantity of interest.

## real_benchmark_results.csv
Part B rule-based abstaining floor on FOLIO and ProofWriter, with the label distribution used to verify the parser.

## completeness_probe.txt
Sensitivity of the relaxation as an unsatisfiability detector on Z3-certified UNSAT instances (threshold 0.5 on the minimum soft clause satisfaction). Every decision is made by Z3 because Z3 is sound; the relaxation is used as a training signal only.

## lambda_sweep.csv
Sensitivity of the result to the faithfulness weight lambda_faith.

## density_sweep.csv
Sensitivity to the anti-collapse density hinge; no setting, including lambda=0, collapsed.

## ratio_sweep.csv
Instance-hardness sweep over the clause/variable ratio, frozen vs coformer.

## negative_control.csv
Shuffled-label negative control: both shuffled runs emit empty formulas.

## contamination_probe.txt
Zero-shot and premise-ablated memorisation probes on the backbone.

## faith_validation_folio.csv
Human-aligned signed-atom faithfulness against gold FOL on FOLIO.

## requirements.txt
Resolved dependency versions for the run that produced these artifacts.

## run_metadata.json
Device, platform, full configuration and derived quantities.

## run_health.txt
Automated health check gating whether these numbers are reportable.

## partc_health.txt
Part C health audit (abstention, fallback, adapter movement, label signal).

## plan_III.json
Plan III specification with its SHA-256, written before any Plan III run.

## results_plan_iii.csv
Plan III per-run results (answer supervision, K, task weight, budget).

## confirmatory_plan_iii.csv
Plan III tests: A2 confirmatory (exact sign-flip, Holm); A1, K, W, B diagnostic.

## plan_iii_learning_curves.csv
Plan III budget stage: ID excess every third epoch (first 512 ID instances).

## plan_iii_nl.csv
Plan III family D: direct-answering and chain-of-thought baselines (descriptive).

## Not produced by this run
- capacity_probe.txt

(These come from optional stages: RUN_EXTRA_DIAGNOSTICS sweeps and Part C. Their absence is recorded rather than hidden.)
