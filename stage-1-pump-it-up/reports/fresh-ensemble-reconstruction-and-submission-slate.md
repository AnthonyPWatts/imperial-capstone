# Fresh ensemble reconstruction and submission slate

## Decision

The 5 September evidence does not support abandoning the base modelling path.
It supports abandoning further row-targeted postprocessing and treating the
remaining architecture changes conservatively.

The locked six-component archive was reconstructed on one fresh five-fold
partition over all **59,400** labelled rows. Its fresh OOF accuracy was
**82.0303%**. No challenger achieved the predeclared **+0.1000 percentage-point**
formal improvement gate. Exactly two one-factor changes, and their one
predeclared cross, met the weaker exploratory guard:

1. split the CatBoost slot equally between complete-identity and spatial-grid
   CatBoost;
2. split the Random Forest slot equally between the incumbent and the fixed
   `max_features=0.3`, leaf-1 variant;
3. apply both changes together.

These three candidates form a fixed exploratory submission slate. The files
were generated, validated and submitted in their fixed order on 5 September.
All three beat the **0.8298** incumbent. The CatBoost identity/spatial-grid bag
scored best at **0.8304**, becoming the new public incumbent.

## Why this is a follow-up, not a new search direction

The preceding [generalisation diagnosis](generalisation-diagnosis.md) found:

- development and public ordering agreed across eight formal base candidates
  (Spearman **1.000**, Pearson **0.934**);
- train-versus-competition covariate differences were small;
- the fresh deep-XGBoost OOF score was **81.5842%** overall, **81.0438%** on
  historical-local IDs and **81.7193%** on all other IDs;
- the historical-local minus other contrast was **-0.6755 percentage points**,
  with a 95% bootstrap interval of **-1.4078 to +0.0463 points**; and
- all three 4 September row-level repair overrides improved reused internal
  evidence but reduced the public score from **0.8298** to **0.8296**,
  **0.8293** and **0.8294**.

The historical subset is therefore moderately harder, but there is no evidence
of a broken test population or a general reversal of the model ranking. The
stronger diagnosis is adaptive overfit in the late row-level search. This
follow-up consequently tested only predeclared model-level changes on fresh
folds; it did not search competition rows.

## Locked protocol and gates

The exact reconstruction used a stratified five-fold split with seed
`20260905` and fingerprint
`1599c84e2f85e2813aa7eec0a2fcc9535a3ea2a717c5c849db6b9212dd3812c3`.
Every OOF row was predicted by a model that excluded it. Historical-local
membership was neither reconstructed nor consulted, and competition data was
not opened while selecting the slate.

The incumbent weights were held fixed:

| Component | Weight |
| --- | ---: |
| Deep top-50 identity XGBoost | 0.33 |
| Spatial-height XGBoost | 0.11 |
| Current Random Forest | 0.18 |
| Categorical-frequency Random Forest | 0.09 |
| Spatial-height Random Forest | 0.09 |
| Complete-identity CatBoost | 0.20 |

The formal ensemble gate required all four of the following:

- mean accuracy change at least **+0.1000 percentage points**;
- wins on at least **3 of 5** folds;
- worst-fold accuracy change no lower than **-0.2500 points**; and
- repair-class recall change no lower than **-2.0000 points**.

The predeclared exploratory guard required a strictly positive exact-ensemble
change, at least **3 of 5** fold wins, a worst-fold change no lower than
**-0.1000 points**, a repair-recall change no lower than **-1.0000 point**, and
eligible component-level evidence. At most one candidate per model family could
survive. A combined candidate was allowed only because two orthogonal families
survived; its composition was fixed before its result was inspected. The
ten-seed deep bag was formal-gate-only because its component gate failed.

## Bounded component follow-ups

The fresh component screens were deliberately small and one-factor. Changes
below are percentage points relative to the named incumbent.

| Candidate comparison | Accuracy change | Fold wins | Worst fold | Repair-recall change | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Spatial-grid CatBoost vs complete identity | +0.0673 | 3/5 | -0.2778 | — | Failed component gate |
| 50:50 identity/spatial-grid CatBoost bag vs identity | +0.1852 | 3/5 | -0.0842 | -0.4864 | Passed its locked component gate |
| Normalised vs raw top-50 identity XGBoost | -0.0118 | 2/5 | -0.0842 | -0.0926 | Rejected |
| Corrected `1/0.9` CatBoost refit scale vs unscaled | -0.0236 | 2/5 | -0.2273 | +0.1390 | Rejected |
| Extra Trees leaf 2 vs current Random Forest | -0.1094 | 1/5 | -0.3367 | +0.2781 | Rejected |
| Random Forest `max_features=0.3`, leaf 1 vs current | +0.1061 | 4/5 | -0.0505 | +0.0693 | Passed formal component gate |
| Random Forest `max_features=0.3`, leaf 2 vs current | +0.6077 | 5/5 | +0.3199 | -2.8261 | Rejected by repair guard |
| 50:50 current/leaf-1 Random Forest bag vs current | +0.0909 | 5/5 | +0.0084 | +0.2546 | Near miss: below +0.1000 gate |
| 50:50 current/leaf-2 Random Forest bag vs current | +0.4966 | 5/5 | +0.3114 | -1.2047 | Passed its fresh bag gate |
| Two-seed deep bag vs seed `20260824` | +0.0522 | 3/5 | -0.0084 | approximately 0 | Stable positive near miss |
| Two-seed deep bag vs seed `20260905` | +0.0387 | 4/5 | -0.0421 | -0.0463 | Stable positive near miss |
| Ten-seed deep bag vs seed `20260824` | +0.0488 | 4/5 | -0.0084 | +0.1389 | Failed +0.1000 gate |
| Ten-seed deep bag vs fixed two-seed bag | -0.0034 | 3 wins, 1 loss, 1 tie | -0.1431 | +0.1389 | Failed gate against both baselines |

The ten-seed test was particularly useful as a negative control. Its seeds,
equal 0.10 weights and two comparisons were locked before fitting; no individual
seed, prefix, bag size or weight was scored for selection. Forty new fold models
were fitted and the two existing seeds were reused. More averaging did not
improve on the fixed two-seed bag.

Legacy-fold exact-archive confirmations were also uniformly negative: the
CatBoost bag changed accuracy by **-0.0126 points**; direct leaf-1 and leaf-2
Random Forest substitutions by **-0.0210** and **-0.0652 points**; their
within-slot bags by **-0.0400** and **-0.0547 points**; and the two-seed deep bag
by **-0.0231 points**. These reused folds were retained as a stability check,
not allowed to override the fresh full-label reconstruction.

## Fresh exact-ensemble results

All rows below use the same fresh incumbent accuracy of **82.0303%**. Accuracy
and recall changes are percentage points. “Net” is additional correct OOF rows
out of 59,400.

| Exact archive candidate | Accuracy | Change | Fold wins | Worst fold | Repair-recall change | Net | Formal | Exploratory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| CatBoost identity/grid 50:50 within its 0.20 slot | 82.0522% | +0.0219 | 4/5 | -0.0253 | -0.1621 | +13 | No | **Yes** |
| Current/leaf-1 Random Forest 50:50 within its 0.18 slot | 82.0455% | +0.0152 | 3/5 | -0.0253 | -0.0463 | +9 | No | **Yes** |
| Current/leaf-2 Random Forest 50:50 within its 0.18 slot | 82.0488% | +0.0185 | 4/5 | -0.1094 | -0.1854 | +11 | No | No |
| Two-seed deep XGBoost 50:50 within its 0.33 slot | 82.0051% | -0.0253 | 1/5 | -0.0673 | -0.0927 | -15 | No | No |
| Ten-seed equal deep bag replacing the 0.33 slot | 82.0202% | -0.0101 | 2/5 | -0.0926 | -0.0464 | -6 | No | No |
| Combined CatBoost and leaf-1 Random Forest changes | 82.0522% | +0.0219 | 3/5 | -0.0589 | -0.1853 | +13 | No | **Yes** |

The leaf-2 Random Forest bag illustrates why every guard matters: its mean was
positive, but its **-0.1094-point** worst fold missed the exploratory boundary.
The two deep bags were negative in the exact ensemble. No outcome-dependent
weight change or replacement was attempted.

The combined candidate has the same rounded mean and net gain as the CatBoost
candidate, but it is not the same prediction: it disagreed with the incumbent
on **332** OOF rows, compared with **272** for CatBoost alone. It also had a
different fold profile. It remains in the slate as the single predeclared
cross, not because a new combination search was run.

## Fixed exploratory submission plan

The three candidates are ordered and fixed as follows:

1. combined CatBoost identity/grid and current/leaf-1 Random Forest within-slot
   bags;
2. CatBoost identity/grid within-slot bag only; and
3. current/leaf-1 Random Forest within-slot bag only.

The generated files and completed submissions are:

| Order | Candidate file | SHA-256 | Disagreements vs incumbent | Submission | Public score | Change vs 0.8298 |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `01-fresh-catboost-rf-combined-50-50-within-slot-bags.csv` | `84454200b1f6182ed7b971163edb824207540f72e27782cec7db5c1d6b4dfb1f` | 68 | `321153` | 0.8302 | +0.0004 |
| 2 | `02-fresh-catboost-identity-grid-50-50-within-slot-bag.csv` | `e492c7ca37ec12750809016a44ecf27fedafacda1eecb3e2db7c8e6368fa446b` | 57 | `321154` | **0.8304** | **+0.0006** |
| 3 | `03-fresh-rf-current-features-0-3-50-50-within-slot-bag.csv` | `d04b4b222f45b8e5285daa730ee2974bd8f921cd6afa1dd05d84dd6925c05fb2` | 35 | `321155` | 0.8303 | +0.0005 |

Pairwise hard-label disagreements are **37** rows for combined versus
CatBoost, **49** for combined versus Random Forest, and **62** for CatBoost
versus Random Forest. None of the files is redundant with the incumbent or
another slate member. Each has exactly **14,850** rows and unique IDs in the
submission-template order, valid class labels and **zero** post-hoc row
changes.

All three are exploratory because their exact fresh gains are only **9 to 13
rows**, well below the formal gate's approximately 60-row minimum. The fresh folds
reduce reuse bias but do not make such small differences certain. The slate is
therefore a bounded test of two architecture hypotheses, not a claim that the
incumbent has been superseded.

The submission run remained non-adaptive: all three validated files were
submitted in the stated order without changing a later file in response to an
earlier score. There was no row-level correction, score-driven weight search or
replacement candidate. After all three submissions, the live page showed a
best score of **0.8304**, rank **#1**, all **3/3** daily slots used and **21**
total submissions. The next slot becomes available on **6 September 2026
(UTC)**.

The public outcome strengthens the architecture-level conclusion: each bounded
model-level change transferred better than the incumbent, while the CatBoost
identity/spatial-grid bag was best. It does not turn the public leaderboard into
row-level evidence, remove uncertainty about the hidden labels, or justify
reopening adaptive micro-postprocessing. The three candidates remain correctly
described as exploratory because none passed the formal fresh-OOF gate.

## Course-aligned lifecycle

| Step | Application in this follow-up |
| --- | --- |
| 1. **Define the goal and scope** | Determine whether late public regressions imply a wrong base path or validation overfit, then test a bounded set of model-level alternatives. Success means fresh, paired evidence under locked gates; the three public slots are confirmatory and exploratory. |
| 2. **Gather the data** | Use the supplied 59,400 labelled rows, supplied 14,850 competition predictors, frozen model artefacts and aggregate public scores. Use no external data. |
| 3. **Explore the data** | Audit internal/public ordering, train/competition shift, historical-local difficulty, component diversity and paired fold changes. |
| 4. **Clean and preprocess the data** | Reuse each archived component's validated preprocessing. Validate IDs, fold alignment, probability shape and class ordering before blending. |
| 5. **Select and engineer features** | Limit new representations to the predeclared spatial grids and fixed Random Forest feature-sampling change; reject normalisation and refit-scale changes that failed fresh evidence. |
| 6. **Define the machine-learning task** | Preserve three-class classification evaluated by accuracy, with repair-class recall as a safety guard. |
| 7. **Partition the data** | Use one fresh stratified five-fold partition over all labels, seed `20260905`; do not recreate or tune on the historical-local split. |
| 8. **Select and train candidate methods** | Fit only the locked CatBoost, Random Forest and XGBoost recipes. Use fixed 50:50 within-slot bags and one equal ten-seed bag; do not search weights. |
| 9. **Evaluate and interpret the results** | Compare complete OOF probabilities with paired mean, fold-win, worst-fold, repair-recall and net-correct metrics. No candidate passed the formal gate; three met the exploratory rules. |
| 10. **Deploy and iterate** | The fixed slate was generated, validated and submitted without adaptation after explicit approval. All outcomes were recorded; the CatBoost identity/spatial-grid bag scored 0.8304 and replaced the 0.8298 public incumbent. |

## No-cheating boundary

Only competition-supplied predictors may be used for final inference. The
[competition reference](../instructions/competition-reference.md#rules-relevant-to-the-project)
records the prohibition on external data. Training labels are used only inside
proper training folds or full-data final refits. Competition labels are unknown
and are never inferred row by row from leaderboard movements.

Aggregate public scores were used only to assess the three already-fixed files.
They were not converted into hidden labels, used for row-wise probes, or used to
alter the slate. The investigation used no external labels, pseudo-labels,
manual web lookup, ID shortcut or target-bearing preprocessing.

## Reproduction and retained evidence

Run from the project root. The commands carrying `--cache-only` validate and
replay retained expensive evidence without silently fitting missing folds. The
ten-seed command validates and reuses compatible caches but will fit any missing
seed-fold cache. The final line is the original explicit-opt-in, one-shot
portfolio-generation command; now that the output exists, its no-overwrite
guard deliberately refuses a repeat run:

```powershell
$py = ".\.venv\Scripts\python.exe"

& $py .\stage-1-pump-it-up\scripts\run_generalisation_diagnosis.py --cache-only
& $py .\stage-1-pump-it-up\scripts\run_deep_xgboost_ten_seed_bag.py
& $py .\stage-1-pump-it-up\scripts\run_fresh_deep_archive_reconstruction.py --cache-only
& $py .\stage-1-pump-it-up\scripts\generate_fresh_reconstruction_portfolio.py --opt-in generate-admissible-fresh-reconstruction-portfolio
```

The remaining bounded screens can be replayed with their correspondingly named
scripts:

```powershell
& $py .\stage-1-pump-it-up\scripts\run_fresh_spatial_grid_catboost_comparison.py
& $py .\stage-1-pump-it-up\scripts\run_normalised_top_common_screen.py
& $py .\stage-1-pump-it-up\scripts\run_catboost_refit_scale_screen.py
& $py .\stage-1-pump-it-up\scripts\run_fresh_rf_family_screen.py
& $py .\stage-1-pump-it-up\scripts\run_locked_catboost_probability_bag_replay.py
& $py .\stage-1-pump-it-up\scripts\run_locked_catboost_bag_ensemble_confirmation.py
& $py .\stage-1-pump-it-up\scripts\run_locked_rf_probability_bag_confirmation.py
& $py .\stage-1-pump-it-up\scripts\run_locked_rf_ensemble_confirmation.py
& $py .\stage-1-pump-it-up\scripts\run_deep_xgboost_seed_average.py
& $py .\stage-1-pump-it-up\scripts\run_deep_xgboost_frozen_ensemble_confirmation.py
```

The authoritative final comparison is
`.runtime/fresh-deep-archive-reconstruction-seed-20260905/result.json`. Its
source, input-evidence and output hashes bind the result to the labelled data,
component caches and generated tables. Supporting results are retained under
`.runtime/generalisation-diagnosis/`, `.runtime/fresh-spatial-grid-catboost/`,
`.runtime/fresh-rf-family-screen-seed-20260905/`,
`.runtime/deep-xgboost-two-seed-variance-check/` and
`.runtime/deep-xgboost-ten-seed-bag-fresh-20260905/`.

The generated CSVs and strict manifest are under
`stage-1-pump-it-up/submissions/2026-09-05-fresh-reconstruction-portfolio/`.
The manifest binds the evidence, data, code, incumbent inputs, replacement
component caches, candidate order, output hashes and distinctness audit.
The three files were subsequently submitted as IDs `321153`, `321154` and
`321155`; their recorded scores are **0.8302**, **0.8304** and **0.8303** in
the same order.
