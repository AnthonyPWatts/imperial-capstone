# Training-only outlier-filtering screen

## Decision

Retain every training row. None of four bounded outlier filters improved the
accepted 55% XGBoost / 45% Random Forest ensemble's 81.625% frozen-fold
accuracy. The closest candidate, strict removal of minority labels within exact
accepted-feature duplicate groups, reached 81.566%: 0.059 percentage points
below baseline despite three very small fold wins.

The two genuinely unsupervised Isolation Forest policies lost on all five
folds. Numerically isolated pumps are difficult observations, but the unchanged
validation population shows that they are useful training evidence rather than
disposable noise. The local test remained closed and no competition prediction
was generated.

## Course-aligned lifecycle

| Step | Application in this experiment |
| --- | --- |
| 1. Define the goal and scope | Test whether conservative removal of anomalous training rows improves competition-aligned three-class accuracy without changing the validation population. |
| 2. Gather the data | Reuse the frozen 47,520-row development partition and accepted component evidence. Do not inspect the local-test labels or competition outcomes. |
| 3. Explore the data | Audit physical contradictions, exact raw and accepted-feature duplicates, robust numeric support, categorical rarity and geographic distance. |
| 4. Clean and preprocess the data | Fit each filter only on the current outer-training partition; retain the accepted fold-fitted model preprocessing after row filtering. |
| 5. Select and engineer features | Use the accepted feature frame. Isolation Forest receives eight continuous support features, with positive skew log-transformed, missing values median-filled and values scaled by training-fold interquartile range. |
| 6. Define the machine-learning task | Training-data editing followed by the unchanged supervised three-class classification task. |
| 7. Partition the data | Reuse the five frozen stratified development folds. Never filter or omit validation rows. |
| 8. Select and train candidate methods | Compare four predeclared filters. Refit the accepted components using the accepted XGBoost tree count for each fold and unchanged Random Forest settings. |
| 9. Evaluate and interpret the results | Apply the established accuracy, fold-win, worst-fold and repair-recall gate; also inspect probability quality, class-specific removal and fixed component crosses. |
| 10. Deploy and iterate | Stop outlier removal for this representation. Move to a materially different, cross-fitted high-cardinality target-encoding experiment. |

## Reconnaissance and candidate contract

Only seven development rows have an unambiguous physical contradiction: a
construction year later than the recording year. No negative values occur in
the fields that must be non-negative. Exact raw-row duplication is also too
sparse to be useful: 47 rows belong to raw duplicate groups, only one group has
conflicting labels, and no group meets the strict consensus rule.

The accepted engineered representation has 874 duplicate rows and 77
conflicting groups. A full-development audit identifies 47 minority labels in
groups of at least three rows with at least two-thirds agreement. This audit
motivated the rule, but the production filter recomputes every group using only
the current outer-training rows.

The four fixed policies were:

| Policy | Training-fold rule | Mean rows removed |
| --- | --- | ---: |
| Physical contradictions | Future construction year or negative amount, population or `num_private` | 5.6 |
| Strict duplicate conflicts | Exact accepted-feature group, at least three rows, at least two-thirds modal-label purity; remove only non-modal labels | 36.2 |
| Isolation Forest 0.5% | Class-blind numeric-support Isolation Forest with fixed seed and 0.5% contamination | 191.0 |
| Isolation Forest 1.0% | Same detector at 1.0% contamination | 381.0 |

Every rule is capped at 2% removal and must retain all target classes. Filter
statistics and category consensus never see a validation row. The validation
fold is scored in full, including its own anomalies.

## Results

| Policy | Mean accuracy | Change | Fold wins | Worst fold | Repair-recall change | Log-loss change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Accepted ensemble | **81.625%** | — | — | — | — | — |
| Strict duplicate conflicts | 81.566% | -0.059 pp | 3/5 | -0.179 pp | -0.029 pp | +0.000850 |
| Physical contradictions | 81.498% | -0.126 pp | 0/5 | -0.232 pp | -0.058 pp | +0.000009 |
| Isolation Forest 1.0% | 81.477% | -0.147 pp | 0/5 | -0.210 pp | 0.000 pp | +0.000677 |
| Isolation Forest 0.5% | 81.395% | -0.229 pp | 0/5 | -0.305 pp | +0.145 pp | +0.000066 |

The strict duplicate candidate's three wins are only +0.011, +0.021 and +0.011
points. Its other folds lose 0.158 and 0.179 points, so this is not a stable
near miss. Duplicate filtering removes an average of 21.4 repair rows, 8.0
functional rows and 6.8 non-functional rows per training fold. Isolation
Forest removal is much closer to the original class balance.

The accepted component accuracies are 81.014% for XGBoost and 80.591% for
Random Forest. Strict duplicate filtering slightly raises Random Forest to
80.631% but lowers XGBoost to 80.966%. Eight fixed, no-refit crosses paired one
filtered component with its unchanged accepted partner. All were below
baseline; the best was accepted XGBoost plus the 0.5% Isolation Forest Random
Forest at 81.572%, down 0.053 points with one fold win.

No candidate passes the established gate of at least +0.10 points mean
accuracy, at least three fold wins, no fold worse than -0.25 points and no more
than two points of repair-recall loss.

## Interpretation

Outlier cohorts have lower OOF accuracy, but that does not imply they should be
removed. A model must still predict difficult and unusual pumps at inference
time. Removing their closest training analogues makes that task harder. Even
the seven obvious construction-year contradictions carry useful information in
their other predictors; the feature engineer already masks the contradictory
year and exposes an inconsistency flag, which is safer than dropping the row.

The duplicate-conflict result makes the same point for possible label noise.
Those minority labels are usually misclassified, yet removing them does not
improve generalisation. Exact equality in the model representation means the
observed predictors cannot distinguish the outcomes; it does not prove that the
minority condition is erroneous.

Stop broad outlier filtering for the accepted representation. Do not tune
contamination values, remove low-confidence rows, or filter validation data to
manufacture a higher score.

## Next experiment

The most credible remaining source of materially new signal is supervised,
cross-fitted encoding of currently deferred high-cardinality fields such as
`ward`, `subvillage`, `wpt_name` and `scheme_name`, possibly alongside the
already audited `funder` and `installer` fields. The encoder must create inner
out-of-fold values for outer-training rows and use mappings fitted only on the
complete outer-training partition for outer validation. Global or outer-fold
target means copied back onto their own training rows would leak the labels.

Start with one smoothed multiclass target-encoding policy and a conservative
field set chosen before evaluation. Compare the fixed accepted components and
their standard 55:45 vote. This is more promising than another filter because
it adds information instead of deleting hard examples.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_outlier_filtering_screen.py
```

Use `--force` only when intentionally refitting all twenty component folds.
Cached evidence and summaries live under the ignored
`.runtime/outlier-filtering-screen/` directory.
