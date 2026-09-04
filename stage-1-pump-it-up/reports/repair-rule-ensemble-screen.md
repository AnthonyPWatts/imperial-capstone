# Repair-rule ensemble screen

## Outcome

The fixed six-rule union recovered **23 net OOF rows** and **8 net local-test rows** by changing 65 and 25 incumbent `functional` decisions respectively to `functional needs repair`. Every development fold improved. After that union scored 0.8296 publicly, a stricter candidate retained only the four non-scheme history rules and added the independent supported-subvillage history rule.

| Candidate | OOF flips | OOF R/F/N (net) | OOF accuracy | OOF delta | Local flips | Local R/F/N (net) | Local accuracy | Local delta | Competition flips |
| --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| full_union | 65 | 39/16/10 (+23) | 0.819465 | +0.000484 | 25 | 14/6/5 (+8) | 0.814646 | +0.000673 | 22 |
| history_only | 47 | 29/12/6 (+17) | 0.819339 | +0.000358 | 15 | 10/3/2 (+7) | 0.814562 | +0.000589 | 14 |
| strict_history_core_plus_residual_history | 26 | 19/3/4 (+16) | 0.819318 | +0.000337 | 9 | 6/1/2 (+5) | 0.814394 | +0.000421 | 12 |

The full-union OOF accuracy changes from 0.818981 to 0.819465; local accuracy changes from 0.813973 to 0.814646. Its repair recall moves from 34.34% to 35.47% OOF and from 33.02% to 34.65% locally. The exact McNemar p-values are 0.0027 and 0.1153; this is promising but still a small, adaptively selected override.

## Fixed rule contract

All rules can act only when the incumbent predicts `functional`. History support counts only `functional` and `functional needs repair`; `non functional` is excluded. OOF history is leakage-safe: each validation fold uses a mapping fitted on the other four folds.

| Rule | Key | Minimum repair rate | Minimum support | Probability guard |
| --- | --- | ---: | ---: | --- |
| scheme_name_history | scheme_name | 0.35 | 50 | final ensemble repair/functional >= 0.60 |
| grid_005_history | grid_005 | 0.75 | 3 | final ensemble repair/functional >= 0.90 |
| lga_subvillage_history | lga_subvillage | 1.00 | 3 | final ensemble repair/functional >= 0.50 |
| exact_coordinate_name_history | exact_coordinate_name | 0.45 | 1 | final ensemble repair/functional >= 0.90 |
| subvillage_history | subvillage | 1.00 | 3 | final ensemble repair/functional >= 0.50 |
| identity_catboost_conflict | none | n/a | n/a | identity CatBoost repair/functional >= 2.00 |

## Fold and local contingency

Cells are actual repair / functional / non-functional among flipped rows, followed by net accuracy contribution. The union rows include overlap de-duplication.

| Rule | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Local |
| --- | --- | --- | --- | --- | --- | --- |
| scheme_name_history | 4/3/1 (+1) | 2/3/0 (-1) | 3/1/0 (+2) | 1/0/1 (+1) | 4/3/1 (+1) | 5/2/2 (+3) |
| grid_005_history | 1/0/0 (+1) | 1/0/0 (+1) | 2/0/0 (+2) | 0/0/0 (+0) | 1/0/0 (+1) | 1/0/0 (+1) |
| lga_subvillage_history | 2/0/0 (+2) | 1/0/0 (+1) | 0/0/0 (+0) | 1/0/1 (+1) | 3/2/2 (+1) | 2/1/1 (+1) |
| exact_coordinate_name_history | 0/0/0 (+0) | 1/0/0 (+1) | 0/0/0 (+0) | 1/0/0 (+1) | 0/0/0 (+0) | 2/0/0 (+2) |
| subvillage_history | 0/0/0 (+0) | 1/0/0 (+1) | 0/0/0 (+0) | 1/0/1 (+1) | 3/1/2 (+2) | 2/1/1 (+1) |
| identity_catboost_conflict | 2/1/2 (+1) | 2/0/0 (+2) | 2/2/0 (+0) | 3/1/0 (+2) | 1/0/2 (+1) | 5/4/3 (+1) |
| history_only_union | 7/3/1 (+4) | 5/3/0 (+2) | 5/1/0 (+4) | 3/0/2 (+3) | 9/5/3 (+4) | 10/3/2 (+7) |
| full_union | 9/4/3 (+5) | 7/3/0 (+4) | 7/3/0 (+4) | 6/1/2 (+5) | 10/5/5 (+5) | 14/6/5 (+8) |

## Competition contribution and overlap

Incremental counts follow the fixed rule order above. Pairwise overlap matrices and the row-level trigger audit are saved under `.runtime/repair-rule-ensemble/`.

| Rule | Standalone | Globally unique | Incremental | Overlaps prior | Exact IDs |
| --- | ---: | ---: | ---: | ---: | --- |
| scheme_name_history | 7 | 5 | 7 | 0 | 18687, 49925, 14308, 32047, 22073, 73994, 55396 |
| grid_005_history | 0 | 0 | 0 | 0 | none |
| lga_subvillage_history | 7 | 2 | 6 | 1 | 49293, 17723, 43614, 60913, 45330, 72067, 55396 |
| exact_coordinate_name_history | 1 | 1 | 1 | 0 | 36828 |
| subvillage_history | 5 | 0 | 0 | 5 | 49293, 17723, 45330, 72067, 55396 |
| identity_catboost_conflict | 9 | 8 | 8 | 1 | 60880, 60481, 65787, 29101, 36347, 32047, 50023, 21096, 13565 |

The history-only candidate changes these 14 IDs:

`[49293, 18687, 17723, 43614, 49925, 14308, 60913, 32047, 22073, 36828, 45330, 72067, 73994, 55396]`

The full union changes these 22 IDs:

`[60880, 49293, 18687, 60481, 17723, 43614, 49925, 14308, 60913, 65787, 29101, 36347, 32047, 22073, 36828, 45330, 50023, 21096, 72067, 73994, 13565, 55396]`

History-only is a strict subset of full union; the identity conflict adds eight unique competition rows. The grid rule triggers no competition row, and the strict subvillage rule adds no row beyond the LGA-subvillage rule on this test set, although both have independent OOF evidence.

## Post-result strict candidate

The strict history core excludes the weaker `scheme_name_history` and identity-conflict blocks. Its union with the independently defined supported-subvillage history rule changes 12 competition rows, of which eight overlap the submitted full union and four are new. It recovers 16 net OOF rows with fold nets `[3, 3, 1, 4, 5]`, and 5 net local rows.

Exact IDs:

`[9086, 49293, 17723, 43614, 60913, 63460, 22746, 36828, 45330, 72067, 33360, 55396]`

## Generated candidates

- `02-repair-rule-history-only.csv` — SHA-256 `53bdf901c2771147d1ff939c0b3e1829d6f696e197fcf81c18108ba4ee9185e5`.
- `01-repair-rule-full-union.csv` — SHA-256 `01c8fa597f57c785009dc89ea2327ba3a43ca0bb3c0e5d7a25386e8f78acbe87`.
- `03-strict-history-core-plus-residual-history.csv` — SHA-256 `bf6966a5fe3b3a50ed2e3cf58a5d8bed8c219a66ad230281a0a8e985cf380d91`.

All files contain 14,850 unique IDs in template order, no missing or invalid labels, and only the intended `functional` to `functional needs repair` transitions.

## Public result

The full union was submitted unchanged on 4 September 2026 as submission `321013` and scored **0.8296**, below the 0.8298 incumbent. If all 14,850 competition rows are scored, four-decimal rounding makes this exactly a loss of two to four correct rows. The evaluator denominator is not disclosed locally, so this does not identify any row label. The history-only and strict candidates remain unsubmitted at this checkpoint.

## Interpretation

This is a targeted correction rather than a replacement model. The original positive direction replicated internally but failed to transfer for the complete 22-row union. The 12-row candidate is explicitly post-result and therefore adaptive: its rationale is to retain the strongest internal history block and add four rows from a separately defined rule outside the failed union, not to claim knowledge of hidden row labels.
