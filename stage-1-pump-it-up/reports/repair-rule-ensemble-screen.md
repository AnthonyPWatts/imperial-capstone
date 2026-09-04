# Repair-rule ensemble screen

## Outcome

The fixed six-rule union recovered **23 net OOF rows** and **8 net local-test rows** by changing 65 and 25 incumbent `functional` decisions respectively to `functional needs repair`. Every development fold improved. The history-only subset retained 17 OOF and 7 local net corrections with fewer competition changes, making it the conservative nested hedge.

| Candidate | OOF flips | OOF R/F/N (net) | OOF accuracy | OOF delta | Local flips | Local R/F/N (net) | Local accuracy | Local delta | Competition flips |
| --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| full_union | 65 | 39/16/10 (+23) | 0.819465 | +0.000484 | 25 | 14/6/5 (+8) | 0.814646 | +0.000673 | 22 |
| history_only | 47 | 29/12/6 (+17) | 0.819339 | +0.000358 | 15 | 10/3/2 (+7) | 0.814562 | +0.000589 | 14 |

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

## Generated candidates

- `02-repair-rule-history-only.csv` — SHA-256 `53bdf901c2771147d1ff939c0b3e1829d6f696e197fcf81c18108ba4ee9185e5`.
- `01-repair-rule-full-union.csv` — SHA-256 `01c8fa597f57c785009dc89ea2327ba3a43ca0bb3c0e5d7a25386e8f78acbe87`.

Both files contain 14,850 unique IDs in template order, no missing or invalid labels, and only the intended `functional` to `functional needs repair` transitions.

## Public result

The full union was submitted unchanged on 4 September 2026 as submission
`321013` and scored **0.8296**, below the 0.8298 incumbent. At four-decimal
score precision, the result is consistent with a loss of roughly two to four
correct rows. This contradicts the positive fold and local-test evidence and
shows that the adaptively discovered overrides did not transfer to the hidden
competition labels. Two daily slots remained after this result; the history-only
file remained unsubmitted at this checkpoint.

## Interpretation

This is a targeted correction rather than a replacement model. The positive direction replicated on all five OOF folds and the held-out local test, but the absolute sample is small and the rules were found through exploratory screening. If allocating a slot to this family, the history-only file is the conservative hedge and the full union is the higher-upside variant; submitting both uses two slots on a highly nested hypothesis, so it should be balanced against an independent model candidate if one is available.
