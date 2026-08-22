# Deferred-name character text screen

## Decision

Stop character n-gram modelling of the deferred names. A fold-fitted logistic
model over the accepted features plus 60,000 character TF-IDF features reaches
**75.619%** standalone. Its single predeclared 5% contribution lowers the
promoted identity vote from 81.740% to **81.705%**, wins only **1/5** folds and
reduces repair recall by **0.782 percentage points**.

No vectoriser, regularisation or weight grid was opened. No local-test or
competition prediction was generated.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test whether spelling fragments generalise across unseen exact names and add diverse ensemble signal. |
| 2. Gather the data | Reuse `funder`, `installer`, `wpt_name`, `subvillage`, `ward` and `scheme_name`. |
| 3. Explore the data | Use the prior audit showing 56.90% unseen exact pump names and normalisation fragmentation. |
| 4. Clean and preprocess the data | Casefold, trim and collapse whitespace; preserve tagged fields, missing and blank states. |
| 5. Select and engineer features | Add one fold-fitted 3–5 character TF-IDF representation capped at 60,000 features to the accepted frame. |
| 6. Define the machine-learning task | Unchanged probabilistic three-class classification. |
| 7. Partition the data | Fit the vocabulary, IDF weights and classifier separately inside each frozen outer-training fold. |
| 8. Select and train candidate methods | Fit one L2 logistic specification and one fixed 5% contribution. |
| 9. Evaluate and interpret the results | Compare standalone and blended performance with the promoted identity vote. |
| 10. Deploy and iterate | Reject the text representation without opening local test. |

## Fixed representation

Each row becomes six field-tagged tokens after conservative normalisation. The
pipeline unions the accepted scaled one-hot features with word-boundary
character 3–5 grams, minimum document frequency three, sublinear term frequency
and a 60,000-feature cap. An L2 logistic classifier uses `C=1`, SAGA, tolerance
0.001 and at most 300 iterations. It converged in 147–153 iterations across the
five folds, each taking about 22 seconds.

The ensemble contribution is exactly 5%; the promoted 44:36:20 recipe retains
its internal proportions within the other 95%, giving final weights 41.8%
XGBoost, 34.2% Random Forest, 19.0% identity CatBoost and 5.0% text logistic.

## Results

| Candidate | Accuracy | Leader change | Fold wins | Worst fold | Repair-recall change | Passes gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Promoted identity vote | **81.7403%** | — | — | — | — | incumbent |
| 5% name-text vote | 81.7045% | -0.0358 pp | 1/5 | -0.0947 pp | -0.7818 pp | No |
| Name-text logistic | 75.6187% | -6.1216 pp | 0/5 | -6.9865 pp | -22.4383 pp | No |

The text representation is materially different but too weak and directionally
unhelpful. Tuning n-gram widths or a sequence of tiny weights would reuse the
same folds adaptively without evidence that the branch contains useful signal.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_name_text_screen.py
```

Runtime evidence remains ignored under `.runtime/name-text-screen/`.
