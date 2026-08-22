# Binary reduction screen

## Conclusion

Removing every `functional needs repair` row from training improves the
remaining functional-versus-non-functional decision, but reduces performance
on the unchanged three-class task. The best binary recipe is an equal soft vote
between Random Forest and bagged LightGBM. It reaches **80.086%** development
accuracy and **79.992%** local-test accuracy, with repair recall fixed at zero.

The accepted three-class XGBoost/Random-Forest vote remains selected. On the
development rows, binary reduction gains 473 correct functional or
non-functional decisions but gives up 1,204 correctly identified repair cases.
That is a net loss of 731 correct rows, or **1.538 percentage points**. The
minority class is difficult, but it is not dispensable under the competition's
ordinary-accuracy objective.

No competition prediction or submission was generated.

## Course-aligned lifecycle

| Step | Application in this experiment |
| --- | --- |
| 1. Define the goal and scope | Test the explicit counterfactual that repair cases are removed from model fitting and can never be predicted, while retaining three-class accuracy as the primary measure. |
| 2. Gather the data | Reuse the validated 59,400 labelled rows and their existing development/local-test partition. Competition rows are not used. |
| 3. Explore the data | Quantify repair prevalence, the resulting accuracy ceiling, candidate performance and class-level confusion. |
| 4. Clean and preprocess the data | Reuse the established fold-fitted feature engineering, imputation, rare grouping, encoding and model-specific scaling. |
| 5. Select and engineer features | Keep the accepted feature policy unchanged so that target reduction, rather than a simultaneous feature change, is tested. |
| 6. Define the machine-learning task | Train a binary functional/non-functional classifier, then score its two possible labels against the full nominal three-class target. |
| 7. Partition the data | Keep the frozen five development folds and the earlier local-test partition. Remove repair rows only from each training partition; never from validation or test. |
| 8. Select and train candidate methods | Compare eleven standalone families and nine predeclared soft votes, including accelerated CatBoost, XGBoost and LightGBM candidates. |
| 9. Evaluate and interpret the results | Use full three-class accuracy first, with conditional binary accuracy, per-class recall, log loss, Brier score and confusion counts as diagnostics. |
| 10. Deploy and iterate | Do not deploy or submit: the best binary recipe trails the accepted three-class workflow. Retain it only as evidence about the cost and value of the repair class. |

## Experiment contract

The target manipulation is performed inside each outer training fold. A model
fitted for one fold therefore sees no repair examples and no validation label.
The validation fold remains complete. Binary probabilities are placed in the
three-class column order with repair probability exactly zero, and the maximum
probability is scored against every validation target.

This distinction matters. Removing repair rows before partitioning would make
the task artificially easier and would not answer the question posed. In the
valid experiment, the 3,454 repair rows form 7.2685% of development, imposing a
hard three-class accuracy ceiling of **92.7315%** even if every remaining row is
classified correctly. The local test has a nearly identical 7.2643% repair
share and a 92.7357% ceiling.

Every candidate uses the existing fixed feature policy and model recipe where
possible. Accelerated learners choose their tree count on a 10% split inside
each binary outer-training partition, then refit on all allowed outer-training
rows before scoring that fold. The model families and ensemble recipes were
specified before comparing their results; arbitrary blend weights were not
searched.

## Standalone screen

| Candidate | Three-class accuracy | Conditional binary accuracy | Functional recall | Non-functional recall | Conditional log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost depth 8, child weight 1 | **79.718%** | **85.967%** | 91.603% | 78.000% | 0.3229 |
| Random Forest | 79.672% | 85.917% | 90.309% | **79.709%** | 0.3819 |
| LightGBM, 63 bagged leaves | 79.611% | 85.851% | 91.285% | 78.170% | 0.3247 |
| CatBoost depth 8 | 79.211% | 85.420% | 91.529% | 76.784% | 0.3323 |
| Histogram boosting | 78.479% | 84.630% | **91.913%** | 74.336% | 0.3454 |
| Extra Trees | 78.432% | 84.580% | 88.216% | 79.440% | 0.7429 |
| MLP ReLU 128–64 | 77.519% | 83.595% | 89.987% | 74.560% | 0.3785 |
| KNN | 76.961% | 82.994% | 89.166% | 74.270% | 0.4890 |
| Logistic regression | 74.762% | 80.622% | 89.778% | 67.682% | 0.4234 |
| Constrained decision tree | 74.371% | 80.200% | 90.828% | 65.179% | 0.5115 |
| Gaussian naïve Bayes | 66.789% | 72.024% | 76.592% | 65.568% | 7.3755 |

XGBoost is the strongest standalone candidate. Random Forest is only 0.046
points behind and makes a different class trade-off, improving non-functional
recall at the cost of functional recall. LightGBM is similarly close and has
stronger probability quality than Random Forest. Those differences provide a
rational ensemble opportunity.

Extra Trees required the most conventional-model compute while producing poor
probability quality and no accuracy advantage. CatBoost selected 3,626–3,999
trees across the five folds; four selections were effectively at the bounded
4,000-tree cap. Its early-stopping loss was therefore not clearly converged,
but its current 79.211% accuracy does not justify extending this already costly
candidate as part of the bounded screen.

## Ensemble reconstruction

| Fixed binary vote | Three-class accuracy | Conditional binary accuracy | Change from XGBoost | Fold wins | Worst fold change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50% Random Forest / 50% LightGBM | **80.086%** | **86.364%** | **+0.368 pp** | 5/5 | +0.074 pp |
| 50% XGBoost / 35% Random Forest / 15% LightGBM | 80.084% | 86.361% | +0.366 pp | 5/5 | +0.200 pp |
| 52.25% XGBoost / 42.75% Random Forest / 5% MLP | 80.055% | 86.330% | +0.337 pp | 5/5 | +0.126 pp |
| 55% XGBoost / 45% Random Forest | 80.038% | 86.311% | +0.320 pp | 5/5 | +0.137 pp |
| 50% XGBoost / 35% Random Forest / 15% CatBoost | 80.006% | 86.277% | +0.288 pp | 5/5 | +0.053 pp |
| 50% Random Forest / 50% histogram boosting | 79.941% | 86.207% | +0.223 pp | 4/5 | -0.032 pp |
| 50% Extra Trees / 50% histogram boosting | 79.804% | 86.060% | +0.086 pp | 2/5 | -0.084 pp |
| 50% XGBoost / 50% LightGBM | 79.695% | 85.942% | -0.023 pp | 3/5 | -0.210 pp |
| 50% XGBoost / 50% CatBoost | 79.550% | 85.785% | -0.168 pp | 2/5 | -0.421 pp |

The equal Random-Forest/LightGBM vote passes the established internal promotion
gate relative to the binary XGBoost leader: at least +0.10 points on average,
at least three fold wins and no fold worse than -0.25 points. This selects it
within the binary experiment. It does not override the separate comparison
against the accepted three-class model.

## What the target reduction changes

| Development result | Accepted three-class vote | Selected binary vote | Binary change |
| --- | ---: | ---: | ---: |
| Full three-class accuracy | **81.625%** | 80.086% | **-1.538 pp** |
| Accuracy conditional on non-repair rows | 85.290% | **86.364%** | +1.073 pp |
| Correct functional/non-functional rows | 37,584 | **38,057** | +473 |
| Correct repair rows | **1,204** | 0 | -1,204 |
| Repair recall | **34.859%** | 0% | -34.859 pp |

The experiment therefore works in the narrow sense: after removing the hard
middle class, the classifier becomes better at the remaining boundary. It does
not work for the actual task. The 473 extra correct decisions among allowed
classes are worth 0.995 percentage points over all development rows, while the
1,204 abandoned repair decisions cost 2.534 points. The net is the observed
1.538-point loss.

The selected binary development confusion counts make the behaviour explicit:

| Actual class | Predicted functional | Predicted repair | Predicted non-functional |
| --- | ---: | ---: | ---: |
| Functional | 23,604 | 0 | 2,203 |
| Functional needs repair | 2,584 | 0 | 870 |
| Non-functional | 3,806 | 0 | 14,453 |

## Local-test assessment

After binary selection was frozen, both components were fitted on all 44,066
non-repair development rows and evaluated against all 11,880 labelled local-
test rows.

| Metric | Selected binary vote |
| --- | ---: |
| Full three-class accuracy | **79.992%** |
| Conditional binary accuracy | **86.258%** |
| Functional recall | 91.414% |
| Repair recall | 0% |
| Non-functional recall | 78.970% |
| Conditional log loss | 0.3185 |
| Predicted functional / repair / non-functional | 63.16% / 0% / 36.84% |

| Actual class | Predicted functional | Predicted repair | Predicted non-functional |
| --- | ---: | ---: | ---: |
| Functional | 5,898 | 0 | 554 |
| Functional needs repair | 645 | 0 | 218 |
| Non-functional | 960 | 0 | 3,605 |

For context, the earlier three-class Random-Forest/histogram vote scored
80.825% on this same local test. Binary reduction improves its conditional
functional/non-functional accuracy by 1.616 points and makes 178 more correct
non-repair decisions, then loses all 277 repair decisions the earlier workflow
got right. The net local-test change is **-0.833 points**. Because this labelled
partition has already been used by earlier experiments, it is a useful
consistency check rather than a newly untouched confirmation set.

## Decision

Retain three-class training and the accepted 55% XGBoost / 45% Random Forest
vote. The repair class is small and difficult, but the accepted model's partial
success on it contributes materially more accuracy than the binary reduction
recovers elsewhere.

Stop the binary-only competition loop here. It remains potentially relevant to
a different operational question—for example, prioritising clearly broken
pumps after repairable cases have been triaged by another process—but that
would be a new target, decision policy and evaluation objective rather than a
competition-model improvement.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_binary_reduction_screen.py --force
```

The ignored runtime directory caches each standalone out-of-fold evaluation,
the fixed ensemble summaries, selection result and local-test assessment.
Omit `--force` to reuse compatible candidate caches; keep it for the deliberate,
computationally expensive from-scratch rebuild. Stable fold-safe logic lives in
`src/binary_reduction_evaluation.py`.
