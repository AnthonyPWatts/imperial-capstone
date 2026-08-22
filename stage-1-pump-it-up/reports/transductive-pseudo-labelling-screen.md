# Transductive pseudo-labelling screen

## Decision

Stop transductive pseudo-labelling. Replacing the accepted XGBoost component
with a student trained on outer-training rows plus competition rows labelled at
at least 98% teacher confidence lowers the promoted vote from 81.7403% to
**81.7298%**. It wins **3/5** folds but loses **0.0105 percentage points** in the
mean and as much as 0.1684 points on one fold.

The pseudo-labelled XGBoost itself falls from 81.0143% to **80.9259%**. Do not
tune the confidence threshold or pseudo-row weight. No local-test or new
competition prediction was generated.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test whether unlabelled competition covariates can improve a frozen-fold XGBoost through high-confidence self-training. |
| 2. Gather the data | Reuse each outer-training partition and the 14,850 unlabelled competition rows. |
| 3. Explore the data | Audit full-data teacher confidence without using competition outcomes; fix 98% before outer-fold fitting. |
| 4. Clean and preprocess the data | Reuse the accepted fold-fitted one-hot policy independently for teacher and student. |
| 5. Select and engineer features | Add no predictors; append only competition rows crossing the fixed confidence threshold. |
| 6. Define the machine-learning task | Semi-supervised transductive three-class classification, still scored on original crisp validation labels. |
| 7. Partition the data | Exclude every outer-validation row from teacher fitting, pseudo-labelling and student fitting. |
| 8. Select and train candidate methods | Fit one teacher and one student XGBoost per fold using the accepted fold-specific tree count. |
| 9. Evaluate and interpret the results | Compare the student component and unchanged 44:36:20 recipe with the promoted identity vote. |
| 10. Deploy and iterate | Reject self-training without threshold or weight tuning. |

## Fixed policy and audit

The teacher is the accepted depth-8, child-weight-1 XGBoost fitted only on the
current outer-training rows. It predicts the separate competition rows; labels
whose maximum probability is at least 0.98 are appended once, unweighted, and a
student is refitted with the same fold-specific tree count. The untouched outer
validation fold is then scored.

| Fold | Trees | Pseudo rows | Functional | Repair | Non-functional | Mean confidence |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,111 | 2,740 | 1,031 | 0 | 1,709 | 99.160% |
| 2 | 1,044 | 2,631 | 984 | 0 | 1,647 | 99.187% |
| 3 | 1,014 | 2,658 | 980 | 0 | 1,678 | 99.196% |
| 4 | 904 | 2,431 | 875 | 0 | 1,556 | 99.185% |
| 5 | 921 | 2,394 | 803 | 0 | 1,591 | 99.187% |

At this threshold every pseudo-row is functional or non-functional. The policy
therefore supplies no new repair examples and mostly reinforces predictions
the teacher already makes confidently.

## Results

| Candidate | Accuracy | Leader change | Fold wins | Worst fold | Repair-recall change | Passes gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Promoted identity vote | **81.7403%** | — | — | — | — | incumbent |
| Pseudo-label identity vote | 81.7298% | -0.0105 pp | 3/5 | -0.1684 pp | +0.0578 pp | No |
| Accepted XGBoost | 81.0143% | -0.7260 pp | 0/5 | -0.8733 pp | -2.4034 pp | No |
| Pseudo-labelled XGBoost | 80.9259% | -0.8144 pp | 0/5 | -0.9996 pp | -2.4320 pp | No |

The split result—three small fold wins but a lower mean—does not support a
second threshold. Lowering confidence would add more uncertain labels, while a
higher threshold would further reduce an already redundant intervention.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_transductive_pseudo_labelling_screen.py
```

Runtime evidence remains ignored under
`.runtime/transductive-pseudo-labelling-screen/`.
