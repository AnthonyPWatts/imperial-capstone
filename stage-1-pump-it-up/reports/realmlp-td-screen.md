# RealMLP-TD bounded screen

## Decision

Reject RealMLP-TD as a final-ensemble component and stop after two frozen
development folds. Every predeclared 5%, 10%, 15% and 20% probability blend
lost rows to the incumbent in both folds. With two fold losses already fixed,
none could reach the existing three-of-five fold-win gate.

The local holdout remained unopened for this model. No full-data refit,
competition prediction or submission file was produced.

## Why this candidate was tested

RealMLP-TD is a materially different neural tabular method from the earlier
scaled one-hot `MLPClassifier` screen. The official `pytabkit` implementation
uses learned categorical embeddings, numerical embeddings, robust scaling,
smooth clipping and tuned optimisation defaults. It therefore offered a
bounded test of whether modern tabular deep learning supplied useful error
diversity beside the tree ensemble.

The experiment used `pytabkit` 1.7.3 and PyTorch 2.14.0 with CUDA 13.0 in the
ignored repo-local `.runtime/realmlp-venv` environment. The shared project
environment and dependency manifest were unchanged.

## Evaluation contract

The screen fixed the following choices before training:

- one seed, `20260904`;
- RealMLP-TD tuned defaults, including 256 epochs and one internal 20%
  validation split selected by classification error;
- no oversampling and no class weights;
- the accepted 29 engineered features plus the six already-audited normalised
  identity fields: `funder`, `installer`, `wpt_name`, `subvillage`, `ward` and
  `scheme_name`;
- native categorical treatment for the 15 accepted categoricals and six
  identities;
- numeric missing values filled with medians learned only from the current
  outer-training fold;
- fixed RealMLP contributions of 5%, 10%, 15% and 20% to the 0.8298 incumbent
  probabilities.

Each completed outer fold was checkpointed before the next fit. The first two
fits took 264.9 and 257.7 seconds on the RTX 4070 SUPER. Their 19,008 validation
rows retained the original frozen fold assignments and ID order.

## Results and futility stop

The incumbent scored 81.876% over the first two folds. RealMLP standalone
scored 76.320%, a loss of 1,056 correct rows. The fixed blends were also
uniformly negative:

| Candidate | Accuracy | Change | Fold 1 net rows | Fold 2 net rows | Repair recall | Repair-recall change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RealMLP standalone | 76.320% | -5.556 pp | -582 | -474 | 14.555% | -18.972 pp |
| 5% RealMLP | 81.766% | -0.110 pp | -11 | -10 | 32.440% | -1.086 pp |
| 10% RealMLP | 81.750% | -0.126 pp | -21 | -3 | 32.151% | -1.376 pp |
| 15% RealMLP | 81.666% | -0.210 pp | -31 | -9 | 31.209% | -2.317 pp |
| 20% RealMLP | 81.550% | -0.326 pp | -46 | -16 | 30.123% | -3.403 pp |

The 5% blend, the least harmful option, changed 114 predictions and lost 21
correct rows. It moved only five incumbent `functional` predictions to
`functional needs repair`; all five changes were net errors. In the opposite
direction it moved 25 repair predictions to `functional` and lost one net
correct row. Increasing the RealMLP contribution reduced repair recall
monotonically. This is the opposite of the required minority-class behaviour.

The standalone model showed the same structural problem more strongly. It
moved 113 `functional` predictions to repair for a net loss of 23, while moving
418 repair predictions to `functional` for a net loss of 113. Learned identity
embeddings did not recover a useful repair boundary under this fixed setup.

## Interpretation

The result is decisive for this candidate rather than for all neural tabular
models. A broader RealMLP hyperparameter search or neural ensemble might behave
differently, but it would be a new and substantially more expensive experiment.
It is not justified when all four low-weight blends lose on both independent
frozen folds and repair recall deteriorates immediately.

Runtime probabilities, transition counts and machine-readable results remain
ignored under `.runtime/realmlp-screen/`. The reproducible runner is
[`scripts/run_realmlp_screen.py`](../scripts/run_realmlp_screen.py), and its
fold-safe feature utilities are in
[`src/realmlp_evaluation.py`](../src/realmlp_evaluation.py).

## Verification

The focused unit tests covered training-fold imputation, identity
normalisation, fixed blend arithmetic and class-probability ordering:

```powershell
.\.venv\Scripts\python.exe -m unittest stage-1-pump-it-up\tests\test_realmlp_evaluation.py
```

All four tests passed. The screen was replayed from both saved fold checkpoints
to reproduce the futility decision and aggregate metrics exactly.
