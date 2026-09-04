# TabM-dagger bounded screen

## Decision

Reject TabM-dagger as a final-push component and stop after the two permitted
frozen development folds. The closest fixed blend, 2% TabM, lost two correct
rows in aggregate with fold nets of -3 and +1. The 5%, 10% and 15% blends lost
7, 18 and 17 rows respectively. None passed the predeclared requirement of a
positive aggregate result without a fold loss.

The local holdout remained unopened. No complete-development fit, competition
prediction or submission file was produced.

## Literature basis and implementation

[TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html)
was published at ICLR 2025. It trains several MLP-like predictors in parallel
while sharing most parameters, then averages their probabilities at inference.
The paper reports that this parameter-efficient ensemble improves optimisation
and overfitting behaviour relative to a plain tabular MLP.

The experiment used version 0.0.3 of the authors'
[official `yandex-research/tabm` package](https://github.com/yandex-research/tabm)
at repository commit `28e47ae301c92ec37787dde1ce923a0793f405b4`. It used
only the supplied competition data: no pretrained weights or external features
participated. The package and `rtdl_num_embeddings` 0.0.12 were installed in
the ignored repo-local `.runtime/realmlp-venv`; the shared project environment
and dependency manifest were unchanged.

## Frozen evaluation contract

The single specification was fixed before fitting:

- official TabM defaults: 32 submodels, two 512-wide blocks, 0.1 dropout;
- piecewise-linear numeric embeddings with 48 bins and dimension 16, the
  package's recommended stronger `TabM` variant;
- official AdamW defaults of learning rate 0.002 and weight decay 0.0003;
- independently evaluated cross-entropy for each submodel and probability-space
  averaging at inference, as required by the official training example;
- the 14 accepted numeric fields and 15 accepted categorical fields;
- the top 50 fold-fitted values from each of the six normalised identity fields,
  with all rare and unseen values sharing an explicit zero category;
- quantile-normalised numerics, fitted within the current inner-training split;
- a stratified 20% inner validation split, accuracy-based early stopping with
  patience 16, and at most 256 epochs;
- no oversampling or class weighting;
- fixed TabM contributions of 2%, 5%, 10% and 15% to the 0.8298 incumbent.

All category dictionaries, numeric medians, quantile transforms and numeric-bin
boundaries were learned from inner-training rows only. Outer-fold rows were used
only once for the reported evaluation.

## Results

The first two frozen folds contain 19,008 rows. Their incumbent accuracy was
81.8761%. Each TabM fit had 856,032 parameters and a categorical one-hot width
of 606. Fits took 20.3 and 26.1 seconds on the RTX 4070 SUPER and selected
epochs 22 and 34.

| Candidate | Accuracy | Net rows | Fold nets | Repair recall | Repair-recall change | F→repair net |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Incumbent | **81.8761%** | — | — | **33.5264%** | — | — |
| TabM standalone | 79.9716% | -362 | -205, -157 | 26.2853% | -7.2411 pp | -20 / 90 |
| 2% TabM blend | **81.8655%** | **-2** | -3, +1 | 33.1644% | -0.3621 pp | 0 / 0 |
| 5% TabM blend | 81.8392% | -7 | -2, -5 | 33.0920% | -0.4345 pp | 0 / 2 |
| 10% TabM blend | 81.7814% | -18 | -4, -14 | 32.5851% | -0.9413 pp | 0 / 4 |
| 15% TabM blend | 81.7866% | -17 | -8, -9 | 32.1506% | -1.3758 pp | 0 / 4 |

`F→repair net` reports the change in correct rows followed by the number of
incumbent `functional` predictions changed to `functional needs repair`.

The standalone model was materially stronger than the earlier conventional MLP
screen, but still 1.9045 percentage points behind the current ensemble. Its
error diversity was directionally wrong for the requested repair recovery: 90
functional-to-repair transitions lost 20 net rows, while 226 repair-to-functional
transitions lost another 38. Even a 2% contribution reduced repair recall; no
fixed blend found a profitable functional-to-repair boundary.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test one modern, non-pretrained neural family for complementary errors and repair recovery. |
| 2. Gather the data | Reuse only the supplied labelled predictors and frozen incumbent probabilities. |
| 3. Explore the data | Confirm 14 numeric fields and compact the six high-cardinality identities to bounded top-50 dictionaries. |
| 4. Clean and preprocess the data | Fit medians, quantile transforms and unknown categories inside each inner-training split. |
| 5. Select and engineer features | Retain accepted features plus the already-audited compact identity representation. |
| 6. Define the machine-learning task | Preserve three-class probabilistic classification scored by accuracy. |
| 7. Partition the data | Reuse the first two frozen outer folds and a seeded inner validation split. |
| 8. Select and train candidate methods | Fit one official-default TabM-dagger specification with no hyperparameter search. |
| 9. Evaluate and interpret the results | Compare standalone and four fixed blends, including repair recall and transition nets. |
| 10. Deploy and iterate | Stop before the local holdout and competition because every blend is negative in aggregate. |

## Reproduction and verification

Run from the project root with the isolated environment after installing
`tabm==0.0.3`:

```powershell
.\.runtime\realmlp-venv\Scripts\python.exe stage-1-pump-it-up\scripts\run_tabm_screen.py --max-folds 2
```

The runner checkpoints each fold under ignored `.runtime/tabm-screen/` and
replayed both checkpoints to reproduce the aggregate result exactly. Focused
tests cover deterministic top-category selection, the rare/unseen bucket and
finite fold-fitted preprocessing.
