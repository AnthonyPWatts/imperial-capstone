# Source-plus-class and identity local comparison

## Decision

Retain the already-prepared original-feature complete-identity candidate ahead
of the weaker source-policy cross. On the previously used 11,880-row local test,
the public-leading source-plus-class 55:45 recipe scores **80.8333%**. Adding
the fixed 20% complete-identity CatBoost raises it to **81.0101%**, a **+0.1768
percentage-point** gain, but the already-prepared identity candidate remains
best at **81.0859%**.

The source-plus-class/identity cross is not promoted or generated. It previously
failed the development gate, and this additional local comparison places it
0.0758 points below the ready candidate. No weights were changed after either
result.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Resolve tomorrow's submission order by comparing the ready identity candidate with the exact 0.8246 public-leading feature recipe. |
| 2. Gather the data | Reuse frozen development, the previously used local test, cached identity probabilities and source-plus-class fold evidence. |
| 3. Explore the data | Use only the named recipes; do not inspect local errors to define another candidate. |
| 4. Clean and preprocess the data | Reuse the source-plus-class fold-fitted feature policy and complete-identity normalisation. |
| 5. Select and engineer features | Make no new features; compare already-fixed original and source-plus-class representations. |
| 6. Define the machine-learning task | Unchanged three-class accuracy, with probability-quality and class-recall diagnostics. |
| 7. Partition the data | Fit source components on 47,520 development rows and score the existing 11,880-row local test once. |
| 8. Select and train candidate methods | Refit source-plus-class XGBoost at its median selected tree count and the unchanged Random Forest; reuse the identically trained identity CatBoost probabilities. |
| 9. Evaluate and interpret the results | Make paired source-vs-cross and ready-identity-vs-cross comparisons without tuning. |
| 10. Deploy and iterate | Keep the ready identity file first; do not generate or upload the weaker cross. |

## Results

| Metric | Source plus class 55:45 | Source plus class + identity | Ready identity candidate |
| --- | ---: | ---: | ---: |
| Accuracy | 80.8333% | 81.0101% | **81.0859%** |
| Balanced accuracy | **66.8534%** | 66.5661% | 66.6233% |
| Macro F1 | 69.1638% | 69.1411% | **69.1655%** |
| Functional recall | 89.1816% | 89.8791% | **89.9411%** |
| Repair recall | **33.3720%** | 32.0973% | 32.0973% |
| Non-functional recall | **78.0066%** | 77.7218% | 77.8313% |
| Log loss | 0.467537 | **0.463037** | 0.463230 |
| Multiclass Brier | 0.269468 | 0.266747 | **0.266671** |

Against source plus class, the cross gains 109 exclusively correct rows and
loses 88, a net 21. Its paired 95% row-bootstrap interval is -0.0505 to +0.4040
points and exact McNemar p is 0.154. Against the ready identity candidate, the
cross gains 38 exclusively correct rows and loses 47, a net loss of nine; its
paired interval is -0.2273 to +0.0758 points.

This is an operational ordering check, not a new independent validation result.
The local test had already confirmed the identity recipe, and the source/identity
cross was known to miss the development promotion threshold. The comparison
therefore cannot rescue that cross merely because it improves the public-leading
source baseline locally.

## Decision at this checkpoint

1. Keep the prepared 44:36:20 original-feature identity candidate ahead of the
   source-plus-class/identity cross.
2. Do not submit the source-plus-class/identity cross on current evidence.
3. Confirm the 81.7929% development-only archive synthesis separately before
   finalising tomorrow's overall order.

The subsequent archive-synthesis confirmation did transfer and now places that
candidate first overall, with this identity candidate second. See
[`archive-synthesis-confirmation.md`](archive-synthesis-confirmation.md).

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_source_identity_confirmation.py
```

Runtime evidence remains ignored under `.runtime/source-identity-confirmation/`.
