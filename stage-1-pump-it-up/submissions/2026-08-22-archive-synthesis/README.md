# Archive-derived synthesis candidate

This directory records the fixed six-component synthesis prepared on 22 August
2026 and submitted first on 23 August 2026.

The candidate averages two previously fixed representation votes and therefore
uses 33% accepted XGBoost, 11% spatial-height XGBoost, 18% accepted Random
Forest, 9% categorical-frequency Random Forest, 9% spatial-height Random Forest
and 20% complete-identity CatBoost. No weight was selected on local-test or
competition predictions.

It reached 81.7929% across the five frozen development folds and 81.1869% on
the previously used local test. The companion complete-identity candidate
reached 81.0859% locally. The synthesis scored **0.8288** publicly, a new
project best: 0.0042 above the previous 0.8246 leader and 0.0028 above the
0.826 project target. The complete-identity companion scored 0.8259.

The ignored CSV filename and SHA-256 are recorded in [`manifest.json`](manifest.json).
Reproduce it from `stage-1-pump-it-up` with:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_archive_synthesis_confirmation.py
..\.venv\Scripts\python.exe .\scripts\generate_archive_synthesis_submission.py
```
