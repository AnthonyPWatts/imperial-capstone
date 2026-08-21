# Full minority-oversampling replay

These three candidate CSVs replay the exact 21 August submission recipes after randomly replicating `functional needs repair` to the natural `non functional` count. The full labelled refit contains 77,907 rows: 32,259 functional, 22,824 repair-needed and 22,824 non-functional.

This is a controlled distribution intervention. Predictor policy, estimators, recipe weights, seeds and fixed XGBoost tree counts remain unchanged. The corresponding labelled local-test comparison loses 1.4–1.8 percentage points of nominal accuracy and gains 20.0–22.4 points of repair recall. Competition repair predictions rise from 3.75–3.95% to 9.69–10.26%.

The CSVs are generated and validated by `scripts/run_oversampled_submission_replay.py`. Detailed recorded results are in `notebooks/09-oversampled-submission-replay.ipynb`. No CSV in this directory was uploaded as part of this experiment.
