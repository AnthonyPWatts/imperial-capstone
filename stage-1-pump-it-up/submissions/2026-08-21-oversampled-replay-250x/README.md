# 2.5× repair-class replay

These three candidates replay the exact 21 August recipes with the final repair-class training count fixed at 2.5 times its natural count. The full labelled refit contains 65,876 rows: 32,259 functional, 10,793 repair-needed and 22,824 non-functional.

Competition repair hard-label shares are 6.91%, 6.98% and 7.10%, close to the independently inferred 7.19% competition prevalence. On the labelled local test, the XGBoost blends lose 0.40–0.43 percentage points of accuracy while gaining 10.9–11.9 points of repair recall.

Probability diagnostics show that the raw repair probabilities are too optimistic even though the winning-label balance is close. This variant is therefore an experimental hard-classification candidate, not a calibrated repair-risk estimator. No candidate in this directory was uploaded here, and 2.5× is the final multiplier examined in this session.
