# 80%-replica minority-oversampling replay

These three candidate CSVs retain 80% of the replicas added by the full oversampling policy. This means 19,123 repair-needed rows in the 74,206-row full labelled refit; it does not mean multiplying the final minority target by 0.8.

Relative to full oversampling, the follow-up recovers a small amount of local-test accuracy while preserving most of the repair-recall gain. Competition repair predictions are 9.14–9.35%, compared with 9.69–10.26% under full oversampling and 3.75–3.95% in the original submissions.

This level was proposed after observing that the full policy overshot the natural prevalence. Its local-test result is therefore adaptive exploratory evidence, not an independent confirmation. The CSVs are valid candidates for a deliberately experimental later upload, but none was uploaded here.
