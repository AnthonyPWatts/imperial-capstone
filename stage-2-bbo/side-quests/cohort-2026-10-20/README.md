# Pooled cohort side quest

Open [the results](index.html) for all sixteen executed notebook copies, with
code hidden and interactive figures preserved. The corresponding `.ipynb` files
sit beside each reading view in `notebooks/Function_N/Week_04/`.

This snapshot, collected on **20 October 2026**, contains **242 usable
observations**: 175 shared starting observations, 24 confirmed own returns and
43 additional peer records. Functions 1–8 have 19, 18, 23, 39, 28, 28, 38 and
49 observations respectively. The original Week 4 notebooks, strategies and
submitted inputs remain unchanged.

All eight observed-data notebooks begin with a paraphrased course description,
the relevant cohort evidence and possible exploration clues. Their prediction
companions include the description, model-specific interpretations and a link
back to the evidence. The notes distinguish stated properties from hypotheses,
retain the scoring qualifications, and cite Mini-lesson 12.8. They describe the
fixed 20 October snapshot; they do not change the calculations or query policies.

Function 1's cohort exploration also compares absolute magnitude with the signed
outputs: a spatial map shares one logarithmic magnitude scale across both signs,
and an interactive terrain folds negative readings upwards while retaining signed
hover values. Its model notebook separately refits the five existing methods to
`g(y) = log10(1 + |y| / 1e-200)`, alongside the original signed target. Shared-scale
maps, matched cross-sections and a common query pool show the change in shape and
exploration proposals. Errors are compared with each target's own mean baseline.
This tests an activity hypothesis; it does not change the signed maximisation score.

Four public peer repositories supplied returned evaluations: the histories in
[kosina/bbocapstone](https://github.com/kosina/bbocapstone) (16 additions),
[tomxandrews/bbocapstone-s1](https://github.com/tomxandrews/bbocapstone-s1) (10),
[arwelbrown/black-box-capstone](https://github.com/arwelbrown/black-box-capstone) (9),
and [mkamar17/bbocapstone](https://github.com/mkamar17/bbocapstone) (8).
The full Canvas-derived cohort roster and repository-access audit are retained
in the private Imperial ML/AI repository under `Reference/Capstone/`.

## Data treatment and limitations

Each accepted peer record has a reported input and returned output, with a
commit-pinned URL, source location and checksum in the local `dataset-audit.json`.
No peer code is executed. Literal Python assignments, notebook data, CSV rows
and literal/cached cells from the Excel tracker are read as data.

The initial arrays in three peer repositories matched the authorised starting
data exactly: 24 function-level checks. They are included once. Seven later
records matched an existing input at portal precision and an output consistent
with its reported rounding precision. Distinct reported outputs at the same
input would remain separate observations; zero and tiny non-zero values remain distinct.

Twenty-four pending queries and six records with inputs reported as exactly
`1.000000` are retained in the audit and excluded from fitting. No boundary
coordinate was guessed or clipped. Reported equal outputs at different inputs
are retained. Peer records are self-reported and were not verified against
the course portal; they may contain rounding or transcription errors.

The pooled progress plots show the highest available return in each reported
round. They are not evidence that one participant's method is superior.
The coverage-plan storyboards treat peer inputs as a fixed background for a
geometric comparison. They do not reconstruct historical information availability.
Model validation and hypothetical suggestions use the collected snapshot;
they are not forecasts made before the original returns. Additional observations
do not increase the course query allowance.

## Course lifecycle

1. **Define the goal and scope** — compare the existing Week 4 methods with more published observations, as a separate learning study.
2. **Gather the data** — inspect the cohort's posted repositories and retain paired inputs and outputs with their sources.
3. **Explore the data** — rerun observed-output, sampling-gap, input-profile and progress views.
4. **Clean and preprocess the data** — validate dimensions and bounds, preserve precision, identify duplicates and exclude missing outputs.
5. **Select and engineer features** — retain the original input settings; Function 1 also compares a log-magnitude target, with no new input features.
6. **Define the machine-learning task** — regression of each unknown function, supporting hypothetical maximisation queries; the Function 1 magnitude comparison explores activity instead of signed score.
7. **Partition the data** — retain leave-one-out validation; no separate final test set is available for this small exploratory study.
8. **Select and train candidate methods** — rerun the copied comparisons, kernels, candidate searches and acquisition settings.
9. **Evaluate and interpret the results** — compare with the mean baseline, inspect sensitivity and distinguish published incumbents from predicted outputs.
10. **Deploy and iterate** — deployment is not applicable; this side quest does not submit queries or change the official strategy.

## Rebuilding locally

Use the capstone repository's `.venv`. Restore the authorised original Week 4
`observations.csv` files as described in the main data guide. From the capstone
root, run:

```powershell
.venv\Scripts\python.exe stage-2-bbo/side-quests/cohort-2026-10-20/download_sources.py
.venv\Scripts\python.exe stage-2-bbo/side-quests/cohort-2026-10-20/build_dataset.py
.venv\Scripts\python.exe stage-2-bbo/side-quests/cohort-2026-10-20/copy_notebooks.py
.venv\Scripts\python.exe stage-2-bbo/side-quests/cohort-2026-10-20/run_notebooks.py
.venv\Scripts\python.exe stage-2-bbo/side-quests/cohort-2026-10-20/export_reading_views.py
```

Copying the notebooks recreates the study sources and clears their saved outputs;
run that step only when rebuilding intentionally. `run-results.json` records the
saved executions. Downloaded sources live under `.runtime/cohort-side-quest/`;
observation CSVs and the detailed audit are ignored by Git.

To refresh only the description-and-clue notes in existing notebooks, run
`.venv\Scripts\python.exe stage-2-bbo/side-quests/cohort-2026-10-20/function_briefs.py`,
then export the reading views. This retains saved code outputs and requires no
model refits. `copy_notebooks.py` also adds these sections during a full rebuild.
