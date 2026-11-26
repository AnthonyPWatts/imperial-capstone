# TabPFN surrogate benchmark

The experiment compares TabPFN-2 with the owner-data Gaussian process on all
eight Imperial black-box optimisation functions. The working capstone
checkout contains the official starters and feedback through Round 9. Its
prepared Week 10 choices and the candidate lists in this experiment have
separate roles. The benchmark reads the prepared inputs as exclusions.

[Measured results and recommendations](results.md) ·
[Runtime, licences and other PFNs](runtime-notes.md)

## Reproduce locally

From the capstone root, use the existing Capstone `.venv` and official
accumulated observations. The optional runtime sits under `.runtime` and
uses the same pinned scientific libraries through a local `.pth` file.
The TabPFN package and CPU PyTorch live in that separate runtime.

~~~powershell
$benchmark = New-Item -ItemType Directory -Path '.runtime/tabpfn-surrogate' -Force
.\.venv\Scripts\python.exe -m venv (Join-Path $benchmark.FullName 'venv')
$python = Join-Path $benchmark.FullName 'venv/Scripts/python.exe'
$basePackages = (Resolve-Path '.venv/Lib/site-packages').Path
Set-Content -LiteralPath (Join-Path $benchmark.FullName 'venv/Lib/site-packages/capstone-base.pth') -Value $basePackages -Encoding utf8NoBOM
& $python -m pip install --no-deps 'torch==2.15.0+cpu' --index-url https://download.pytorch.org/whl/cpu
& $python -m pip install 'tabpfn==10.2.0'
& $python -m pip check

$checkpoint = Join-Path $benchmark.FullName 'tabpfn-v2-regressor.ckpt'
if (-not (Test-Path -LiteralPath $checkpoint)) {
    Invoke-WebRequest -Uri 'https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/02a44705ff7790b50cded404bc59adb94da456a1/tabpfn-v2-regressor.ckpt' -OutFile $checkpoint
}
$expected = '2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'
if ((Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
    throw 'The checkpoint hash differs from the pinned public model.'
}

# F1 first, with separate output files for the trial.
& $python stage-2-bbo/side-quests/tabpfn-surrogate/run_benchmark.py --functions 1 --out-dir .runtime/tabpfn-surrogate/f1-trial

# All eight functions.
& $python stage-2-bbo/side-quests/tabpfn-surrogate/run_benchmark.py
~~~

The runner checks the weight hash, loads an explicit local model path and
blocks Python socket access during inference. Downloading the public
packages and checkpoint requires network access before the benchmark.
The inference output contains observation-level details, so it is stored
in the ignored `.runtime/tabpfn-surrogate/results/` directory by default.
`summary.json` identifies the functions in that invocation, versions and
settings. Per-function JSON, CSV and PNG files give the detailed readback.

## Ten-step course workflow

1. **Define the goal and scope** — assess whether TabPFN adds predictive or
   shortlist evidence under the existing weekly query budget.
2. **Gather the data** — use the official 175 starters and 72 returned
   observations. Check each return against its round record.
3. **Explore the data** — use raw score units within each function and
   compare fitted landscapes and measured support. F1 uses the existing
   201×201 coverage raster plus a zoom around the narrow measured peak.
   Higher-dimensional pairs hold other coordinates at the incumbent.
4. **Clean and preprocess the data** — keep the portal coordinates and raw
   targets. Fit both models' transformations from the training part of
   each fold. Group equal six-decimal coordinates for retrospective holdout.
5. **Select and engineer features** — use the same observed coordinates for
   both methods. The GP can fit the existing official positive-source mean
   on F1 from training values. F8's separate starter quadratic remains
   part of the existing decision evidence.
6. **Define the machine-learning task** — regress returned objective scores
   and examine whether high-value rows rank near the top.
7. **Partition the data** — hold out each observed coordinate in turn. In
   chronological replay, predict round *t* using the starters and rounds
   strictly before *t*.
8. **Select and train candidate methods** — use the owner Matérn GP on raw
   output, with its existing function-specific starting scales, and four
   TabPFN-2 estimators with seed 907. The GP's predictive observation
   variance includes estimated noise. TabPFN's 5th, 50th and 95th predictive
   percentiles come from its regression distribution.
9. **Evaluate and interpret the results** — report RMSE, MAE, Spearman
   correlation and recall among the highest observed quartile
   (top ⌈n/4⌉, ties broken by row order). Check 90% interval coverage and
   width. The saved kernel and model settings reflect current knowledge.
   Chronological replay restricts the fitted data but uses the originally
   selected historical inputs. It measures prospective-style prediction
   at those points. An optimisation policy would need a separate test of
   the inputs each model would have chosen.
10. **Deploy and iterate** — experimental shortlist evidence only. Re-evaluate
    on newly received official data and retain the existing portal workflow.

## Candidate comparison

Both models score the same seed-fixed global and near-incumbent Sobol points
and coordinate arms. We use a batched pool because the methods rank thousands
of common locations efficiently. This keeps the comparison on a single
search design without repeated differential evolution calls. All proposals
are clipped to `[0, 0.999999]`, rounded to six decimals, deduplicated and
screened against official, already-submitted and prepared coordinates. F5
retains the existing 0.020 separation rule.

The CSVs keep GP expected improvement in the next returned score, TabPFN
mean, Euclidean distance to the nearest official observation and absolute
mean disagreement as separate columns. Ranks and percentiles are per
function. The additional 60% GP-EI, 30% TabPFN-mean and 10% coverage rank
blend is a diagnostic. Model disagreement is an uncalibrated difference in
means. Each CSV is ordered by GP expected improvement.

Local runtime notes cover the hosted budgets and why the original PFNs4BO
package remains a separate follow-up. The benchmark uses local inference
and consumes no live submissions.
