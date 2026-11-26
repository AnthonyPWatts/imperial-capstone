# TabPFN runtime and alternatives

Verified 26 November 2026.

## Local regression

The experiment uses `tabpfn==10.2.0` and explicitly selects TabPFNv2.
The Python package and the model weights have separate version numbers. The
current package defaults to TabPFN-3.5 when the model is unspecified.
[Package release](https://pypi.org/project/tabpfn/10.2.0/) ·
[Prior Labs model list](https://docs.priorlabs.ai/models)

- Run on Windows x64 with CPU PyTorch in the isolated experiment environment.
  The package requires Python 3.10 or later and PyTorch 2.5 or later. The
  official CPU index includes a Windows x64 Python 3.14 wheel for PyTorch
  2.15.0. Prior Labs lists a TabPFNv2 model limit of 10,000 training rows
  and 500 features and gives a lower CPU limit of 1,000 rows for older
  checkpoints. The capstone tables fit within both limits.
  [PyTorch Windows setup](https://pytorch.org/get-started/locally/) ·
  [Official CPU wheel index](https://download.pytorch.org/whl/cpu/torch/) ·
  [Prior Labs local setup](https://github.com/PriorLabs/TabPFN/blob/v10.2.0/README.md#installation)
- The selected `tabpfn-v2-regressor.ckpt` is **44,390,977 bytes**. The
  repository is public and the pinned download is accessible without an
  account. Its SHA-256 is
  `2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736`.
  An explicit local `model_path` keeps model loading separate from inference
  on the coursework data. [Pinned checkpoint](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/02a44705ff7790b50cded404bc59adb94da456a1/tabpfn-v2-regressor.ckpt) ·
  [Pinned model repository](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/tree/02a44705ff7790b50cded404bc59adb94da456a1)
- The 10.2.0 code is Apache 2.0. The V2 weights use the Prior Labs License,
  which adds attribution conditions for distribution and explicitly exempts
  internal benchmarking without external communication from that extra
  condition. The benchmark cites the checkpoint and keeps the weights in
  its local cache. The newer 2.5, 2.6, 3 and 3.5 weight families have
  separate non-commercial terms. Their documented automatic download can
  ask for login and licence acceptance. Prior Labs also documents a
  manual download route.
  [V2 weight licence, section 10](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/blob/02a44705ff7790b50cded404bc59adb94da456a1/LICENSE.txt) ·
  [Model weight access](https://docs.priorlabs.ai/models/accessing-model-weights) ·
  [Manual download instructions](https://github.com/PriorLabs/TabPFN/blob/v10.2.0/README.md#installation--setup)

The regressor's default `predict(X_test)` returns the predictive mean.
For a separate uncertainty check, `predict(X_test, output_type="quantiles",
quantiles=[0.05, 0.5, 0.95])` returns one array per quantile. Assess the
empirical coverage and interval width on held-out returns before relying on
those intervals. A batch acquisition rule requires separate evidence about
the joint predictions across candidate points.
[Prior Labs regression example](https://github.com/PriorLabs/TabPFN/blob/v10.2.0/examples/tabpfn_for_regression.py)

## Hosted service and research alternatives

The hosted `tabpfn-client` is a separate, authenticated service that sends
inputs to Prior Labs for processing. The default usage budgets are **5 million
tokens daily and 20 million monthly**, with account-specific overrides. The
minimum prediction charge is 5,000 tokens for v2.x and 10,000 for v3 or 3.5.
The academic plan offers eligible students and academics twice the regular
monthly allowance after verification. The experiment runs inference on the
local machine.
[Cloud client](https://github.com/PriorLabs/tabpfn-client#stable-release) ·
[Metering](https://docs.priorlabs.ai/api/metering) ·
[Academic plan](https://docs.priorlabs.ai/changelog/november-2026#academic-plan-and-research-compute-grants)

PFNs4BO is a separate research implementation trained on priors for Bayesian
optimisation. Müller and colleagues evaluated it against GP priors and
hyperparameter optimisation benchmarks and studied posterior-derived
acquisition, user priors, and irrelevant dimensions. Its reported results
apply to those models and benchmarks. The original package supports
distribution-based EI, PI and quantile UCB, but its pinned source is
unmaintained and its `scikit-learn<1.2` dependency is incompatible with
this Python 3.14 experiment. Checkpoints also use the older PyTorch
serialisation workflow. PFNs4BO remains a literature comparison here.
[ICML paper](https://proceedings.mlr.press/v202/muller23a.html) ·
[Original code and maintenance note](https://github.com/automl/PFNs4BO/blob/4c0d901e4e0f1d5afd4b12e33974525cf3493dd3/README.md#L107-L121) ·
[Pinned package requirements](https://github.com/automl/PFNs4BO/blob/4c0d901e4e0f1d5afd4b12e33974525cf3493dd3/pyproject.toml#L14-L37)
