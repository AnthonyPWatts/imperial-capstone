# Capstone Python environment

This is a standalone repository. Both stages share the `.venv` at this
repository's root. A surrounding course checkout, if present, is independent.

- On Windows, use `run-python.ps1` or the explicit local `.venv\Scripts\python.exe`
  for Python, pip, tests and notebook execution. Never fall back to a surrounding
  repository's environment because it happens to be activated.
- Run `setup-venv.ps1` to create or refresh the local environment. The supported
  Windows baseline is 64-bit Python 3.14; setup checks it before installation.
- Select the local environment separately for notebook kernels. Verify
  `sys.executable` when the selected kernel is uncertain.
- Keep dependency changes in this repository's `requirements.txt`; do not
  import a package freeze from the surrounding course repository.
- For environment-helper changes, run `run-python.ps1 -m unittest discover -s tests -v`.
  Follow the [setup guide](README.md#local-setup) for editor and platform details.
