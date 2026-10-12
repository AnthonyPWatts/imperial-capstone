<#
.SYNOPSIS
Runs Python from this checkout's .venv, regardless of the active environment.
.DESCRIPTION
Pass arguments exactly as for python. Keeps the caller's working directory and
returns Python's exit code. Run setup-venv.ps1 first. Invoke, do not dot-source.
.EXAMPLE
.\run-python.ps1 -m pip check
.EXAMPLE
.\run-python.ps1 -m unittest discover -s stage-2-bbo/tests
#>
# No parameter block: forward Python flags such as -m, -c and --help unchanged.
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    throw "Capstone's Python was not found: $venvPython. Run '$PSScriptRoot\setup-venv.ps1' first."
}

& $venvPython @args
exit $LASTEXITCODE
