<#
.SYNOPSIS
Creates or refreshes the standalone Capstone Python 3.14 environment on Windows.
.EXAMPLE
.\setup-venv.ps1
.EXAMPLE
.\setup-venv.ps1 -Recreate -Python 'C:\Python314\python.exe'
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$Python,
    [switch]$Recreate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
# Check native exit codes explicitly on both Windows PowerShell and PowerShell 7.
$PSNativeCommandUseErrorActionPreference = $false

$repoRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$venvPath = [IO.Path]::GetFullPath((Join-Path $repoRoot '.venv'))
$venvPython = Join-Path $venvPath 'Scripts\python.exe'
$requirements = Join-Path $repoRoot 'requirements.txt'

function Test-ProjectPython {
    param([string]$Executable, [string[]]$Prefix = @(), [switch]$RequireBase, [switch]$RequireVenv)
    # A missing launcher registration is an expected failed probe, not a script error.
    $ErrorActionPreference = 'Continue'
    $probe = 'import struct, sys; assert sys.version_info[:2] == (3, 14) and struct.calcsize(''P'') == 8'
    if ($RequireBase) { $probe += '; assert sys.prefix == sys.base_prefix' }
    if ($RequireVenv) {
        $probe += '; import pathlib, pip; assert sys.prefix != sys.base_prefix; assert pathlib.Path(sys.prefix).resolve() == pathlib.Path(sys.argv[1]).resolve()'
        $probe += '; assert ''include-system-site-packages = false'' in pathlib.Path(sys.prefix, ''pyvenv.cfg'').read_text().lower()'
    }
    & $Executable @Prefix -I -c $probe $venvPath 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Find-Python {
    if ($Python) {
        $command = Get-Command -Name $Python -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if (-not $command -or -not (Test-ProjectPython $command.Source -RequireBase)) {
            throw "Use a base 64-bit Python 3.14 installation, not an activated virtual environment: $Python"
        }
        return [pscustomobject]@{ Executable = $command.Source; Prefix = @() }
    }

    foreach ($name in @('py.exe', 'python.exe')) {
        $prefix = @()
        if ($name -eq 'py.exe') { $prefix = @('-3.14') }
        foreach ($command in @(Get-Command -Name $name -CommandType Application -ErrorAction SilentlyContinue)) {
            if (Test-ProjectPython $command.Source $prefix -RequireBase) {
                return [pscustomobject]@{ Executable = $command.Source; Prefix = $prefix }
            }
        }
    }
    throw 'No base 64-bit Python 3.14 was found. Install it or pass -Python with its executable path. Do not pass another project''s .venv.'
}

function Invoke-Python {
    param([string]$Executable, [string[]]$PythonArguments, [string]$Action)
    & $Executable @PythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Action failed (exit code $LASTEXITCODE). Check the output above; the environment has been left in place."
    }
}

if (-not (Test-Path -LiteralPath $requirements -PathType Leaf)) {
    throw "Requirements file not found: $requirements"
}
if ((Split-Path -Parent $venvPath) -ne $repoRoot) {
    throw "Refusing to modify an environment outside the repository: $venvPath"
}
$exists = Test-Path -LiteralPath $venvPath
if ($exists) {
    $item = Get-Item -LiteralPath $venvPath -Force
    if (-not $item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "The .venv path must be an ordinary directory, not a file or junction: $venvPath"
    }
}
if ($Recreate -and $env:VIRTUAL_ENV -and
    [IO.Path]::GetFullPath($env:VIRTUAL_ENV).TrimEnd('\') -eq $venvPath.TrimEnd('\')) {
    throw 'Deactivate this environment and close its notebook kernels before using -Recreate.'
}
if ($env:VIRTUAL_ENV -and
    [IO.Path]::GetFullPath($env:VIRTUAL_ENV).TrimEnd('\') -ne $venvPath.TrimEnd('\')) {
    Write-Warning "Another environment is active: $env:VIRTUAL_ENV. Setup targets only $venvPath; your shell remains unchanged. Use run-python.ps1 for Capstone commands."
}

$create = $Recreate -or -not $exists
if ($create) {
    $basePython = Find-Python
    if ($basePython.Executable.StartsWith($venvPath + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Use a base Python installation, rather than this .venv, to recreate the environment.'
    }
} elseif (-not (Test-Path -LiteralPath $venvPython -PathType Leaf) -or -not (Test-ProjectPython $venvPython -RequireVenv)) {
    throw 'The existing .venv must be an isolated, working 64-bit Python 3.14 environment with pip. Close its kernels and deactivate it, then use -Recreate to replace it.'
} elseif ($Python) {
    throw 'To change the interpreter of an existing environment, use -Recreate together with -Python.'
}

$action = 'Create or reuse .venv and install the pinned requirements'
if ($Recreate) { $action = 'Clear and recreate .venv, then install the pinned requirements' }
if (-not $PSCmdlet.ShouldProcess($venvPath, $action)) { return }

if ($create) {
    $arguments = @($basePython.Prefix) + @('-I', '-m', 'venv', '--prompt', 'imperial-capstone')
    if ($Recreate) { $arguments += '--clear' }
    $arguments += $venvPath
    Write-Host "Creating environment: $venvPath"
    Invoke-Python $basePython.Executable $arguments 'Environment creation'
} else {
    Write-Host "Reusing environment: $venvPath"
}

if (-not (Test-ProjectPython $venvPython -RequireVenv)) {
    throw 'The created environment failed validation. Check the Python installation before retrying.'
}
Invoke-Python $venvPython @('-I', '-m', 'pip', '--disable-pip-version-check', 'install', '-r', $requirements) 'Dependency installation'
Invoke-Python $venvPython @('-I', '-m', 'pip', 'check') 'Dependency validation'
Write-Host "Ready: $venvPython"
Write-Host "Run Capstone Python: & '$repoRoot\run-python.ps1' <Python arguments>"
Write-Host "Optional activation: & '$venvPath\Scripts\Activate.ps1'"
