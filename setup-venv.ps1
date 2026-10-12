<#
.SYNOPSIS
Creates or refreshes this repository's Python environment.
.EXAMPLE
.\setup-venv.ps1
.EXAMPLE
.\setup-venv.ps1 -Recreate -Python 'C:\Python313\python.exe'
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

function Test-Python3 {
    param([string]$Executable, [string[]]$Prefix = @(), [switch]$RequirePip)
    # A missing launcher registration is an expected failed probe, not a script error.
    $ErrorActionPreference = 'Continue'
    $probe = 'import sys; print(sys.version_info.major)'
    if ($RequirePip) { $probe = 'import sys, pip; print(sys.version_info.major)' }
    $version = & $Executable @Prefix -I -c $probe 2>$null
    return ($LASTEXITCODE -eq 0 -and "$version".Trim() -eq '3')
}

function Find-Python {
    if ($Python) {
        $command = Get-Command -Name $Python -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if (-not $command -or -not (Test-Python3 $command.Source)) {
            throw "The specified interpreter is not a working Python 3 executable: $Python"
        }
        return [pscustomobject]@{ Executable = $command.Source; Prefix = @() }
    }

    foreach ($name in @('py.exe', 'python.exe')) {
        $prefix = @()
        if ($name -eq 'py.exe') { $prefix = @('-3') }
        foreach ($command in @(Get-Command -Name $name -CommandType Application -ErrorAction SilentlyContinue)) {
            if (Test-Python3 $command.Source $prefix) {
                return [pscustomobject]@{ Executable = $command.Source; Prefix = $prefix }
            }
        }
    }
    throw 'No working Python 3 was found. Install Python or pass -Python with its executable path.'
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

$create = $Recreate -or -not $exists
if ($create) {
    $basePython = Find-Python
    if ($basePython.Executable.StartsWith($venvPath + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Use a base Python installation, rather than this .venv, to recreate the environment.'
    }
} elseif (-not (Test-Path -LiteralPath $venvPython -PathType Leaf) -or -not (Test-Python3 $venvPython -RequirePip)) {
    throw 'The existing .venv is incomplete or unusable. Run this command with -Recreate to replace it.'
} elseif ($Python) {
    throw 'To change the interpreter of an existing environment, use -Recreate together with -Python.'
}

$action = 'Create or reuse .venv and install the pinned requirements'
if ($Recreate) { $action = 'Clear and recreate .venv, then install the pinned requirements' }
if (-not $PSCmdlet.ShouldProcess($venvPath, $action)) { return }

if ($create) {
    $arguments = @($basePython.Prefix) + @('-m', 'venv')
    if ($Recreate) { $arguments += '--clear' }
    $arguments += $venvPath
    Write-Host "Creating environment: $venvPath"
    Invoke-Python $basePython.Executable $arguments 'Environment creation'
} else {
    Write-Host "Reusing environment: $venvPath"
}

Invoke-Python $venvPython @('-m', 'pip', '--disable-pip-version-check', 'install', '-r', $requirements) 'Dependency installation'
Invoke-Python $venvPython @('-m', 'pip', 'check') 'Dependency validation'
Write-Host "Ready: $venvPython"
Write-Host "Optional activation: & '$venvPath\Scripts\Activate.ps1'"
