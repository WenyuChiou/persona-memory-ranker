param([switch]$Smoke)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
function Invoke-Pmr {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & python -u -m persona_memory_ranker.cli @Arguments
    if ($LASTEXITCODE -ne 0) { throw "pmr failed: $Arguments" }
}
if ($Smoke) { Invoke-Pmr download --persona-limit 4 } else { Invoke-Pmr download }
Invoke-Pmr prepare
Invoke-Pmr features --split train
Invoke-Pmr features --split val
Invoke-Pmr train
Invoke-Pmr evaluate --split cv
Invoke-Pmr evaluate --split val
Invoke-Pmr audit-queue
if (-not $Smoke) {
    Invoke-Pmr freeze
    Invoke-Pmr features --split benchmark
    Invoke-Pmr predict --split benchmark
    Invoke-Pmr evaluate --split benchmark
}
