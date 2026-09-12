param([switch]$Generate)
$ErrorActionPreference = 'Stop'
$selectorRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $selectorRoot
$env:PYTHONPATH = Join-Path $selectorRoot 'src'
function Invoke-Selector {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    python -u -m persona_memory_selector.cli @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Selector failed: $Arguments" }
}
$rScript = Get-ChildItem 'C:/Program Files/R/R-*/bin/Rscript.exe' | Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
if (-not $rScript) { throw 'Rscript is required' }
if (-not (Test-Path 'artifacts/selector/frozen.json')) {
    Invoke-Selector prepare
    Invoke-Selector features
    & $rScript R/selector/train.R
    if ($LASTEXITCODE -ne 0) { throw 'R training failed' }
    python scripts/verify_selector_artifacts.py
    if ($LASTEXITCODE -ne 0) { throw 'Source alignment or R/Python parity failed' }
    Invoke-Selector tune
    Invoke-Selector freeze
}
Invoke-Selector verify
& $rScript R/selector/evaluate_test.R
if ($LASTEXITCODE -ne 0) { throw 'R evaluation failed' }
if ($Generate) {
    Invoke-Selector generate --split val
    Invoke-Selector generate --split test
    Invoke-Selector generate --split test --diagnostics
    Invoke-Selector stress
    Invoke-Selector blind-export --split val
    Invoke-Selector blind-export --split test
    python scripts/verify_selector_study.py
    if ($LASTEXITCODE -ne 0) { throw 'Completed study artifact verification failed' }
    Invoke-Selector review-summary
    foreach ($selectorScript in @('build_blind_review.py', 'build_selector_demo.py', 'selector_results.py')) {
        python (Join-Path 'scripts' $selectorScript)
        if ($LASTEXITCODE -ne 0) { throw "Artifact build failed: $selectorScript" }
    }
}
