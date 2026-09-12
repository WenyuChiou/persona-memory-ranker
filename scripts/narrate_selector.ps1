param([Parameter(Mandatory=$true)][ValidateSet('M1','M2')][string]$Milestone)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$contentPath = Join-Path $projectRoot "artifacts\selector\presentations\$Milestone-content.json"
$content = Get-Content -LiteralPath $contentPath -Raw -Encoding utf8 | ConvertFrom-Json
$audioDirectory = Join-Path $projectRoot "artifacts\selector\presentations\$Milestone\audio"
New-Item -ItemType Directory -Path $audioDirectory -Force | Out-Null
Add-Type -AssemblyName System.Speech
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $speaker.SelectVoice('Microsoft David Desktop')
    $speaker.Rate = -3
    for ($i = 0; $i -lt $content.slides.Count; $i++) {
        $destination = Join-Path $audioDirectory ('slide-{0:D2}.wav' -f ($i+1))
        $speaker.SetOutputToWaveFile($destination)
        $speaker.Speak([string]$content.slides[$i].notes)
        $speaker.SetOutputToNull()
        Write-Output ('Narrated {0} slide {1}/{2}' -f $Milestone,($i+1),$content.slides.Count)
    }
} finally {
    $speaker.Dispose()
}
