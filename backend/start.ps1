param(
    [int]$Port = 8001,
    [string]$BindAddress = "127.0.0.1"
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    Write-Error "Project virtual environment not found at '$python'. Create it and install requirements as described in README.md."
    exit 1
}

Push-Location $projectRoot
try {
    & $python -m uvicorn backend.app.main:app --app-dir $projectRoot --reload --host $BindAddress --port $Port
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

exit $exitCode
