Write-Host "Activating virtual environment..." -ForegroundColor Green
Set-Location ".."

# Check if .venv exists
if (-Not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host ""
    Write-Host "ERROR: Failed to activate virtual environment" -ForegroundColor Red
    Write-Host "Make sure .venv exists in the project root" -ForegroundColor Red
    Write-Host "Run WinSetup.bat first to create the virtual environment" -ForegroundColor Yellow
    return
}

# Activate the virtual environment
& ".\.venv\Scripts\Activate.ps1"

# Verify activation
if ($env:VIRTUAL_ENV) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Virtual Environment Successfully Activated!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Virtual Environment Path: $env:VIRTUAL_ENV" -ForegroundColor Yellow
    Write-Host "Expected Python Location: $env:VIRTUAL_ENV\Scripts\python.exe" -ForegroundColor Yellow
    Write-Host "Checking Python version from venv:" -ForegroundColor Cyan
    python --version
    Write-Host "Verifying Python executable:" -ForegroundColor Cyan
    python -c "import sys; print('Python executable:', sys.executable)"
    Write-Host ""
    Write-Host "To deactivate, run: deactivate" -ForegroundColor Magenta
    Write-Host "========================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "ERROR: Virtual environment activation failed" -ForegroundColor Red
    Write-Host "The activation script ran but VIRTUAL_ENV was not set" -ForegroundColor Red
}
