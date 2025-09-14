@echo off
echo Activating virtual environment...
cd ..
call .\.venv\Scripts\activate.bat

REM Check if activation was successful
if defined VIRTUAL_ENV (
    echo.
    echo ========================================
    echo Virtual Environment Successfully Activated!
    echo ========================================
    echo Virtual Environment Path: %VIRTUAL_ENV%
    echo Expected Python Location: %VIRTUAL_ENV%\Scripts\python.exe
    echo Checking Python version from venv:
    python --version
    echo Verifying pip is from venv:
    python -c "import sys; print('Python executable:', sys.executable)"
    echo.
    echo To deactivate, run: deactivate
    echo ========================================
) else (
    echo.
    echo ERROR: Failed to activate virtual environment
    echo Make sure .venv exists in the project root
)