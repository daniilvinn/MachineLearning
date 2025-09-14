@echo off
echo Setting up the Machine Learning repository...
echo.

REM Check which Python command is available
set PYTHON_CMD=
python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
    echo Found Python: python
    goto :python_found
)

py --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py
    echo Found Python: py
    goto :python_found
)

python3 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python3
    echo Found Python: python3
    goto :python_found
)

echo ERROR: Python is not installed or not in PATH
echo Please install Python and make sure one of these commands is available:
echo   - python
echo   - py
echo   - python3
pause
exit /b 1

:python_found
echo Changing to project root directory...
cd ..

echo Creating virtual environment (.venv)...
%PYTHON_CMD% -m venv .venv

REM Check if virtual environment was created successfully
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Upgrading pip...
%PYTHON_CMD% -m pip install --upgrade pip

echo Installing required packages from requirements.txt...
pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install some packages
    echo Please check the error messages above
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo To activate the virtual environment manually, run:
echo   .venv\Scripts\activate.bat
echo.
echo To deactivate the virtual environment, run:
echo   deactivate
echo.
pause
