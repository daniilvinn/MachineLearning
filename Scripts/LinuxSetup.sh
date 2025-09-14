#!/bin/bash

echo "Setting up the Machine Learning repository..."
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check which Python command is available
PYTHON_CMD=""
if command_exists python3; then
    PYTHON_CMD="python3"
    echo "Found Python: python3"
elif command_exists python; then
    PYTHON_CMD="python"
    echo "Found Python: python"
elif command_exists py; then
    PYTHON_CMD="py"
    echo "Found Python: py"
else
    echo "ERROR: Python is not installed or not in PATH"
    echo "Please install Python and make sure one of these commands is available:"
    echo "  - python3 (recommended)"
    echo "  - python"
    echo "  - py"
    exit 1
fi

echo "Changing to project root directory..."
cd ..

echo "Creating virtual environment (.venv)..."
$PYTHON_CMD -m venv .venv

# Check if virtual environment was created successfully
if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip..."
$PYTHON_CMD -m pip install --upgrade pip

echo "Installing required packages from requirements.txt..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install some packages"
    echo "Please check the error messages above"
    exit 1
fi

echo ""
echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo ""
echo "To activate the virtual environment manually, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To deactivate the virtual environment, run:"
echo "  deactivate"
echo ""
