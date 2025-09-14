#!/bin/bash

echo "Activating virtual environment..."
cd ..

# Check if .venv exists
if [ ! -f ".venv/bin/activate" ]; then
    echo ""
    echo "ERROR: Failed to activate virtual environment"
    echo "Make sure .venv exists in the project root"
    echo "Run LinuxSetup.sh first to create the virtual environment"
    exit 1
fi

# Activate the virtual environment
source .venv/bin/activate

# Verify activation
if [ -n "$VIRTUAL_ENV" ]; then
    echo ""
    echo "========================================"
    echo "Virtual Environment Successfully Activated!"
    echo "========================================"
    echo "Virtual Environment Path: $VIRTUAL_ENV"
    echo "Expected Python Location: $VIRTUAL_ENV/bin/python"
    echo "Checking Python version from venv:"
    python --version || python3 --version
    echo "Verifying Python executable:"
    python -c "import sys; print('Python executable:', sys.executable)" 2>/dev/null || \
    python3 -c "import sys; print('Python executable:', sys.executable)"
    echo ""
    echo "To deactivate, run: deactivate"
    echo "========================================"
else
    echo ""
    echo "ERROR: Virtual environment activation failed"
    echo "The activation script ran but VIRTUAL_ENV was not set"
    exit 1
fi
