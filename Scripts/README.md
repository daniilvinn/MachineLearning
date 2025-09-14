# Machine Learning Scripts Directory

This directory contains setup and activation scripts for the Machine Learning repository. These scripts help you prepare your development environment and manage Python virtual environments across different platforms.

## 📁 Files Overview

| File | Purpose | Platform |
|------|---------|----------|
| `WinSetup.bat` | Creates virtual environment and installs dependencies | Windows (CMD/PowerShell) |
| `LinuxSetup.sh` | Creates virtual environment and installs dependencies | Linux/macOS |
| `EnvActivate.bat` | Activates virtual environment with verification | Windows Command Prompt |
| `EnvActivate.ps1` | Activates virtual environment with verification | Windows PowerShell |
| `EnvActivate.sh` | Activates virtual environment with verification | Linux/macOS |

## 🚀 Quick Start

### First Time Setup

#### Windows Users:
```cmd
# Run from the Scripts directory
WinSetup.bat
```

#### Linux/macOS Users:
```bash
# Make executable and run from the Scripts directory
chmod +x LinuxSetup.sh
./LinuxSetup.sh
```

### Daily Usage - Activating Virtual Environment

#### Windows Command Prompt:
```cmd
Scripts\EnvActivate.bat
```

#### Windows PowerShell:
```powershell
.\Scripts\EnvActivate.ps1
```

#### Linux/macOS:
```bash
# Make executable (first time only)
chmod +x Scripts/EnvActivate.sh
./Scripts/EnvActivate.sh
```

## 📋 Detailed Script Descriptions

### Setup Scripts

#### `WinSetup.bat`
**Purpose:** Comprehensive Windows setup script
- **Auto-detects Python:** Supports `python`, `py`, and `python3` commands
- **Creates virtual environment:** `.venv` folder in project root
- **Installs dependencies:** All packages from `requirements.txt`
- **Error handling:** Clear error messages and validation steps
- **Usage:** Double-click or run from command line

#### `LinuxSetup.sh`
**Purpose:** Cross-platform setup script for Unix-like systems
- **Auto-detects Python:** Supports `python3`, `python`, and `py` commands
- **Creates virtual environment:** `.venv` folder in project root
- **Installs dependencies:** All packages from `requirements.txt`
- **Bash compatibility:** Works on Linux, macOS, WSL
- **Usage:** `chmod +x LinuxSetup.sh && ./LinuxSetup.sh`

### Activation Scripts

#### `EnvActivate.bat`
**Purpose:** Windows Command Prompt activation with verification
- **Activates virtual environment:** Sources `.venv\Scripts\activate.bat`
- **Verification checks:** Confirms `VIRTUAL_ENV` is set
- **Python path verification:** Shows which Python executable is active
- **User-friendly output:** Clear success/error messages

#### `EnvActivate.ps1`
**Purpose:** PowerShell activation with enhanced features
- **Native PowerShell:** Proper environment variable handling
- **Colored output:** Enhanced visual feedback
- **Comprehensive verification:** Multiple activation checks
- **PowerShell integration:** Works seamlessly in PowerShell terminals

#### `EnvActivate.sh`
**Purpose:** Unix-like system activation script
- **Cross-platform:** Linux, macOS, WSL compatible
- **Python fallback:** Handles both `python` and `python3`
- **Verification:** Confirms virtual environment activation
- **Error handling:** Clear messaging for troubleshooting

## 🛠️ Dependencies Installed

The setup scripts install these packages from `requirements.txt`:
- `torch>=2.0.0` - PyTorch deep learning framework
- `torchvision>=0.15.0` - Computer vision utilities
- `Pillow>=9.0.0` - Image processing
- `numpy>=1.21.0` - Numerical computing
- `matplotlib>=3.5.0` - Plotting and visualization
- `argparse` - Command line argument parsing
- `pandas>=1.3.0` - Data manipulation
- `scipy>=1.7.0` - Scientific computing

## 📁 Directory Structure After Setup

```
MachineLearning/
├── .venv/                    # Virtual environment (created by setup scripts)
│   ├── Scripts/ (Windows)    # Windows executables
│   ├── bin/ (Linux/macOS)    # Unix executables
│   └── lib/                  # Python packages
├── Scripts/
│   ├── WinSetup.bat         # Windows setup
│   ├── LinuxSetup.sh        # Linux/macOS setup
│   ├── EnvActivate.bat      # Windows CMD activation
│   ├── EnvActivate.ps1      # PowerShell activation
│   ├── EnvActivate.sh       # Linux/macOS activation
│   └── README.md            # This file
├── requirements.txt         # Python dependencies
└── ...                      # Your project files
```

## 🔧 Troubleshooting

### PowerShell Execution Policy Error

**Error Message:**
```
.\EnvActivate.ps1 : File cannot be loaded because running scripts is 
disabled on this system. For more information, see about_Execution_Policies...
```

**Solution:**
Run this command **once** in PowerShell to allow local scripts:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Alternative Solutions:**
1. **Bypass for single use:**
   ```powershell
   PowerShell -ExecutionPolicy Bypass -File "Scripts\EnvActivate.ps1"
   ```

2. **Use batch file instead:**
   ```cmd
   Scripts\EnvActivate.bat
   ```

3. **Unblock file:** Right-click `EnvActivate.ps1` → Properties → Unblock

### Virtual Environment Not Found

**Error:** `ERROR: Failed to activate virtual environment`

**Solutions:**
1. **Run setup first:** Execute `WinSetup.bat` or `LinuxSetup.sh`
2. **Check location:** Ensure `.venv` exists in project root
3. **Permissions:** On Linux/macOS, ensure scripts are executable with `chmod +x`

### Python Not Found

**Error:** `ERROR: Python is not installed or not in PATH`

**Solutions:**
1. **Install Python:** Download from [python.org](https://python.org)
2. **Add to PATH:** Ensure Python is accessible from command line
3. **Try alternatives:** Scripts support `python`, `py`, `python3` commands

## 🎯 Best Practices

1. **Always activate before coding:** Use activation scripts before working on the project
2. **Deactivate when done:** Run `deactivate` command when switching projects
3. **Use appropriate script:** PowerShell users should use `.ps1`, CMD users use `.bat`
4. **Keep dependencies updated:** Re-run setup scripts after updating `requirements.txt`

---

For more information about the Machine Learning project, see the main repository README.
