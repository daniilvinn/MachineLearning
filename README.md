# 🧠 MachineLearning

A repository that gathers all of my projects in the Machine Learning field, designed primarily for **educational purposes** and learning deep learning concepts through hands-on implementation.

## 📚 Educational Purpose

This repository serves as a collection of machine learning and deep learning projects built from scratch to understand fundamental concepts, architectures, and training techniques. Each project focuses on demonstrating specific ML/DL principles rather than production-ready implementations.

## 📖 Overview

This repository contains various machine learning projects implemented in PyTorch, focusing on computer vision, neural network architectures, and deep learning techniques. The projects are designed to be educational, showing step-by-step implementations of popular ML algorithms and architectures.

## 🤖 Models

### [🖼️ Deblur](Deblur/)
**Image Deblurring Network** - A CNN-based deep learning model for removing Gaussian blur from images.

- **Architecture**: 7-layer CNN with residual connections (1.33M parameters)
- **Training Approach**: Single-image overfitting for demonstration purposes
- **Loss Functions**: Combined MSE + VGG19 Perceptual Loss
- **Features**: Mixed precision training, gradient clipping, TF32 acceleration
- **Input/Output**: 512×512 RGB images
- **Use Case**: Educational demonstration of image restoration techniques

The model learns to map blurred images back to sharp versions using a symmetric encoder-decoder architecture with skip connections to preserve original image information.

## 🚀 Setup

### Quick Start

This project uses the **[Scripts/](Scripts/)** directory for easy environment setup across different platforms. The setup process will create a virtual environment and install all required dependencies automatically.

#### Windows Users:
```cmd
# Run from project root directory
Scripts\WinSetup.bat
```

#### Linux/macOS Users:
```bash
# Make executable and run from project root
chmod +x Scripts/LinuxSetup.sh
./Scripts/LinuxSetup.sh
```

#### Daily Usage (Activate Virtual Environment):

**Windows (Command Prompt):**
```cmd
Scripts\EnvActivate.bat
```

**Windows (PowerShell):**
```powershell
.\Scripts\EnvActivate.ps1
```

**Linux/macOS:**
```bash
chmod +x Scripts/EnvActivate.sh  # First time only
./Scripts/EnvActivate.sh
```

### What the Setup Does:
- 🐍 **Auto-detects Python**: Works with `python`, `py`, or `python3` commands
- 📦 **Creates Virtual Environment**: Isolated `.venv` folder in project root
- ⬇️ **Installs Dependencies**: All packages from `requirements.txt`
- ✅ **Verification**: Confirms successful setup with clear messaging

For detailed setup instructions, troubleshooting, and advanced usage, see **[Scripts/README.md](Scripts/README.md)**.

## 📋 Requirements

The following Python packages are automatically installed during setup:

- **torch>=2.0.0** - PyTorch deep learning framework
- **torchvision>=0.15.0** - Computer vision utilities and models
- **Pillow>=9.0.0** - Python Imaging Library for image processing
- **numpy>=1.21.0** - Numerical computing and array operations
- **matplotlib>=3.5.0** - Plotting and data visualization
- **pandas>=1.3.0** - Data manipulation and analysis
- **scipy>=1.7.0** - Scientific computing and optimization
- **argparse** - Command-line argument parsing

### System Requirements:
- **Python 3.8+** (Python 3.9+ recommended)
- **CUDA-compatible GPU** (optional, but recommended for faster training)
- **4GB+ RAM** (8GB+ recommended for GPU training)

## 💻 Hardware Recommendations

- **GPU**: CUDA-compatible GPU with 4GB+ VRAM for optimal performance
- **CPU**: Fallback support available but significantly slower
- **Memory**: 8GB+ system RAM recommended
- **Storage**: SSD recommended for faster data loading

## 📁 Project Structure

```
MachineLearning/
├── .venv/                    # Virtual environment (created during setup)
├── Scripts/                  # Setup and activation scripts
│   ├── WinSetup.bat         # Windows environment setup
│   ├── LinuxSetup.sh        # Linux/macOS environment setup
│   ├── EnvActivate.*        # Environment activation scripts
│   └── README.md            # Detailed setup documentation
├── Deblur/                  # Image deblurring project
│   ├── Deblur.py           # Main training script
│   ├── PerceptualLoss.py   # VGG19 perceptual loss implementation
│   ├── data/               # Training images directory
│   └── README.md           # Project-specific documentation
├── requirements.txt         # Python dependencies
├── LICENSE                 # Project license
└── README.md               # This file
```

## 🎯 Getting Started

1. **Clone the repository**
2. **Run the appropriate setup script** from the `Scripts/` directory
3. **Activate the virtual environment** using the activation scripts
4. **Navigate to a project folder** (e.g., `Deblur/`) 
5. **Follow project-specific README** for detailed instructions

## 🔮 Future Projects

This repository will continue to grow with additional machine learning projects covering:
- Generative models (GANs, VAEs)
- Natural Language Processing
- Reinforcement Learning
- Object Detection and Segmentation
- Transfer Learning examples

## ⚠️ Educational Notice

These projects are designed for learning purposes and may not be optimized for production use. They emphasize clarity and educational value over performance optimization.

## 📄 License

This project is for educational purposes. Please refer to the [LICENSE](LICENSE) file for usage terms.