# 🖼️ Multi-Dataset Image Recognition Network

A comprehensive deep learning project that implements a Wide Convolutional Neural Network (CNN) for multi-class image recognition using PyTorch. This network is designed to classify images across 1000+ classes from 10 different datasets, creating a unified recognition system capable of identifying digits, fashion items, objects, food, flowers, pets, and more.

## Project Overview

This project demonstrates advanced multi-dataset deep learning for image recognition, combining multiple popular computer vision datasets into a single unified classification system. The network learns to distinguish between hundreds of different classes ranging from handwritten digits to complex food dishes, making it one of the most comprehensive classification systems in this repository.

Key highlights include:
- **Multi-dataset training** across 10+ popular computer vision datasets
- **Wide CNN architecture** optimized for handling large numbers of classes
- **Mixed precision training** for efficient GPU utilization
- **Extensive data augmentation** to improve generalization
- **Automatic checkpoint saving** for easy model resumption
- **Interactive testing interface** with confidence scores and top-5 predictions

## 🏗️ Model Architecture


### Network Design
The `WideMultiCNN` is a wide CNN architecture designed to handle many classes efficiently by using more filters per layer while keeping the network relatively shallow to avoid vanishing gradients.

**Architecture Flow:**
```
Input (64×64×3) → [Wide Block 1] → 32×32×128 → [Wide Block 2] → 16×16×256 →
[Wide Block 3] → 8×8×512 → [Wide Block 4] → 4×4×1024 → [Wide Classifier] → Output
```

**Detailed Layer Structure:**

**Wide Block 1 (64×64 → 32×32):**
- Conv2d(3→128, kernel=3×3, padding=1) + BatchNorm2d + ReLU
- Conv2d(128→128, kernel=3×3, padding=1) + BatchNorm2d + ReLU  
- MaxPool2d(2×2) + Dropout(0.25)

**Wide Block 2 (32×32 → 16×16):**
- Conv2d(128→256, kernel=3×3, padding=1) + BatchNorm2d + ReLU
- Conv2d(256→256, kernel=3×3, padding=1) + BatchNorm2d + ReLU
- MaxPool2d(2×2) + Dropout(0.25)

**Wide Block 3 (16×16 → 8×8):**
- Conv2d(256→512, kernel=3×3, padding=1) + BatchNorm2d + ReLU
- Conv2d(512→512, kernel=3×3, padding=1) + BatchNorm2d + ReLU
- MaxPool2d(2×2) + Dropout(0.25)

**Wide Block 4 (8×8 → 4×4):**
- Conv2d(512→1024, kernel=3×3, padding=1) + BatchNorm2d + ReLU
- Conv2d(1024→1024, kernel=3×3, padding=1) + BatchNorm2d + ReLU
- MaxPool2d(2×2) + Dropout(0.25)

**Wide Classifier:**
- Flatten()
- Linear(1024×4×4 → 2048) + ReLU + Dropout(0.5)
- Linear(2048 → 1024) + ReLU + Dropout(0.5)
- Linear(1024 → num_classes)

**Total Parameters: ~83 Million** (varies with exact number of classes)

### Key Features
- **Wide Architecture**: More filters per layer for better feature extraction
- **Batch Normalization**: Accelerates training and improves stability
- **Progressive Dropout**: Increasing dropout rates in deeper layers (0.25 → 0.5)
- **Extensive Regularization**: Batch normalization + dropout + weight decay
- **Efficient Design**: Optimized for handling 1000+ classes without vanishing gradients

## 📊 Datasets

The model is trained on a combination of 10 popular computer vision datasets, creating a unified classification system:

### Core Datasets (Always Included):

1. **MNIST (10 classes)** - Handwritten digits (0-9)
   - Labels: `digit_0` to `digit_9`
   - 60,000 training + 10,000 test samples

2. **Fashion-MNIST (10 classes)** - Fashion items
   - Labels: `t-shirt`, `trouser`, `pullover`, `dress`, `coat`, `sandal`, `shirt`, `sneaker`, `bag`, `ankle_boot`
   - 60,000 training + 10,000 test samples

3. **CIFAR-10 (10 classes)** - Common objects  
   - Labels: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`
   - 50,000 training + 10,000 test samples

4. **CIFAR-100 (100 classes)** - Diverse objects
   - Labels: 100 fine-grained categories (animals, vehicles, household items, etc.)
   - 50,000 training + 10,000 test samples

### Extended Datasets (Downloaded when Available):

5. **SVHN (10 classes)** - Street View House Numbers
   - Labels: `svhn_digit_0` to `svhn_digit_9`
   - Real-world digit recognition in natural scenes

6. **STL-10 (10 classes)** - High-resolution objects
   - Labels: `stl_airplane`, `stl_bird`, `stl_car`, etc.
   - Higher resolution (96×96) images

7. **Food-101 (101 classes)** - Food dishes
   - Labels: Real food names (`apple_pie`, `sushi`, `pizza`, etc.)
   - Diverse culinary recognition

8. **Flowers-102 (102 classes)** - Flower species
   - Labels: `flower_001` to `flower_102`
   - Fine-grained botanical classification

9. **Oxford-IIIT Pet (37 classes)** - Pet breeds
   - Labels: `pet_01` to `pet_37`
   - Cat and dog breed recognition

10. **Caltech-101/256** - Object categories
    - Caltech-101: 102 classes (`caltech101_001` to `caltech101_102`)
    - Caltech-256: 257 classes (`caltech256_001` to `caltech256_257`)

**Total Classes: 1000+** (varies based on successfully downloaded datasets)

## 🎯 Training Details

### Multi-Dataset Strategy
The training combines all available datasets using a `UnifiedDataset` wrapper that:
- Assigns unique label offsets to each dataset to prevent class conflicts
- Applies appropriate transforms (grayscale→RGB conversion for MNIST-style datasets)
- Creates a unified training pipeline with consistent preprocessing

### Data Augmentation
**Training Transform Pipeline:**
```python
- Resize to 72×72 pixels
- Random crop to 64×64 pixels (target resolution)
- Random horizontal flip (50% probability)
- Random rotation (±15 degrees)
- Color jitter (brightness, contrast, saturation, hue)
- Normalize to [-1, 1] range
```

**Test Transform Pipeline:**
```python
- Direct resize to 64×64 pixels
- Normalize to [-1, 1] range (no augmentation)
```

### Training Configuration
- **Mixed Precision Training**: Uses `torch.amp.autocast()` and `GradScaler` for efficiency
- **Optimizer**: AdamW with weight decay (1e-4) for L2 regularization
- **Learning Rate**: 0.001 with Cosine Annealing scheduler
- **Batch Size**: 32
- **Epochs**: 100 (with automatic checkpoint saving)
- **Gradient Clipping**: Implicit through mixed precision scaling

### Loss Function
- **CrossEntropy Loss**: Standard multi-class classification loss
- **Automatic Class Balancing**: Natural balancing through dataset combination

## 🚀 Setup and Usage

### 📁 Training the Model

To train the model on all available datasets:

```bash
# Navigate to ImageRecognition directory
cd ImageRecognition/

# Start training
python Network.py
```

The training script will:
1. Automatically download all available datasets
2. Create a unified training pipeline
3. Train for 100 epochs with automatic checkpoint saving
4. Display training progress and accuracy metrics
5. Save model checkpoints after every epoch

### 🔍 Testing with Custom Images

To test the trained model on your own images:

```bash
# Test with a custom image
python Test.py path/to/your/image.jpg

# Test with specific model checkpoint
python Test.py path/to/your/image.jpg --model wide_multi_cnn_model_epoch_050.pth
```

The test script will:
1. Automatically detect and load the latest checkpoint (or use specified model)
2. Preprocess your image to 64×64 resolution
3. Make predictions with confidence scores
4. Display top 5 predictions with probabilities
5. Show original full-resolution image alongside predictions

### ⚙️ Hyperparameter Tuning

#### Learning Rate and Optimizer
**Location**: Lines 84-90 in `Network.py`
```python
# Modify these values for different training behavior
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```
- **Lower LR (1e-4 to 5e-4)**: More stable training, slower convergence
- **Higher LR (5e-3 to 1e-2)**: Faster convergence, risk of instability
- **Weight Decay**: Controls L2 regularization strength

#### Batch Size  
**Location**: Line 248 in `Network.py`
```python
train_loader, test_loader, label_names = get_multi_datasets(batch_size=32)
```
- **Smaller batches (16-24)**: Better for limited GPU memory, more noise in gradients
- **Larger batches (48-64)**: More stable gradients, requires more GPU memory

#### Number of Epochs
**Location**: Line 261 in `Network.py`
```python
train_losses, train_accuracies, test_accuracies = train_model(
    model, train_loader, test_loader, label_names, epochs=100
)
```

## 📋 Requirements

- **PyTorch** (≥2.0.0) - Deep learning framework
- **torchvision** (≥0.15.0) - Computer vision utilities and datasets
- **PIL (Pillow)** (≥9.0.0) - Image processing
- **NumPy** (≥1.21.0) - Numerical computing
- **Matplotlib** (≥3.5.0) - Plotting and visualization
- **argparse** - Command-line argument parsing

## 💻 Hardware Recommendations

- **GPU**: CUDA-compatible GPU with 8GB+ VRAM for optimal training
- **Memory**: 16GB+ system RAM recommended for large dataset handling
- **Storage**: 50GB+ free space for all datasets and model checkpoints
- **CPU**: Multi-core processor for data loading (with num_workers=2)

**Performance Notes:**
- Training time: ~4-6 hours on RTX 5070 Ti for 100 epochs
- Model size: ~214MB per checkpoint file
- Dataset download: ~15GB total for all datasets

## 📊 Output and Monitoring

### Training Output
The model provides comprehensive training feedback:

```
Epoch [1/100], Loss: 2.8456, Train Acc: 45.23%, Test Acc: 42.18%
Model saved to wide_multi_cnn_model_epoch_001.pth
Epoch [10/100], Loss: 1.2341, Train Acc: 72.45%, Test Acc: 68.32%
Model saved to wide_multi_cnn_model_epoch_010.pth
```

### Testing Output  
The test script shows detailed prediction results:

```
Top prediction: pizza
Confidence: 87.3%

Top 5 predictions:
1. pizza: 87.3%
2. spaghetti_bolognese: 8.2%
3. lasagna: 2.1%
4. cheese_plate: 1.5%
5. garlic_bread: 0.9%
```

### Visualization
Both training and testing scripts provide matplotlib visualizations:
- **Training**: Loss curves and accuracy plots over epochs
- **Testing**: Original image display with top-5 prediction bar chart

## ⚠️ Limitations

1. **Dataset Dependency**: Some datasets may fail to download due to server issues
2. **Fixed Resolution**: All images are resized to 64×64, potentially losing detail
3. **Class Imbalance**: Different datasets contribute different numbers of samples
4. **Memory Requirements**: Large combined dataset requires significant RAM/VRAM
5. **Single Image Format**: Only RGB images supported (grayscale converted automatically)

## 🔮 Future Improvements

- **Higher Resolution**: Train on 128×128 or 224×224 images for better detail preservation
- **Architecture Upgrades**: Experiment with ResNet, EfficientNet, or Vision Transformer architectures
- **Advanced Augmentation**: Add mixup, cutmix, or autoaugment techniques
- **Transfer Learning**: Use pre-trained backbones (ImageNet) for better initialization
- **Class Balancing**: Implement weighted sampling for more balanced training
- **Real-time Inference**: Optimize model for mobile/edge deployment
- **Additional Datasets**: Include ImageNet subsets, custom domain-specific datasets

## 📁 File Structure

```
ImageRecognition/
├── Network.py              # Main training script and model definition
├── Test.py                 # Interactive testing script for custom images
├── Dataset.py              # Multi-dataset loader with unified labeling
├── data/                   # Downloaded datasets directory (created automatically)
├── wide_multi_cnn_model_epoch_*.pth  # Model checkpoints (saved during training)
├── wide_multi_cnn_model.pth          # Final model (if manually saved)
└── README.md               # This file
```

## 🎯 Getting Started

1. **Start Training**:
   ```bash
   python Network.py
   ```
   This will download all datasets and begin training automatically.

2. **Test Your Images**:
   ```bash
   python Test.py your_image.jpg
   ```

3. **Monitor Progress**:
   - Watch console output for training metrics
   - Model checkpoints are saved every epoch
   - Training plots will display after completion

## License

This project is for educational purposes. Please refer to the main project license for usage terms.
