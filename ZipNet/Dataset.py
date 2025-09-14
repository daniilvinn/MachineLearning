import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image

class MultiDatasetLoader:
    """Downloads and manages multiple image datasets"""
    
    def __init__(self, data_dir="data/datasets", image_size=512):
        self.data_dir = data_dir
        self.image_size = image_size
        os.makedirs(data_dir, exist_ok=True)
        
        # Base transform for resizing and converting to tensor
        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
    def download_datasets(self):
        """Download multiple datasets with diverse content"""
        print("Downloading datasets...")
        
        datasets_list = []
        
        # CIFAR-10: animals, vehicles, objects
        print("Downloading CIFAR-10 (animals, vehicles, objects)...")
        cifar10 = datasets.CIFAR10(
            root=os.path.join(self.data_dir, 'cifar10'),
            train=True,
            download=True,
            transform=self.base_transform
        )
        datasets_list.append(cifar10)
        
        # STL-10: higher resolution, similar categories
        print("Downloading STL-10 (higher resolution diverse images)...")
        stl10 = datasets.STL10(
            root=os.path.join(self.data_dir, 'stl10'),
            split='train',
            download=True,
            transform=self.base_transform
        )
        datasets_list.append(stl10)
        
        # STL-10 unlabeled (more diverse content)
        print("Downloading STL-10 unlabeled (additional diverse images)...")
        stl10_unlabeled = datasets.STL10(
            root=os.path.join(self.data_dir, 'stl10'),
            split='unlabeled',
            download=True,
            transform=self.base_transform
        )
        datasets_list.append(stl10_unlabeled)
        
        print(f"Downloaded {len(datasets_list)} datasets")
        print(f"Total images: CIFAR-10: {len(cifar10)}, STL-10 labeled: {len(stl10)}, STL-10 unlabeled: {len(stl10_unlabeled)}")
        
        return datasets_list

class AugmentedDataset(Dataset):
    """Dataset that extends original data with augmentations"""
    
    def __init__(self, original_datasets, augmentations_per_image=2):
        self.original_datasets = ConcatDataset(original_datasets)
        self.augmentations_per_image = augmentations_per_image
        
        # Augmentation transforms that extend the dataset
        self.augmentation_transforms = transforms.Compose([
            transforms.ToPILImage(),  # Convert tensor back to PIL for augmentations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10, fill=0),
            transforms.ColorJitter(
                brightness=0.15,   # ±15% brightness
                contrast=0.1,      # ±10% contrast
                saturation=0.1,    # ±10% saturation
                hue=0.03          # ±3% hue
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),  # Small translations
                scale=(0.95, 1.05),      # Small scaling
                fill=0
            ),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        # Return original + augmented versions
        return len(self.original_datasets) * (1 + self.augmentations_per_image)
    
    def __getitem__(self, idx):
        # Calculate which original image and which version (original or augmented)
        original_idx = idx // (1 + self.augmentations_per_image)
        version_idx = idx % (1 + self.augmentations_per_image)
        
        # Get original image
        original_image, _ = self.original_datasets[original_idx]
        
        if version_idx == 0:
            # Return original image
            return original_image
        else:
            # Return augmented version
            return self.augmentation_transforms(original_image)

def create_dataset(data_dir="data/datasets", image_size=512, augmentations_per_image=2, max_samples=None):
    """Create the complete augmented dataset"""
    
    # Initialize dataset loader
    loader = MultiDatasetLoader(data_dir, image_size)
    
    # Download datasets
    original_datasets = loader.download_datasets()
    
    # Limit dataset size if specified (for faster training/testing)
    if max_samples is not None:
        print(f"Limiting dataset to {max_samples} samples from each dataset")
        limited_datasets = []
        for dataset in original_datasets:
            # Create a subset of the dataset
            indices = list(range(min(len(dataset), max_samples)))
            subset = torch.utils.data.Subset(dataset, indices)
            limited_datasets.append(subset)
        original_datasets = limited_datasets
    
    # Create augmented dataset
    augmented_dataset = AugmentedDataset(original_datasets, augmentations_per_image)
    
    print(f"Created dataset with {len(augmented_dataset)} total images")
    print(f"({len(augmented_dataset) // (1 + augmentations_per_image)} original + "
          f"{len(augmented_dataset) - len(augmented_dataset) // (1 + augmentations_per_image)} augmented)")
    
    return augmented_dataset

def create_dataloader(dataset, batch_size=8, shuffle=True, num_workers=2):
    """Create DataLoader for the dataset"""
    # On Windows, disable multiprocessing to avoid issues
    import platform
    if platform.system() == "Windows":
        print(f"Windows detected: Setting num_workers=0 to avoid multiprocessing issues")
        num_workers = 0
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )