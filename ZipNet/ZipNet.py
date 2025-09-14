import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from PerceptualLoss import VGG19Loss
from Dataset import create_dataset, create_dataloader

class ZipNet(nn.Module):
    def __init__(self, dropout_rate=0.2, noise_std=0.1):
        super(ZipNet, self).__init__()
        self.noise_std = noise_std
        
        # Enhanced encoder for 1/64 compression (512x512 -> 32x32x4)
        # Need 4 downsampling steps of 2x each (2^4 = 16x reduction)
        self.encoder = nn.Sequential(
            # First conv block (512x512 -> 512x512)
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            # Downsample to 256x256
            nn.MaxPool2d(2, 2),
            
            # Second conv block (256x256 -> 256x256)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            # Downsample to 128x128
            nn.MaxPool2d(2, 2),
            
            # Third conv block (128x128 -> 128x128)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            # Downsample to 64x64
            nn.MaxPool2d(2, 2),
            
            # Fourth conv block (64x64 -> 64x64)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            # Downsample to 32x32
            nn.MaxPool2d(2, 2),
            
            # Final conv to latent space with 4 channels (32x32x4)
            nn.Conv2d(256, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Enhanced decoder with much more capacity (32x32x4 -> 512x512x3)
        self.decoder = nn.Sequential(
            # Start from latent space (4 channels at 32x32)
            nn.Conv2d(4, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            
            # First upsample block (32x32 -> 64x64)
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            
            # Second upsample block (64x64 -> 128x128)
            nn.ConvTranspose2d(256, 192, kernel_size=2, stride=2),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            
            # Third upsample block (128x128 -> 256x256)
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            
            # Fourth upsample block (256x256 -> 512x512)
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            
            # Final conv to RGB output
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode to latent space
        latent = self.encoder(x)
        
        # Add Gaussian noise during training only
        if self.training:
            noise = torch.randn_like(latent) * self.noise_std
            latent = latent + noise
        
        # Decode from latent space
        return self.decoder(latent)
        
    def encode(self, x):
        """Separate method to get latent representation"""
        return self.encoder(x)

if __name__ == '__main__':
    # Initialize device and settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Enable TF32 for faster training on CUDA and set memory management
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Clear CUDA cache to start fresh
        torch.cuda.empty_cache()
        
        # Set memory fraction to 80% (more conservative than 90%)
        torch.cuda.set_per_process_memory_fraction(0.8, 0)
        
        # Print GPU memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
        print(f"GPU Memory: {total_memory:.1f} GB total")
        print(f"Setting memory limit to 80% ({total_memory * 0.8:.1f} GB)")
        
        # Enable memory-efficient attention if available (for newer PyTorch versions)
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            print("Enabled Flash Attention for memory efficiency")
        except:
            pass

    net = ZipNet()
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.0001)
    scaler = torch.amp.GradScaler(device_type)
    net.to(device)

    # Model checkpoint path
    checkpoint_path = 'data/zipnet_checkpoint.pth'

    # Load model weights if checkpoint exists
    if os.path.exists(checkpoint_path):
        print("Loading existing model weights...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {start_epoch}")
    else:
        print("No existing checkpoint found. Training from scratch...")
        start_epoch = 0

    # Create comprehensive dataset with multiple datasets
    print("Creating dataset with diverse images (animals, landscapes, cities, humans)...")
    dataset = create_dataset(
        data_dir="data/datasets", 
        image_size=512, 
        augmentations_per_image=2,  # Each original image gets 2 augmented versions
        max_samples=1000  # Limit each dataset to 1000 samples for manageable training
    )

    # Create dataloader with reduced batch size to avoid OOM errors
    dataloader = create_dataloader(
        dataset, 
        batch_size=4,  # Reduced batch size to fit within memory limits
        shuffle=True, 
        num_workers=4
    )

    # Load a sample image for evaluation/plotting (from the original single image)
    original_image = Image.open("data/sample.jpg").resize((512, 512))
    original_image = transforms.ToTensor()(original_image).unsqueeze(0).to(device)

    # Define loss functions
    mse_criterion = nn.MSELoss()
    perceptual_criterion = VGG19Loss(device, device_type)

    # Loss weights for combining MSE and perceptual losses
    mse_weight = 1.0
    perceptual_weight = 0.1

    # Train the network
    total_epochs = 1
    print(f"Training on {len(dataset)} total images")
    print(f"Batch size: {dataloader.batch_size}, Batches per epoch: {len(dataloader)}")
    print(f"Dataset includes diverse content: animals, landscapes, cities, humans, and more")

    for epoch in range(start_epoch, total_epochs):
        net.train()  # Ensure model is in training mode
        epoch_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_perceptual_loss = 0.0
        
        for batch_idx, batch_images in enumerate(dataloader):
            batch_images = batch_images.to(device)
            optimizer.zero_grad()
            
            # Forward pass with autocast
            with torch.amp.autocast(device_type):
                reconstructed = net(batch_images)
                # Calculate combined loss (MSE + Perceptual)
                mse_loss = mse_criterion(reconstructed, batch_images)
                perceptual_loss = perceptual_criterion(reconstructed, batch_images)
                loss = mse_weight * mse_loss + perceptual_weight * perceptual_loss
            
            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        # Accumulate losses
        epoch_loss += loss.item()
        epoch_mse_loss += mse_loss.item()
        epoch_perceptual_loss += perceptual_loss.item()
        
        # Clear cache periodically to prevent memory buildup
        if batch_idx % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Print memory usage periodically
        if batch_idx % 100 == 0 and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            cached = torch.cuda.memory_reserved(0) / 1024**3      # GB
            print(f"  Batch {batch_idx}: GPU Memory - Allocated: {allocated:.1f}GB, Cached: {cached:.1f}GB")
    
    # Calculate average losses for the epoch
    avg_loss = epoch_loss / len(dataloader)
    avg_mse_loss = epoch_mse_loss / len(dataloader)
    avg_perceptual_loss = epoch_perceptual_loss / len(dataloader)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Avg Total Loss: {avg_loss:.6f}, Avg MSE: {avg_mse_loss:.6f}, Avg Perceptual: {avg_perceptual_loss:.6f}")

    # Save model checkpoint
    print("Saving model checkpoint...")
    torch.save({
        'epoch': total_epochs,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }, checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

    # Plot original vs reconstructed image and save latent space
    with torch.no_grad():
        net.eval()
        
        # Get latent representation with autocast (without noise in eval mode)
        with torch.amp.autocast(device_type):
            latent = net.encode(original_image)  # Use encode method to get clean latent representation
            reconstructed = net(original_image)  # Full forward pass in eval mode (no noise)
        
        # Save latent space to file
        torch.save(latent, 'data/latent_space.ls')
        print(f"Latent space shape: {latent.shape}")
        print(f"Latent space saved to 'data/latent_space.ls'")
        print(f"Reconstructed image shape: {reconstructed.shape}")
        
        # Convert tensors to numpy arrays for plotting
        original_np = original_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        reconstructed_np = reconstructed.squeeze(0).permute(1, 2, 0).cpu().float().numpy()

        # Prepare latent space visualization (show first 3 channels as RGB)
        latent_np = latent.squeeze(0).cpu().float().numpy()
        latent_rgb = latent_np[:3]  # Take first 3 channels
        latent_rgb = (latent_rgb - latent_rgb.min()) / (latent_rgb.max() - latent_rgb.min())  # Normalize to [0,1]
        latent_rgb = np.transpose(latent_rgb, (1, 2, 0))  # Change from CHW to HWC
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot original image
        axes[0].imshow(original_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Plot latent space
        axes[1].imshow(latent_rgb)
        axes[1].set_title(f'Latent Space\n(Shape: {latent.shape[1:]})')
        axes[1].axis('off')
        
        # Plot reconstructed image
        axes[2].imshow(reconstructed_np)
        axes[2].set_title('Reconstructed Image')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('data/reconstruction_result.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\nTraining completed! Reconstruction result saved as 'reconstruction_result.png'")

