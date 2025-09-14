import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from PerceptualLoss import VGG19Loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Enable TF32 for faster training on CUDA
if(torch.cuda.is_available()):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

class ZipNet(nn.Module):
    def __init__(self):
        super(ZipNet, self).__init__()
        self.encoder = nn.Sequential(
            # First conv layer
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Downsample by 2x (H/2, W/2)
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            # Downsample by 2x again (H/4, W/4) - total 4x smaller
            nn.MaxPool2d(2, 2),
            # Final conv to latent space with 4 channels
            nn.Conv2d(128, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            # Start from latent space (4 channels)
            nn.Conv2d(4, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            # Upsample by 2x (H*2, W*2)
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Upsample by 2x again (H*4, W*4) - back to original size
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            # Final conv to RGB output
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

net = ZipNet()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
scaler = torch.amp.GradScaler(device_type)
net.to(device)

# Load data sample
image = Image.open("data/sample.jpg").resize((256, 256))
image = np.array(image)
image = torch.from_numpy(image).float() / 255.0
image = image.permute(2, 0, 1)
image = image.unsqueeze(0)
image = image.to(device)

# Define loss functions
mse_criterion = nn.MSELoss()
perceptual_criterion = VGG19Loss(device, device_type)

# Loss weights for combining MSE and perceptual losses
mse_weight = 1.0
perceptual_weight = 0.1

# Train the network
for i in range(1000):
    optimizer.zero_grad()
    
    # Forward pass with autocast
    with torch.amp.autocast(device_type):
        reconstructed = net(image)
        # Calculate combined loss (MSE + Perceptual)
        mse_loss = mse_criterion(reconstructed, image)
        perceptual_loss = perceptual_criterion(reconstructed, image)
        loss = mse_weight * mse_loss + perceptual_weight * perceptual_loss
    
    # Backward pass with scaled gradients
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    if i % 100 == 0:
        print(f"Epoch {i}, Total Loss: {loss.item():.6f}, MSE: {mse_loss.item():.6f}, Perceptual: {perceptual_loss.item():.6f}")

# Plot original vs reconstructed image and save latent space
with torch.no_grad():
    net.eval()
    
    # Get latent representation with autocast
    with torch.amp.autocast(device_type):
        latent = net.encoder(image)
        reconstructed = net.decoder(latent)
    
    # Save latent space to file
    torch.save(latent, 'data/latent_space.ls')
    print(f"Latent space shape: {latent.shape}")
    print(f"Latent space saved to 'data/latent_space.ls'")
    print(f"Reconstructed image shape: {reconstructed.shape}")
    
    # Convert tensors to numpy arrays for plotting
    original_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
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

