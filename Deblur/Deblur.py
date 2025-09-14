import torch
import torch.nn as nn
import torch.optim as optim
import PIL.Image as Image
import PIL.ImageFilter as ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from PerceptualLoss import VGG19Loss

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Enable TF32 for faster training on CUDA
if(torch.cuda.is_available()):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Load data sample
image = Image.open("data/sample.jpg")
image = image.resize((512, 512))
blurred_image = image.filter(ImageFilter.GaussianBlur(radius=5))

# Convert images to numpy arrays, then to float32 tensors
image = torch.from_numpy(np.array(image)).float() / 255.0
blurred_image = torch.from_numpy(np.array(blurred_image)).float() / 255.0

# Convert to tensors and add batch dimension
image = image.permute(2, 0, 1)
blurred_image = blurred_image.permute(2, 0, 1)

image = image.unsqueeze(0)
blurred_image = blurred_image.unsqueeze(0)

# Move tensors to device
image = image.to(device)
blurred_image = blurred_image.to(device)

class DeblurNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Residual connection to preserve color information
        residual = x
        x = self.main(x)
        x = x + residual
        return self.sigmoid(x)

net = DeblurNet().to(device)


# Hyperparameters
num_epochs = 20000
lr = 0.0001
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)

# Initialize loss functions
print("Initializing loss functions...")
mse_loss = nn.MSELoss()
print("MSE loss initialized")

print("Loading VGG19 for perceptual loss...")
try:
    perceptual_loss_fn = VGG19Loss(device)
    print("Perceptual loss initialized successfully")
except Exception as e:
    print(f"Error initializing perceptual loss: {e}")
    print("Falling back to MSE loss only")
    perceptual_loss_fn = None

# Loss weights
mse_weight = 1.0
perceptual_weight = 0.1 if perceptual_loss_fn is not None else 0.0

print(f"Loss weights - MSE: {mse_weight}, Perceptual: {perceptual_weight}")

# Initialize mixed precision training 
scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')

print("Starting training...")
# Training loop with mixed precision
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Use autocast for mixed precision training
    with torch.amp.autocast(device.type):
        # Forward pass: only pass the blurred image
        predicted = net(blurred_image)
        
        # Calculate combined loss
        mse = mse_loss(predicted, image)
        if perceptual_loss_fn is not None:
            perceptual = perceptual_loss_fn(predicted, image)
            loss = mse_weight * mse + perceptual_weight * perceptual
        else:
            perceptual = torch.tensor(0.0)  # Dummy value for logging
            loss = mse_weight * mse

    # Use scaler for backward pass
    scaler.scale(loss).backward()
    
    # Gradient clipping to prevent exploding gradients
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    
    # Update weights
    scaler.step(optimizer)
    scaler.update()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Total Loss: {loss.item():.6f}, MSE: {mse.item():.6f}, Perceptual: {perceptual.item():.6f}")

# After training, visualize the results
with torch.no_grad():
    with torch.amp.autocast(device.type):
        predicted_image = net(blurred_image)
    
    # Convert tensors back to float32 and numpy arrays for visualization
    original = image.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    blurred = blurred_image.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    reconstructed = predicted_image.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    
    # Clamp values to [0, 1] range
    original = np.clip(original, 0, 1)
    blurred = np.clip(blurred, 0, 1)
    reconstructed = np.clip(reconstructed, 0, 1)

# Plot the results
try:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(blurred)
    axes[1].set_title('Blurred Image')
    axes[1].axis('off')

    axes[2].imshow(reconstructed)
    axes[2].set_title('Reconstructed Image')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    # Save the reconstructed image
    reconstructed_pil = Image.fromarray((reconstructed * 255).astype(np.uint8))
    reconstructed_pil.save('reconstructed_image.jpg')
    print("Reconstructed image saved as 'reconstructed_image.jpg'")
    
except Exception as e:
    print(f"Error in visualization: {e}")
    print(f"Original shape: {original.shape}, dtype: {original.dtype}")
    print(f"Blurred shape: {blurred.shape}, dtype: {blurred.dtype}")
    print(f"Reconstructed shape: {reconstructed.shape}, dtype: {reconstructed.dtype}")
    print(f"Reconstructed min/max: {reconstructed.min():.3f}/{reconstructed.max():.3f}")

