import torch
import torch.nn as nn
import torch.optim as optim
import torch.amp as amp
import matplotlib.pyplot as plt
import numpy as np
from Dataset import get_multi_datasets

class WideMultiCNN(nn.Module):
    def __init__(self, num_classes=1000):  # Support for 1000+ classes
        super(WideMultiCNN, self).__init__()
        
        # Wide CNN architecture - more filters per layer, fewer layers to avoid vanishing gradients
        # Input: 64x64x3
        self.features = nn.Sequential(
            # First wide block - 64x64 -> 32x32
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 32x32
            nn.Dropout(0.25),
            
            # Second wide block - 32x32 -> 16x16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 16x16
            nn.Dropout(0.25),
            
            # Third wide block - 16x16 -> 8x8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 8x8
            nn.Dropout(0.25),
            
            # Fourth wide block - 8x8 -> 4x4
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 4x4
            nn.Dropout(0.25)
        )
        
        # Wide classifier with more neurons
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, 2048),  # Much wider
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, test_loader, label_names, epochs=100, save_path_prefix='wide_multi_cnn_model'):
    """
    Train the wide CNN model with mixed precision and weight decay
    Saves model parameters after every epoch
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # CrossEntropy loss and AdamW optimizer with weight decay (L2 regularization)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Mixed precision training components
    scaler = amp.GradScaler()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print(f"Training on {device} with mixed precision")
    print(f"Total classes: {len(label_names)}")
    print(f"Training for {epochs} epochs")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Statistics
            running_loss += loss.item()
            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, LR: {current_lr:.6f}')
        
        epoch_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        
        # Testing phase
        model.eval()
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                with amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                    outputs = model(images)
                
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        
        test_accuracy = 100 * correct_test / total_test
        
        # Update learning rate
        scheduler.step()
        
        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, '
              f'Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')
        
        # Save model parameters after every epoch
        epoch_save_path = f'{save_path_prefix}_epoch_{epoch+1:03d}.pth'
        torch.save(model.state_dict(), epoch_save_path)
        print(f'Model saved to {epoch_save_path}')
    
    return train_losses, train_accuracies, test_accuracies

def plot_training_results(train_losses, train_accuracies, test_accuracies):
    """
    Plot training loss and accuracy curves
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    ax2.plot(epochs, test_accuracies, 'r-', label='Test Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_predictions(model, test_loader, label_names, num_samples=3):
    """
    Plot 3 image-label prediction pairs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Get random samples from test set
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Select first 3 samples
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Make predictions
    with torch.no_grad():
        images_gpu = images.to(device)
        outputs = model(images_gpu)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i in range(num_samples):
        img = images[i].permute(1, 2, 0).numpy()
        # Denormalize for display
        img = img * 0.5 + 0.5  # Assuming normalization was (x - 0.5) / 0.5
        img = np.clip(img, 0, 1)
        
        true_label = label_names[labels[i].item()]
        pred_label = label_names[predicted[i].item()]
        confidence = probabilities[i][predicted[i]].item() * 100
        
        axes[i].imshow(img)
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to train multi-dataset recognition with 1000+ classes
    """
    print("Loading multiple datasets with extensive augmentation...")
    train_loader, test_loader, label_names = get_multi_datasets(batch_size=32)  # Smaller batch for larger images
    
    print("Creating wide CNN model...")
    model = WideMultiCNN(num_classes=len(label_names))
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("Starting training with mixed precision...")
    train_losses, train_accuracies, test_accuracies = train_model(
        model, train_loader, test_loader, label_names, epochs=100, save_path_prefix='wide_multi_cnn_model'
    )
    
    print("Plotting training results...")
    plot_training_results(train_losses, train_accuracies, test_accuracies)
    
    # Final model is already saved after the last epoch
    print("Model parameters saved after every epoch during training")
    
    print("Plotting 3 prediction examples...")
    plot_predictions(model, test_loader, label_names)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
