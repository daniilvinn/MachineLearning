import torch.nn as nn
import torchvision.models as models
import torch

# Perceptual Loss using VGG features
class VGG19Loss(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Load pre-trained VGG19 (using new weights parameter)
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.feature_extractor = nn.ModuleList(vgg[:36])  # Extract features up to conv4_4
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Freeze VGG parameters
        
        # Move to device
        self.feature_extractor.to(device)
    
    def extract_features(self, x):
        """Extract features from VGG19 at multiple layers"""
        features = []
        for layer in self.feature_extractor:
            x = layer(x)
            # Collect features at certain layers (after ReLU activations)
            if isinstance(layer, nn.ReLU):
                features.append(x)
        return features
        
    def forward(self, predicted, target):
        # Extract features from both images
        pred_features = self.extract_features(predicted)
        target_features = self.extract_features(target)
        
        # Calculate perceptual loss
        perceptual_loss = 0
        for pred_feat, target_feat in zip(pred_features, target_features):
            perceptual_loss += torch.nn.functional.mse_loss(pred_feat, target_feat)
        
        return perceptual_loss