"""
GAN models for font generation.
"""

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, conditional=False, num_classes=26):

        super().__init__()
        self.z_dim = z_dim
        self.conditional = conditional
        
        # Calculate input dimension
        input_dim = z_dim + (num_classes if conditional else 0)
        self.project = nn.Sequential(
            nn.Linear(input_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True)
        )
        
        # Upsample
        self.main = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z, class_label=None):

        # Concatenate z and class label if conditional
        if self.conditional and class_label is not None:
            input_tensor = torch.cat([z, class_label], dim=1)
        else:
            input_tensor = z
        
        # Project to spatial dimensions
        x = self.project(input_tensor)
        x = x.view(x.size(0), 128, 7, 7)
        x = self.main(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, conditional=False, num_classes=26):
        super().__init__()
        self.conditional = conditional
        self.features = nn.Sequential(
            # 28x28x1 -> 14x14x64
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # 14x14x64 -> 7x7x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # 7x7x128 -> 3x3x256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25)
        )
        
        # Calculate feature dimension after convolutions
        feature_dim = 256 * 3 * 3  
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + (num_classes if conditional else 0), 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, class_label=None):
        # Extract features
        features = self.features(img)
        features = features.view(features.size(0), -1) 
        if self.conditional and class_label is not None:
            features = torch.cat([features, class_label], dim=1)
        
        # Classify
        output = self.classifier(features)
        return output