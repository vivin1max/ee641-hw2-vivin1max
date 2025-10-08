import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalDrumVAE(nn.Module):
    def __init__(self, z_high_dim=4, z_low_dim=12):
        super().__init__()
        self.z_high_dim = z_high_dim
        self.z_low_dim = z_low_dim
        
        # Encoder
        self.encoder_low = nn.Sequential(
            nn.Conv1d(9, 32, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Flatten()  # → [512]
        )
        
        # Low-level latent parameters
        self.fc_mu_low = nn.Linear(512, z_low_dim)
        self.fc_logvar_low = nn.Linear(512, z_low_dim)
        
        # Encoder from z_low to z_high
        self.encoder_high = nn.Sequential(
            nn.Linear(z_low_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # High-level latent parameters
        self.fc_mu_high = nn.Linear(32, z_high_dim)
        self.fc_logvar_high = nn.Linear(32, z_high_dim)
        
        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(z_high_dim + z_low_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 4 * 128) 
        )
        
        # Transpose convolutions for upsampling
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(),
            nn.ConvTranspose1d(32, 9, kernel_size=3, padding=1)  
        )
        
    def encode_hierarchy(self, x):
        x = x.transpose(1, 2).float()
        
        # Encode to z_low parameters
        encoded_low = self.encoder_low(x)
        mu_low = self.fc_mu_low(encoded_low)
        logvar_low = self.fc_logvar_low(encoded_low)
        
        # Sample z_low using reparameterization
        z_low = self.reparameterize(mu_low, logvar_low)
        
        # Encode z_low to z_high parameters
        encoded_high = self.encoder_high(z_low)
        mu_high = self.fc_mu_high(encoded_high)
        logvar_high = self.fc_logvar_high(encoded_high)
        
        return mu_low, logvar_low, mu_high, logvar_high
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode_hierarchy(self, z_high, z_low=None, temperature=1.0):
        if z_low is None:
            z_low = torch.randn(z_high.size(0), self.z_low_dim).to(z_high.device)
        
        # Concatenate latent codes
        z_combined = torch.cat([z_high, z_low], dim=1)
        
        # Decode through fully connected layers
        decoded = self.decoder_fc(z_combined)
        decoded = decoded.view(decoded.size(0), 128, 4)  # Reshape for conv transpose
        
        # Decode through transpose convolutions
        pattern_logits = self.decoder_conv(decoded)
        
        # Apply temperature scaling
        pattern_logits = pattern_logits / temperature
        pattern_logits = pattern_logits.transpose(1, 2)
        
        return pattern_logits
    
    def forward(self, x, beta=1.0, temperature=1.0):
        # Encode
        mu_low, logvar_low, mu_high, logvar_high = self.encode_hierarchy(x)
        
        # Sample latent codes
        z_low = self.reparameterize(mu_low, logvar_low)
        z_high = self.reparameterize(mu_high, logvar_high)
        
        # Decode
        recon_logits = self.decode_hierarchy(z_high, z_low, temperature)
        
        return recon_logits, mu_low, logvar_low, mu_high, logvar_high