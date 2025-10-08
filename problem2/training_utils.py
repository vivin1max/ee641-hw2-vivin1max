"""Training for hierarchical VAE with posterior collapse prevention."""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

def train_hierarchical_vae(model, data_loader, num_epochs=100, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # KL annealing schedule
    def kl_anneal_schedule(epoch):
        # Cyclical annealing 
        cycle_len = 20
        cycle_pos = epoch % cycle_len
        if cycle_pos < cycle_len // 2:
            return cycle_pos / (cycle_len // 2) * 1.0  
        else:
            return 1.0  
    
    free_bits = 0.5  
    
    history = defaultdict(list)
    
    for epoch in range(num_epochs):
        beta = kl_anneal_schedule(epoch)
        
        for batch_idx, patterns in enumerate(data_loader):
            patterns = patterns.to(device)
            
            # Forward
            recon_logits, mu_low, logvar_low, mu_high, logvar_high = model(patterns[0], beta=beta)
            
            recon_loss = nn.functional.binary_cross_entropy_with_logits(
                recon_logits.reshape(-1), patterns[0].reshape(-1), reduction='sum'
            )
            
            kl_low = -0.5 * torch.sum(1 + logvar_low - mu_low.pow(2) - logvar_low.exp())
            kl_high = -0.5 * torch.sum(1 + logvar_high - mu_high.pow(2) - logvar_high.exp())
            
            kl_low = torch.max(kl_low, torch.tensor(free_bits * model.z_low_dim).to(device))
            kl_high = torch.max(kl_high, torch.tensor(free_bits * model.z_high_dim).to(device))
            
            total_loss = recon_loss + beta * (kl_low + kl_high)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Log metrics
            if batch_idx % 10 == 0:
                history['total_loss'].append(float(total_loss.item()))
                history['recon_loss'].append(float(recon_loss.item()))
                history['kl_low'].append(float(kl_low.item()))
                history['kl_high'].append(float(kl_high.item()))
                history['beta'].append(float(beta))
                history['epoch'].append(float(epoch + batch_idx/len(data_loader)))
        
        print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}, Beta = {beta:.3f}, "
              f"Recon = {recon_loss.item():.4f}, KL_low = {kl_low.item():.2f}, KL_high = {kl_high.item():.2f}")
    
    return history

def sample_diverse_patterns(model, n_styles=5, n_variations=10, device='cuda'):
    model.eval()
    patterns = []
    
    with torch.no_grad():
        z_high_samples = torch.randn(n_styles, model.z_high_dim).to(device)
        
        for i, z_high in enumerate(z_high_samples):
            style_patterns = []
            z_high_expanded = z_high.unsqueeze(0).repeat(n_variations, 1)
            z_low_samples = torch.randn(n_variations, model.z_low_dim).to(device)
            pattern_logits = model.decode_hierarchy(z_high_expanded, z_low_samples)
            pattern_probs = torch.sigmoid(pattern_logits)
            
            style_patterns.append(pattern_probs.cpu().numpy())
            patterns.append(style_patterns)
    
    return np.array(patterns)  

def analyze_posterior_collapse(model, data_loader, device='cuda'):
    model.eval()
    kl_dims_low = []
    kl_dims_high = []
    
    with torch.no_grad():
        for patterns, _, _ in data_loader:
            patterns = patterns.to(device)
            
            # Encode validation data
            mu_low, logvar_low, mu_high, logvar_high = model.encode_hierarchy(patterns)
            
            # Compute KL divergence per dimension
            kl_per_dim_low = -0.5 * (1 + logvar_low - mu_low.pow(2) - logvar_low.exp())
            kl_per_dim_high = -0.5 * (1 + logvar_high - mu_high.pow(2) - logvar_high.exp())
            
            kl_dims_low.append(kl_per_dim_low.mean(0).cpu())
            kl_dims_high.append(kl_per_dim_high.mean(0).cpu())
    
    # Average across batches
    avg_kl_low = torch.stack(kl_dims_low).mean(0)
    avg_kl_high = torch.stack(kl_dims_high).mean(0)
    
    # Identify collapsed dimensions 
    collapsed_low = (avg_kl_low < 0.1).sum().item()
    collapsed_high = (avg_kl_high < 0.1).sum().item()
    
    return {
        'kl_per_dim_low': avg_kl_low.numpy(),
        'kl_per_dim_high': avg_kl_high.numpy(),
        'collapsed_low': collapsed_low,
        'collapsed_high': collapsed_high,
        'utilization_low': 1 - collapsed_low / model.z_low_dim,
        'utilization_high': 1 - collapsed_high / model.z_high_dim
    }