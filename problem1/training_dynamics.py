"""GAN training and mode collapse analysis."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

def train_gan(generator, discriminator, data_loader, num_epochs=100, device='cuda',
              checkpoint_dir=None, checkpoint_interval=10):
    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Loss
    criterion = nn.BCELoss()
    
    # History
    history = defaultdict(list)
    
    for epoch in range(num_epochs):
        for batch_idx, (real_images, labels) in enumerate(data_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            d_optimizer.zero_grad()
            real_outputs = discriminator(real_images)
            d_real_loss = criterion(real_outputs, real_labels)
            z = torch.randn(batch_size, generator.z_dim).to(device)
            fake_images = generator(z).detach()
            fake_outputs = discriminator(fake_images)
            d_fake_loss = criterion(fake_outputs, fake_labels)
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            g_optimizer.zero_grad()
            z = torch.randn(batch_size, generator.z_dim).to(device)
            fake_images = generator(z)
            fake_outputs = discriminator(fake_images)
            g_loss = criterion(fake_outputs, real_labels)
            g_loss.backward()
            g_optimizer.step()
            if batch_idx % 10 == 0:
                history['d_loss'].append(float(d_loss.item()))
                history['g_loss'].append(float(g_loss.item()))
                history['epoch'].append(float(epoch + batch_idx/len(data_loader)))
        if epoch % 10 == 0:
            coverage_score, stats = analyze_mode_coverage(generator, device)
            history['mode_coverage'].append(float(coverage_score))
            history.setdefault('coverage_details', []).append(stats)
            print(f"Epoch {epoch}: Coverage = {coverage_score:.2f} Missing: {len(stats['missing_letters'])}")
        if checkpoint_dir and (epoch % checkpoint_interval == 0 or epoch == num_epochs - 1):
            cp_path = Path(checkpoint_dir) / f"generator_epoch_{epoch}.pth"
            torch.save({'epoch': epoch, 'generator_state_dict': generator.state_dict()}, cp_path)
            d_cp_path = Path(checkpoint_dir) / f"discriminator_epoch_{epoch}.pth"
            torch.save({'epoch': epoch, 'discriminator_state_dict': discriminator.state_dict()}, d_cp_path)
    return history

def analyze_mode_coverage(generator, device, n_samples=1000):
    from provided.metrics import mode_coverage_score
    generator.eval()
    with torch.no_grad():
        batches = []
        batch_size = 256
        remaining = n_samples
        while remaining > 0:
            cur = min(batch_size, remaining)
            z = torch.randn(cur, generator.z_dim, device=device)
            imgs = generator(z)  
            imgs = (imgs + 1) / 2
            batches.append(imgs.cpu())
            remaining -= cur
        all_imgs = torch.cat(batches, dim=0)  
        stats = mode_coverage_score(all_imgs)
    generator.train()
    return stats['coverage_score'], stats

def visualize_mode_collapse(history, save_path):

    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot losses
    epochs = history['epoch']
    ax1.plot(epochs, history['d_loss'], label='Discriminator Loss', alpha=0.7)
    ax1.plot(epochs, history['g_loss'], label='Generator Loss', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Losses')
    ax1.legend()
    ax1.grid(True)
    
    # Plot mode coverage over time
    coverage_epochs = list(range(0, len(history['mode_coverage']) * 10, 10))
    ax2.plot(coverage_epochs, history['mode_coverage'], 'ro-')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mode Coverage')
    ax2.set_title('Mode Collapse Over Time')
    ax2.set_ylim(0, 1)
    ax2.grid(True)
    
    # Histogram of final mode coverage
    if len(history['mode_coverage']) > 0:
        ax3.bar(['Final Coverage'], [history['mode_coverage'][-1]], color='skyblue')
        ax3.set_ylim(0, 1)
        ax3.set_ylabel('Coverage Score')
        ax3.set_title('Final Mode Coverage')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()