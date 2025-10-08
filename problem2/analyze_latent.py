import torch
import numpy as np
import matplotlib.pyplot as plt

def _pca_2d(x: np.ndarray):
    x = x - x.mean(axis=0, keepdims=True)
    cov = np.cov(x, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    comps = eigvecs[:, idx[:2]]
    return x @ comps

def visualize_latent_hierarchy(model, data_loader, device='cuda', save_path='results/latent_analysis'):
    model.eval()
    
    z_highs = []
    z_lows = []
    styles = []
    with torch.no_grad():
        for patterns, style_labels, _ in data_loader:
            patterns = patterns.to(device)
            mu_low, _, mu_high, _ = model.encode_hierarchy(patterns)
            
            z_highs.append(mu_high.cpu().numpy())
            z_lows.append(mu_low.cpu().numpy())
            styles.extend(style_labels.numpy())
    
    z_highs = np.concatenate(z_highs, axis=0)
    z_lows = np.concatenate(z_lows, axis=0)
    styles = np.array(styles)
    
    # PCA to 2D for z_high
    if z_highs.shape[1] > 2:
        z_high_2d = _pca_2d(z_highs)
    else:
        z_high_2d = z_highs
    
    # Plot z_high colored by genre
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    style_names = ['rock', 'jazz', 'hiphop', 'electronic', 'latin']
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, (style, color) in enumerate(zip(style_names, colors)):
        mask = styles == i
        ax1.scatter(z_high_2d[mask, 0], z_high_2d[mask, 1], 
                   c=color, label=style, alpha=0.6)
    
    ax1.set_title('High-level Latent Space (Styles)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot z_low colored by genre 
    for i, (style, color) in enumerate(zip(style_names, colors)):
        mask = styles == i
        ax2.scatter(z_lows[mask, 0], z_lows[mask, 1], 
                   c=color, label=style, alpha=0.6)
    
    ax2.set_title('Low-level Latent Space (Variations)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/latent_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return z_highs, z_lows, styles

def interpolate_styles(model, pattern1, pattern2, n_steps=10, device='cuda', save_path='results/latent_analysis'):
    model.eval()
    
    with torch.no_grad():
        pattern1 = pattern1.unsqueeze(0).to(device)
        pattern2 = pattern2.unsqueeze(0).to(device)
        
        # Encode both patterns
        mu_low1, _, mu_high1, _ = model.encode_hierarchy(pattern1)
        mu_low2, _, mu_high2, _ = model.encode_hierarchy(pattern2)
        
        # Create interpolation steps
        alphas = np.linspace(0, 1, n_steps)
        
        interpolated_patterns = []
        for alpha in alphas:
            z_high_interp = (1 - alpha) * mu_high1 + alpha * mu_high2
            z_low_interp = (1 - alpha) * mu_low1 + alpha * mu_low2
            pattern_logits = model.decode_hierarchy(z_high_interp, z_low_interp)
            pattern_probs = torch.sigmoid(pattern_logits)
            interpolated_patterns.append(pattern_probs.cpu().numpy()[0])
        fig, axes = plt.subplots(2, n_steps, figsize=(20, 6))
        
        for i, pattern in enumerate(interpolated_patterns):
            # Show pattern as heatmap
            axes[0, i].imshow(pattern, cmap='Blues', aspect='auto')
            axes[0, i].set_title(f'Step {i}')
            axes[0, i].axis('off')
            
            # Show binary pattern
            binary_pattern = (pattern > 0.5).astype(int)
            axes[1, i].imshow(binary_pattern, cmap='Blues', aspect='auto')
            axes[1, i].axis('off')
        
        axes[0, 0].set_ylabel('Probabilities')
        axes[1, 0].set_ylabel('Binary')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/style_interpolation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return interpolated_patterns

def measure_disentanglement(model, data_loader, device='cuda'):
    model.eval()
    
    # Collect latent codes by style
    style_groups = {i: {'z_high': [], 'z_low': []} for i in range(5)}
    
    with torch.no_grad():
        for patterns, style_labels, _ in data_loader:
            patterns = patterns.to(device)
            mu_low, _, mu_high, _ = model.encode_hierarchy(patterns)
            
            for i, style in enumerate(style_labels):
                style_groups[style.item()]['z_high'].append(mu_high[i].cpu().numpy())
                style_groups[style.item()]['z_low'].append(mu_low[i].cpu().numpy())
    for style in style_groups:
        style_groups[style]['z_high'] = np.array(style_groups[style]['z_high'])
        style_groups[style]['z_low'] = np.array(style_groups[style]['z_low'])
    
    # Compute within-style vs between-style variance for z_high
    within_var_high = []
    between_var_high = []
    
    all_z_high = np.concatenate([style_groups[s]['z_high'] for s in style_groups])
    overall_mean_high = np.mean(all_z_high, axis=0)
    
    for style in style_groups:
        z_high = style_groups[style]['z_high']
        if len(z_high) > 1:
            within_var_high.append(np.var(z_high, axis=0))
            style_mean = np.mean(z_high, axis=0)
            between_var_high.append((style_mean - overall_mean_high) ** 2)
    
    within_var_high = np.mean(within_var_high, axis=0)
    between_var_high = np.mean(between_var_high, axis=0)
    
    # Disentanglement score 
    disentanglement_score = np.mean(between_var_high / (within_var_high + 1e-8))
    
    return {
        'disentanglement_score': disentanglement_score,
        'within_var_high': within_var_high,
        'between_var_high': between_var_high,
        'style_groups': style_groups
    }

def controllable_generation(model, style_targets, device='cuda', save_path='results/generated_patterns'):
    model.eval()
    # Generate patterns for each style
    n_samples_per_style = 10
    generated_patterns = {}
    
    with torch.no_grad():
        for style_idx, style_name in enumerate(['rock', 'jazz', 'hiphop', 'electronic', 'latin']):
            if style_idx in style_targets:
                patterns = []
                
                # Sample from high-level prior (style-specific)
                z_high = torch.randn(n_samples_per_style, model.z_high_dim).to(device)
                
                # Sample from low-level prior (variations)
                z_low = torch.randn(n_samples_per_style, model.z_low_dim).to(device)
                
                # Decode to patterns
                pattern_logits = model.decode_hierarchy(z_high, z_low)
                pattern_probs = torch.sigmoid(pattern_logits)
                
                patterns = pattern_probs.cpu().numpy()
                generated_patterns[style_name] = patterns
                fig, axes = plt.subplots(2, 5, figsize=(15, 6))
                for i in range(min(10, n_samples_per_style)):
                    row = i // 5
                    col = i % 5
                    axes[row, col].imshow(patterns[i], cmap='Blues', aspect='auto')
                    axes[row, col].set_title(f'{style_name} {i+1}')
                    axes[row, col].axis('off')
                
                plt.tight_layout()
                plt.savefig(f'{save_path}/{style_name}_samples.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    return generated_patterns

def style_transfer_examples(model, dataset, device='cuda', save_path='results/generated_patterns/style_transfer.png'):
    model.eval()
    # pick 5 random pairs  from different styles
    import random
    pairs = []
    idxs = list(range(min(len(dataset), 200)))
    random.shuffle(idxs)
    style_to_idxs = {}
    for i in idxs:
        _, style, _ = dataset[i]
        style_to_idxs.setdefault(int(style), []).append(i)
    styles = list(style_to_idxs.keys())
    for _ in range(min(5, len(styles))):
        if len(styles) < 2:
            break
        src_style, tgt_style = random.sample(styles, 2)
        src_idx = random.choice(style_to_idxs[src_style])
        tgt_idx = random.choice(style_to_idxs[tgt_style])
        pairs.append((src_idx, tgt_idx))
    if not pairs:
        return
    # Generate transfers
    fig, axes = plt.subplots(len(pairs), 3, figsize=(9, 2.2*len(pairs)))
    if len(pairs) == 1:
        axes = np.expand_dims(axes, 0)
    with torch.no_grad():
        for r, (si, ti) in enumerate(pairs):
            src_pat, src_style, _ = dataset[si]
            tgt_pat, tgt_style, _ = dataset[ti]
            src_pat = src_pat.unsqueeze(0).to(device)
            tgt_pat = tgt_pat.unsqueeze(0).to(device)
            mu_low_s, _, mu_high_s, _ = model.encode_hierarchy(src_pat)
            mu_low_t, _, mu_high_t, _ = model.encode_hierarchy(tgt_pat)
            logits = model.decode_hierarchy(mu_high_t, mu_low_s)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
            # plot
            axes[r,0].imshow(src_pat[0].cpu().numpy(), cmap='Blues', aspect='auto')
            axes[r,0].set_title('Source (rhythm)')
            axes[r,0].axis('off')
            axes[r,1].imshow(tgt_pat[0].cpu().numpy(), cmap='Blues', aspect='auto')
            axes[r,1].set_title('Target (style)')
            axes[r,1].axis('off')
            axes[r,2].imshow(probs, cmap='Blues', aspect='auto')
            axes[r,2].set_title('Transferred')
            axes[r,2].axis('off')
    plt.tight_layout()
    Path = __import__('pathlib').Path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path

def plot_kl_trends(log_path='results/training_log.json', save_path='results/latent_analysis/kl_trends.png'):
    import json
    from pathlib import Path
    Path(Path(save_path).parent).mkdir(parents=True, exist_ok=True)
    try:
        data = json.load(open(log_path, 'r'))
    except Exception:
        return
    train = data.get('train') or []
    if not train:
        return
    kl_low = [e.get('kl_low', 0.0) for e in train]
    kl_high = [e.get('kl_high', 0.0) for e in train]
    plt.figure(figsize=(6,3))
    plt.plot(kl_low, label='KL low')
    plt.plot(kl_high, label='KL high')
    plt.xlabel('Epoch')
    plt.ylabel('KL (nats)')
    plt.title('KL trends (annealing)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def main():
    """Run all analysis functions on trained model."""
    import os
    from pathlib import Path
    from dataset import DrumPatternDataset
    from hierarchical_vae import HierarchicalDrumVAE
    from torch.utils.data import DataLoader
    import shutil, os
    Path('results/latent_analysis').mkdir(parents=True, exist_ok=True)
    Path('results/generated_patterns').mkdir(parents=True, exist_ok=True)
    
    # Load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'results/best_model.pth'
    
    if not os.path.exists(model_path):
        print(f"No trained model found at {model_path}")
        return
    
    # Load model
    model = HierarchicalDrumVAE(z_high_dim=4, z_low_dim=12)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # Load dataset
    val_dataset = DrumPatternDataset(r'C:\Users\Vivin\ee641-hw2-vivin1max\data\drums', split='val')
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print("Running latent space analysis...")
    
    # Run analysis
    z_highs, z_lows, styles = visualize_latent_hierarchy(model, val_loader, device)
    disentanglement = measure_disentanglement(model, val_loader, device)
    generated = controllable_generation(model, [0, 1, 2, 3, 4], device)
    style_transfer_examples(model, val_dataset, device, save_path='results/generated_patterns/style_transfer.png')
    plot_kl_trends('results/training_log.json', 'results/latent_analysis/kl_trends.png')

    pattern1 = val_dataset[0][0]
    pattern2 = val_dataset[10][0]
    interpolated = interpolate_styles(model, pattern1, pattern2, device=device)
    if os.path.exists('results/latent_analysis/style_interpolation.png'):
        shutil.copy('results/latent_analysis/style_interpolation.png', 'results/generated_patterns/interpolation.png')
    
    print(f"Analysis complete. Disentanglement score: {disentanglement['disentanglement_score']:.3f}")

if __name__ == '__main__':
    main()