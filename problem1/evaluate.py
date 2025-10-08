import torch
import sys
from pathlib import Path as _P
_root = _P(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

def _load_real_letter_mean(data_dir, split, letter):
    try:
        from dataset import FontDataset
        ds = FontDataset(data_dir, split=split)
        idxs = []
        letter_id = ord(letter) - 65
        imgs = []
        for i in range(len(ds)):
            img, lid = ds[i]
            if lid == letter_id:
                imgs.append(img)
        if not imgs:
            return None
        return torch.stack(imgs, dim=0).mean(0, keepdim=True)  
    except Exception:
        return None

def targeted_letter_latent(generator, device, target_img, steps=300, lr=0.05):
    """Optimize z so G(z) = target image (pixel MSE)."""
    generator.eval()
    z = torch.randn(1, generator.z_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=lr)
    target_img = target_img.to(device)
    for i in range(steps):
        optimizer.zero_grad()
        out = generator(z)
        loss = torch.mean((out - target_img)**2)
        loss.backward()
        optimizer.step()
    return z.detach()

def letter_interpolation_A_to_Z(generator, device, data_dir, save_path='results/visualizations/letter_interpolation_AZ.png'):
    """Interpolate between optimized z for mean 'A' and 'Z' real images."""
    Path(Path(save_path).parent).mkdir(parents=True, exist_ok=True)
    mean_A = _load_real_letter_mean(data_dir, 'train', 'A')
    mean_Z = _load_real_letter_mean(data_dir, 'train', 'Z')
    if mean_A is None or mean_Z is None:
        print('Could not load mean real letters A or Z; skipping A→Z interpolation.')
        return
    z_A = targeted_letter_latent(generator, device, mean_A)
    z_Z = targeted_letter_latent(generator, device, mean_Z)
    steps = 12
    imgs = []
    with torch.no_grad():
        for i in range(steps):
            alpha = i/(steps-1)
            z = (1-alpha)*z_A + alpha*z_Z
            img = generator(z)[0,0].cpu().numpy()
            imgs.append(img)
    fig, axes = plt.subplots(1, steps, figsize=(steps*1.1, 1.5))
    for i, ax in enumerate(fig.axes):
        ax.imshow(imgs[i], cmap='gray', vmin=-1, vmax=1)
        ax.axis('off')
    fig.suptitle('Optimized Latent Interpolation A → Z', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved A→Z letter interpolation to {save_path}')

def coverage_timeline_from_checkpoints(checkpoint_dir, generator_cls, device, save_path='results/visualizations/coverage_timeline.png'):
    from training_dynamics import analyze_mode_coverage
    cp_dir = Path(checkpoint_dir)
    if not cp_dir.exists():
        print(f'Checkpoint dir {checkpoint_dir} not found; skipping coverage timeline.')
        return
    checkpoints = sorted(cp_dir.glob('generator_epoch_*.pth'), key=lambda p: int(p.stem.split('_')[-1]))
    if not checkpoints:
        print('No generator checkpoints found for timeline.')
        return
    epochs = []
    coverages = []
    missing_counts = []
    for cp in checkpoints:
        epoch = int(cp.stem.split('_')[-1])
        state = torch.load(cp, map_location=device)
        G = generator_cls(z_dim=100).to(device)
        G.load_state_dict(state['generator_state_dict'])
        cov, stats = analyze_mode_coverage(G, device, n_samples=800)
        epochs.append(epoch)
        coverages.append(cov)
        missing_counts.append(len(stats['missing_letters']))
    # Plot
    Path(Path(save_path).parent).mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(epochs, coverages, 'o-', color='tab:blue', label='Coverage')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Coverage', color='tab:blue')
    ax1.set_ylim(0,1)
    ax2 = ax1.twinx()
    ax2.plot(epochs, missing_counts, 's--', color='tab:red', label='Missing Letters')
    ax2.set_ylabel('# Missing', color='tab:red')
    ax2.set_ylim(0,26)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines+lines2, labels+labels2, loc='lower left')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved coverage timeline to {save_path}')

def coverage_timeline_from_log(log_path='results/training_log.json', save_path='results/visualizations/coverage_timeline.png'):
    log_file = Path(log_path)
    if not log_file.exists():
        print(f'Training log {log_path} not found; skipping coverage timeline fallback.')
        return
    try:
        data = json.load(open(log_file, 'r'))
        coverages = data.get('mode_coverage', [])
        details = data.get('coverage_details', [])
        if not coverages:
            print('No mode_coverage entries in training log; skipping timeline fallback.')
            return
        epochs = [i*10 for i in range(len(coverages))]
        missing_counts = []
        for i in range(len(coverages)):
            try:
                missing = details[i].get('missing_letters', []) if i < len(details) else []
            except Exception:
                missing = []
            missing_counts.append(len(missing))
        # Plot
        Path(Path(save_path).parent).mkdir(parents=True, exist_ok=True)
        fig, ax1 = plt.subplots(figsize=(6,4))
        ax1.plot(epochs, coverages, 'o-', color='tab:blue', label='Coverage')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Coverage', color='tab:blue')
        ax1.set_ylim(0,1)
        ax2 = ax1.twinx()
        ax2.plot(epochs, missing_counts, 's--', color='tab:red', label='Missing Letters')
        ax2.set_ylabel('# Missing', color='tab:red')
        ax2.set_ylim(0,26)
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines+lines2, labels+labels2, loc='lower left')
        fig.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved coverage timeline (from log) to {save_path}')
    except Exception as e:
        print('Failed to build coverage timeline from log:', e)

def interpolation_experiment(generator, device, save_path='results/interpolation.png'):
    generator.eval()
    with torch.no_grad():
        # Generating two random latent codes
        z1 = torch.randn(1, generator.z_dim).to(device)
        z2 = torch.randn(1, generator.z_dim).to(device)
        steps = 10
        interpolated_images = []
        
        for i in range(steps):
            alpha = i / (steps - 1)
            z_interp = (1 - alpha) * z1 + alpha * z2
            img = generator(z_interp)
            interpolated_images.append(img.cpu().numpy()[0, 0])
        fig, axes = plt.subplots(1, steps, figsize=(20, 2))
        for i, img in enumerate(interpolated_images):
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Step {i}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"Interpolation experiment saved to {save_path}")

def style_consistency_experiment(generator, device, save_path='results/generated_samples.png'):
    generator.eval()
    with torch.no_grad():
        # Generating a grid of samples
        n_samples = 64
        z = torch.randn(n_samples, generator.z_dim).to(device)
        fake_images = generator(z)
        # Creating grid visualization
        fig, axes = plt.subplots(8, 8, figsize=(16, 16))
        for i, ax in enumerate(axes.flat):
            img = fake_images[i].cpu().numpy()[0]
            ax.imshow(img, cmap='gray')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"Generated samples saved to {save_path}")

def mode_recovery_experiment(generator, device, save_path='results/mode_analysis.png'):
    from training_dynamics import analyze_mode_coverage
    
    generator.eval()
    coverage, stats = analyze_mode_coverage(generator, device, n_samples=1000)

    with torch.no_grad():
        z = torch.randn(100, generator.z_dim).to(device)
        fake_images = generator(z)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        sample_grid = fake_images[:25].cpu().numpy()
        grid_img = np.zeros((5*28, 5*28))
        for i in range(5):
            for j in range(5):
                idx = i*5 + j
                grid_img[i*28:(i+1)*28, j*28:(j+1)*28] = sample_grid[idx, 0]
        
        ax1.imshow(grid_img, cmap='gray')
        ax1.set_title('Generated Samples')
        ax1.axis('off')
        
        # coverage score
        ax2.bar(['Mode Coverage'], [coverage], color='skyblue')
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Coverage Score')
        missing = stats['missing_letters']
        ax2.set_title(f'Mode Coverage: {coverage:.3f}\nMissing: {len(missing)}')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"Mode analysis saved to {save_path}, Coverage: {coverage:.3f}, Missing letters: {stats['missing_letters']}")
    return coverage, stats

def main():

    import os
    from pathlib import Path
    from models import Generator
    
    Path('results/visualizations').mkdir(parents=True, exist_ok=True)    
    # Loading trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = 'results/best_generator.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"No trained model found at {checkpoint_path}")
        return
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator = Generator(z_dim=100)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.to(device)
    
    print("Running evaluation experiments...")
    
    # Run experiments
    interpolation_experiment(generator, device, 'results/visualizations/interpolation.png')
    style_consistency_experiment(generator, device, 'results/visualizations/generated_samples.png')
    coverage, stats = mode_recovery_experiment(generator, device, 'results/visualizations/mode_analysis.png')
    # A→Z targeted latent interpolation 
    data_dir = checkpoint.get('config', {}).get('data_dir', None)
    if data_dir:
        letter_interpolation_A_to_Z(generator, device, data_dir, 'results/visualizations/letter_interpolation_AZ.png')
    cp_dir = checkpoint.get('config', {}).get('checkpoint_dir', 'checkpoints')
    if Path(cp_dir).exists():
        coverage_timeline_from_checkpoints(cp_dir, type(generator), device, 'results/visualizations/coverage_timeline.png')
    else:
        coverage_timeline_from_log('results/training_log.json', 'results/visualizations/coverage_timeline.png')
    # Coverage histogram (letter survival)
    try:
        from training_dynamics import analyze_mode_coverage
        cov2, final_stats = analyze_mode_coverage(generator, device, n_samples=1200)
        raw_counts = final_stats['letter_counts']
        letters = [chr(65+i) for i in range(26)]
        counts = [raw_counts.get(i, 0) for i in range(26)]
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10,3))
        ax.bar(letters, counts, color='steelblue')
        ax.set_title('Letter Survival Histogram (Generated)')
        ax.set_ylabel('Count')
        ax.set_xlabel('Letter')
        plt.tight_layout()
        fig.savefig('results/visualizations/coverage_histogram.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print('Saved coverage_histogram.png')
    except Exception as e:
        print('Could not generate coverage histogram:', e)
    print(f"Evaluation complete. Final mode coverage: {coverage:.3f} Missing: {stats['missing_letters']}")
  
if __name__ == '__main__':
    main()