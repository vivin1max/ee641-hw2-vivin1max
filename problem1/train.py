"""Main GAN training script."""

import torch
import sys
from pathlib import Path as _P
root_path = _P(__file__).resolve().parent.parent
root_path = _P(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from pathlib import Path

from dataset import FontDataset
from models import Generator, Discriminator
from training_dynamics import train_gan, analyze_mode_coverage, visualize_mode_collapse
from provided.visualize import plot_alphabet_grid
from fixes import train_gan_with_fix

def main():
    # Config
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'batch_size': 64,
    'num_epochs': 100,
        'z_dim': 100,
        'learning_rate': 0.0002,
        'data_dir': r'C:\Users\Vivin\ee641-hw2-vivin1max\data\fonts',
        'checkpoint_dir': 'checkpoints',
        'results_dir': 'results',
        'experiment': 'vanilla',  # 'vanilla' or 'fixed'
        'fix_type': 'feature_matching'  # Used if experiment='fixed'
    }
    
    # Create output dirs
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['results_dir']).mkdir(parents=True, exist_ok=True)
    
    # Data
    train_dataset = FontDataset(config['data_dir'], split='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    # Models
    generator = Generator(z_dim=config['z_dim']).to(config['device'])
    discriminator = Discriminator().to(config['device'])
    
    # Visualization dir
    vis_dir = Path(config['results_dir'])/ 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Train
    if config['experiment'] == 'vanilla':
        print("Training vanilla GAN (expect mode collapse)...")
        history = train_gan(
            generator,
            discriminator,
            train_loader,
            num_epochs=config['num_epochs'],
            device=config['device'],
            checkpoint_dir=config['checkpoint_dir'],
            checkpoint_interval=10
        )
    else:
        print(f"Training GAN with {config['fix_type']} fix...")
        history = train_gan_with_fix(
            generator,
            discriminator,
            train_loader,
            num_epochs=config['num_epochs'],
            fix_type=config['fix_type'],
            checkpoint_dir=config['checkpoint_dir'],
            checkpoint_interval=10
        )

    # Alphabet grids
    for epoch_tag in [10, 30, 50, 100]:
        if config['num_epochs'] >= epoch_tag:
            fig = plot_alphabet_grid(generator, device=config['device'], z_dim=config['z_dim'], seed=epoch_tag)
            fig.savefig(vis_dir / f'alphabet_epoch_{epoch_tag}.png', dpi=150, bbox_inches='tight')
            import matplotlib.pyplot as plt
            plt.close(fig)
    
    # Save results
    with open(f"{config['results_dir']}/training_log.json", 'w') as f:
        clean_history = {}
        for k, v in history.items():
            if isinstance(v, list):
                if k == 'coverage_details':
                    clean_history[k] = v
                else:
                    clean_history[k] = [float(x) if isinstance(x, (int, float)) else x for x in v]
            else:
                clean_history[k] = v
        json.dump(clean_history, f, indent=2)
    
    # Save final model
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'config': config,
        'final_epoch': config['num_epochs']
    }, f"{config['results_dir']}/best_generator.pth")
    
    # Mode collapse plot
    visualize_mode_collapse(history, f"{config['results_dir']}/mode_collapse_analysis.png")

    print(f"Training complete. Results saved to {config['results_dir']}/")

if __name__ == '__main__':
    main()