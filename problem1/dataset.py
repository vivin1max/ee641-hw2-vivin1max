"""
Dataset loader for font generation task.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import numpy as np

class FontDataset(Dataset):
    def __init__(self, data_dir, split='train'):

        self.data_dir = data_dir
        self.split = split
        
        # Load metadata from metadata.json
        metadata_path = os.path.join(data_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if split == 'train':
            self.samples = metadata['train_samples']
        else:
            self.samples = metadata['val_samples']
        self.letter_to_id = {chr(65+i): i for i in range(26)}  
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and process image
        image_path = os.path.join(self.data_dir, self.split, sample['filename'])
        image = Image.open(image_path).convert('L')  
        image = image.resize((28, 28), Image.LANCZOS)
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = image_np * 2.0 - 1.0 
        
        # Convert to tensor 
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        
        # Get letter ID
        letter_id = self.letter_to_id[sample['letter']]
        
        return image_tensor, letter_id