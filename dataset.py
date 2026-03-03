import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os

class WikiArtDataset(Dataset):
    """WikiArt Dataset with dummy image option"""
    
    def __init__(self, style_csv, artist_csv, genre_csv, 
                 class_files=None, img_dir=None, 
                 transform=None, use_dummy=True):
        self.img_dir = img_dir
        self.transform = transform
        self.use_dummy = use_dummy
        
        # Load CSV files
        self.style_df = pd.read_csv(style_csv, header=None)
        self.artist_df = pd.read_csv(artist_csv, header=None)
        self.genre_df = pd.read_csv(genre_csv, header=None)
        
        # Extract labels
        self.image_paths = self.style_df[0].values
        self.style_labels = self.style_df[1].values
        self.artist_labels = self.artist_df[1].values
        self.genre_labels = self.genre_df[1].values
        
        # Load class names if provided
        self.class_names = {}
        if class_files:
            for task, path in class_files.items():
                class_df = pd.read_csv(path, header=None, names=['name', 'idx'])
                self.class_names[task] = dict(zip(class_df['idx'], class_df['name']))
        
        print(f"Dataset initialized with {len(self.image_paths)} samples")
        print(f"  Style classes: {len(self.style_df[1].unique())}")
        print(f"  Artist classes: {len(self.artist_df[1].unique())}")
        print(f"  Genre classes: {len(self.genre_df[1].unique())}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.use_dummy:
            # Generate dummy image
            dummy_img = torch.randn(3, 224, 224)
            if self.transform:
                dummy_img = self.transform(dummy_img)
            img = dummy_img
        else:
            # Real image loading
            img_path = self.image_paths[idx]
            if self.img_dir:
                full_path = os.path.join(self.img_dir, img_path)
            else:
                full_path = img_path
            
            try:
                img = Image.open(full_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
            except:
                img = torch.randn(3, 224, 224)
        
        # Labels
        labels = {
            'style': torch.tensor(self.style_labels[idx], dtype=torch.long),
            'artist': torch.tensor(self.artist_labels[idx], dtype=torch.long),
            'genre': torch.tensor(self.genre_labels[idx], dtype=torch.long)
        }
        
        return img, labels
    
    def get_class_name(self, task, idx):
        """Get class name from index"""
        if task in self.class_names:
            return self.class_names[task].get(idx, f"Class_{idx}")
        return f"Class_{idx}"