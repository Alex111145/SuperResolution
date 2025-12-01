"""
Dataset per Super-Resolution Astronomica (TIFF 16-bit EDITION)
Carica immagini TIFF 16-bit pre-normalizzate e le converte in Tensori [0, 1].
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
import numpy as np
import random
from PIL import Image

class AstronomicalDataset(Dataset):
    """
    Dataset per caricare coppie LR-HR da file TIFF 16-bit.
    """
    
    def __init__(self, split_file, base_path, augment=True):
        self.base_path = Path(base_path)
        self.augment = augment
        
        # Carica il JSON con le coppie
        with open(split_file, 'r') as f:
            self.pairs = json.load(f)
        
        # print(f"📦 Dataset: {len(self.pairs)} coppie da {Path(split_file).name}")
    
    def _load_tiff(self, path):
        """Carica un TIFF 16-bit e lo converte in float32 [0, 65535]."""
        if Path(path).is_absolute():
            file_path = Path(path)
        else:
            file_path = self.base_path / path
            
        try:
            # Caricamento con PIL
            img = Image.open(file_path)
            # Conversione in Numpy Array
            data = np.array(img, dtype=np.float32)
            return data
        except Exception as e:
            print(f"⚠️ Errore caricamento {file_path.name}: {e}")
            return np.zeros((64, 64), dtype=np.float32)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # 1. Carica TIFF (Valori 0 -> 65535)
        # Nota: I file sono già stati normalizzati nello step precedente
        lr = self._load_tiff(pair['ground_path'])
        hr = self._load_tiff(pair['hubble_path'])
        
        # 2. Scaling [0, 1] per la Rete Neurale
        # Essendo uint16, il massimo teorico è 65535.0
        lr = lr / 65535.0
        hr = hr / 65535.0
        
        # 3. Data Augmentation
        if self.augment:
            if random.random() < 0.5:
                lr = np.flipud(lr).copy()
                hr = np.flipud(hr).copy()
            if random.random() < 0.5:
                lr = np.fliplr(lr).copy()
                hr = np.fliplr(hr).copy()
            k = random.randint(0, 3)
            if k > 0:
                lr = np.rot90(lr, k).copy()
                hr = np.rot90(hr, k).copy()
        
        # 4. Contiguità e Tensori
        lr = np.ascontiguousarray(lr)
        hr = np.ascontiguousarray(hr)
        
        lr_tensor = torch.from_numpy(lr).unsqueeze(0) # [1, H, W]
        hr_tensor = torch.from_numpy(hr).unsqueeze(0) # [1, H, W]
        
        return {'lr': lr_tensor, 'hr': hr_tensor}
    
    def __len__(self):
        return len(self.pairs)