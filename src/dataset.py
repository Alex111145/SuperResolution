import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
import numpy as np
import random
from PIL import Image

class AstronomicalDataset(Dataset):
    def __init__(self, split_file, base_path, augment=True):
        self.base_path = Path(base_path)
        self.augment = augment
        
        with open(split_file, 'r') as f:
            self.pairs = json.load(f)
            
        print(f"📦 Dataset: {len(self.pairs)} coppie da {Path(split_file).name}")

    def _load_tiff_as_tensor(self, path):
        try:
            img = Image.open(path)
            arr = np.array(img, dtype=np.float32)
            arr = arr / 65535.0
            tensor = torch.from_numpy(arr)
            
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
                
            return tensor
            
        except Exception as e:
            print(f"❌ Errore caricamento {path}: {e}")
            return torch.zeros(1, 128, 128)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        path_lr = pair['ground_path']
        path_hr = pair['hubble_path']
        
        if not Path(path_lr).is_absolute(): path_lr = self.base_path / path_lr
        if not Path(path_hr).is_absolute(): path_hr = self.base_path / path_hr

        lr_tensor = self._load_tiff_as_tensor(path_lr)
        hr_tensor = self._load_tiff_as_tensor(path_hr)

        if self.augment:
            if random.random() > 0.5:
                lr_tensor = torch.flip(lr_tensor, [-1])
                hr_tensor = torch.flip(hr_tensor, [-1])
                
            if random.random() > 0.5:
                lr_tensor = torch.flip(lr_tensor, [-2])
                hr_tensor = torch.flip(hr_tensor, [-2])
                
            k = random.randint(0, 3)
            if k > 0:
                lr_tensor = torch.rot90(lr_tensor, k, [-2, -1])
                hr_tensor = torch.rot90(hr_tensor, k, [-2, -1])
        
        if lr_tensor.stride()[0] < 0: lr_tensor = lr_tensor.contiguous()
        if hr_tensor.stride()[0] < 0: hr_tensor = hr_tensor.contiguous()

        if torch.isnan(lr_tensor).any() or torch.isnan(hr_tensor).any():
            lr_tensor = torch.nan_to_num(lr_tensor)
            hr_tensor = torch.nan_to_num(hr_tensor)

        return {'lr': lr_tensor, 'hr': hr_tensor}

    def __len__(self):
        return len(self.pairs)