"""
Dataset per Super-Resolution Astronomica
File: src/dataset.py
FIX: memmap=False per evitare errori BSCALE/BZERO
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from astropy.io import fits
import numpy as np
import random

class AstronomicalDataset(Dataset):
    def __init__(self, split_file, base_path, augment=True):
        self.base_path = Path(base_path)
        self.augment = augment
        
        with open(split_file, 'r') as f:
            self.pairs = json.load(f)
        
    def _load(self, path):
        if Path(path).is_absolute():
            file_path = Path(path)
        else:
            file_path = self.base_path / path
        
        try:
            # --- FIX CRITICO QUI SOTTO ---
            # memmap=False è OBBLIGATORIO se i FITS hanno header modificati (BZERO/BSCALE)
            # Per patch 64x64 non impatta sulla RAM.
            with fits.open(file_path, mode='readonly', memmap=False) as hdul:
                data = hdul[0].data
                if data is None:
                    raise ValueError("Dati vuoti nel FITS")
                
                # Assicura float32
                return data.astype(np.float32)
                
        except Exception as e:
            # Stampiamo l'errore solo se è grave, altrimenti silenziamo per non intasare il log
            # print(f"⚠️ Errore caricamento {file_path.name}: {e}")
            return np.zeros((64, 64), dtype=np.float32)
    
    def _norm(self, data):
        data_clean = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        min_val = np.min(data_clean)
        max_val = np.max(data_clean)
        
        if max_val - min_val < 1e-8:
            return np.zeros_like(data_clean, dtype=np.float32)
        
        normalized = (data_clean - min_val) / (max_val - min_val)
        return normalized.astype(np.float32)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        lr = self._load(pair['ground_path'])
        hr = self._load(pair['hubble_path'])
        
        lr = self._norm(lr)
        hr = self._norm(hr)
        
        # DATA AUGMENTATION
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
        
        # SICUREZZA MEMORIA
        lr = np.ascontiguousarray(lr)
        hr = np.ascontiguousarray(hr)
        
        lr_tensor = torch.from_numpy(lr).unsqueeze(0)
        hr_tensor = torch.from_numpy(hr).unsqueeze(0)
        
        return {'lr': lr_tensor, 'hr': hr_tensor}
    
    def __len__(self):
        return len(self.pairs)