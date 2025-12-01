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
    Ottimizzato per caricamento veloce e robustezza.
    """
    def __init__(self, split_file, base_path, augment=True):
        self.base_path = Path(base_path)
        self.augment = augment
        
        # Caricamento lista coppie dal JSON
        with open(split_file, 'r') as f:
            self.pairs = json.load(f)
            
        print(f"📦 Dataset caricato: {len(self.pairs)} coppie da {Path(split_file).name}")

    def _load_tiff_as_tensor(self, path):
        """Carica un TIFF 16-bit e lo converte in Tensore Float [0-1]."""
        try:
            # 1. Carica immagine con PIL (gestisce nativamente i 16-bit)
            img = Image.open(path)
            
            # 2. Converti in array Numpy Float32
            arr = np.array(img, dtype=np.float32)
            
            # 3. NORMALIZZAZIONE CRUCIALE (16-bit -> 0..1)
            arr = arr / 65535.0
            
            # 4. Converti in Tensore PyTorch
            tensor = torch.from_numpy(arr)
            
            # 5. Aggiungi dimensione canale se manca: [H, W] -> [1, H, W]
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
                
            return tensor
            
        except Exception as e:
            print(f"❌ Errore caricamento {path}: {e}")
            # Ritorna un tensore nero di emergenza per non crashare il worker
            return torch.zeros(1, 128, 128)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Percorsi (gestisce sia assoluti che relativi)
        path_lr = pair['ground_path']
        path_hr = pair['hubble_path']
        
        # Se sono relativi, li collega alla root
        if not Path(path_lr).is_absolute(): path_lr = self.base_path / path_lr
        if not Path(path_hr).is_absolute(): path_hr = self.base_path / path_hr

        # Caricamento e Normalizzazione
        lr_tensor = self._load_tiff_as_tensor(path_lr)
        hr_tensor = self._load_tiff_as_tensor(path_hr)

        # --- DATA AUGMENTATION (Flip/Rotate) ---
        if self.augment:
            # Random Horizontal Flip
            if random.random() > 0.5:
                lr_tensor = torch.flip(lr_tensor, [-1])
                hr_tensor = torch.flip(hr_tensor, [-1])
                
            # Random Vertical Flip
            if random.random() > 0.5:
                lr_tensor = torch.flip(lr_tensor, [-2])
                hr_tensor = torch.flip(hr_tensor, [-2])
                
            # Random 90-degree Rotation
            k = random.randint(0, 3)
            if k > 0:
                lr_tensor = torch.rot90(lr_tensor, k, [-2, -1])
                hr_tensor = torch.rot90(hr_tensor, k, [-2, -1])
        
        # FIX: Gestione stride negativi che possono causare errori con PyTorch
        if lr_tensor.stride()[0] < 0: lr_tensor = lr_tensor.contiguous()
        if hr_tensor.stride()[0] < 0: hr_tensor = hr_tensor.contiguous()

        # Controllo NaN (Sicurezza)
        if torch.isnan(lr_tensor).any() or torch.isnan(hr_tensor).any():
            lr_tensor = torch.nan_to_num(lr_tensor)
            hr_tensor = torch.nan_to_num(hr_tensor)

        return {'lr': lr_tensor, 'hr': hr_tensor}

    def __len__(self):
        return len(self.pairs)