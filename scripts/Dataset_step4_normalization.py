#!/usr/bin/env python3
import os
import shutil
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ================= CONFIGURAZIONE =================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# --- PARAMETRI DI CONTRASTO ---
USE_LOG_STRETCH = True 

# AUMENTA QUESTO VALORE PER NERI PIÙ PROFONDI
# 1.0 = Standard (conservativo)
# 10.0 = Nero deciso
# 20.0 = Nero molto profondo (elimina tutto il rumore, rischia di mangiare nebulosità deboli)
BLACK_CLIP_PERCENTILE = 5.0  

NUM_SAMPLES_PER_IMG = 4000
BATCH_SIZE = 32
NUM_WORKERS = 4
DEBUG_INTERVAL = 50 

# ================= DATASET PYTORCH =================
class RawFitsDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        try:
            with fits.open(path) as hdul:
                data = hdul[0].data
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                tensor = torch.from_numpy(data.astype(np.float32))
                return tensor
        except Exception:
            return torch.zeros((1, 1))

# ================= FUNZIONI STATISTICHE ROBUSTE =================

def calculate_robust_stats(file_list):
    """
    Calcola i percentili per definire il range dinamico.
    """
    print(f"   📊 Campionamento statistico su {len(file_list)} immagini...")
    
    dataset = RawFitsDataset(file_list)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    
    sampled_pixels = []
    
    for batch in tqdm(loader, desc="   Sampling", ncols=100):
        if USE_LOG_STRETCH:
            batch = torch.log1p(torch.maximum(batch, torch.tensor(0.0)))

        batch_flat = batch.view(-1)
        valid_mask = batch_flat > 1e-5 
        valid_pixels = batch_flat[valid_mask]
        
        if valid_pixels.numel() > 0:
            num_take = min(valid_pixels.numel(), NUM_SAMPLES_PER_IMG * batch.shape[0])
            indices = torch.randperm(valid_pixels.numel())[:num_take]
            sampled_pixels.append(valid_pixels[indices].numpy())
            
    if not sampled_pixels:
        print("⚠️ Nessun dato valido trovato! Uso fallback 0-1.")
        return 0.0, 1.0

    full_sample = np.concatenate(sampled_pixels)
    
    # QUI AVVIENE LA MAGIA DEL NERO
    global_min = np.percentile(full_sample, BLACK_CLIP_PERCENTILE) 
    global_max = np.percentile(full_sample, 99.99) # Il bianco rimane invariato
    
    print(f"      Min (Nero) calcolato al {BLACK_CLIP_PERCENTILE}° percentile: {global_min:.4f}")
    
    return global_min, global_max

# ================= VISUALIZZAZIONE =================

def save_debug_png(hr_raw, lr_raw, hr_norm, lr_norm, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), facecolor='#000000') # Sfondo nero per check reale
    
    # RAW (Log View)
    axes[0,0].imshow(np.log1p(np.maximum(hr_raw, 1e-5)), cmap='inferno')
    axes[0,0].set_title("Hubble RAW (Log)", color='white')
    
    axes[0,1].imshow(np.log1p(np.maximum(lr_raw, 1e-5)), cmap='viridis')
    axes[0,1].set_title("Obs RAW (Log)", color='white')
    
    # NORMALIZED
    axes[1,0].imshow(hr_norm, cmap='gray', vmin=0, vmax=65535)
    axes[1,0].set_title("Hubble AI Input (DEEP BLACK)", color='white')
    
    axes[1,1].imshow(lr_norm, cmap='gray', vmin=0, vmax=65535)
    axes[1,1].set_title("Obs AI Input (DEEP BLACK)", color='white')
    
    for ax in axes.flat: ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, facecolor='#000000')
    plt.close()

def select_target_directory():
    subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs']]
    valid = [d for d in subdirs if (d / '6_patches_final').exists()]
    
    if not valid: return None
    print("\nSELEZIONA TARGET:")
    for i, d in enumerate(sorted(valid)): print(f" {i+1}: {d.name}")
    try:
        idx = int(input("Scelta: ")) - 1
        return sorted(valid)[idx]
    except: return None

# ================= MAIN =================

def main():
    target_dir = select_target_directory()
    if not target_dir: return

    input_dir = target_dir / '6_patches_final'
    output_dir = target_dir / '7_dataset_ready_LOG'
    debug_dir = target_dir / '7_dataset_debug_png'

    if output_dir.exists(): shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    if debug_dir.exists(): shutil.rmtree(debug_dir)
    debug_dir.mkdir(parents=True)

    all_pairs = sorted(list(input_dir.glob("pair_*")))
    hubble_files = [p / "hubble.fits" for p in all_pairs if (p/"hubble.fits").exists()]
    obs_files = [p / "observatory.fits" for p in all_pairs if (p/"observatory.fits").exists()]

    if not hubble_files: return

    print(f"\n🚀 NORMALIZZAZIONE DEEP BLACK (Clip {BLACK_CLIP_PERCENTILE}%)")
    
    print("\n--- Analisi Hubble ---")
    h_min, h_max = calculate_robust_stats(hubble_files)

    print("\n--- Analisi Observatory ---")
    o_min, o_max = calculate_robust_stats(obs_files)
    
    print(f"\n--- Generazione TIFF 16-bit ({len(all_pairs)} coppie) ---")
    for i, pair_path in enumerate(tqdm(all_pairs, ncols=100)):
        try:
            with fits.open(pair_path / "hubble.fits") as h: d_h = np.nan_to_num(h[0].data)
            with fits.open(pair_path / "observatory.fits") as o: d_o = np.nan_to_num(o[0].data)

            raw_h_copy = d_h.copy()
            raw_o_copy = d_o.copy()

            if USE_LOG_STRETCH:
                d_h = np.log1p(np.maximum(d_h, 0))
                d_o = np.log1p(np.maximum(d_o, 0))

            # Normalizzazione Globale
            d_h_norm = (d_h - h_min) / (h_max - h_min + 1e-8)
            d_o_norm = (d_o - o_min) / (o_max - o_min + 1e-8)

            d_h_norm = np.clip(d_h_norm, 0, 1)
            d_o_norm = np.clip(d_o_norm, 0, 1)

            h_u16 = (d_h_norm * 65535).astype(np.uint16)
            o_u16 = (d_o_norm * 65535).astype(np.uint16)

            p_out = output_dir / pair_path.name
            p_out.mkdir(exist_ok=True)
            Image.fromarray(h_u16, mode='I;16').save(p_out / "hubble.tiff")
            Image.fromarray(o_u16, mode='I;16').save(p_out / "observatory.tiff")

            if i % DEBUG_INTERVAL == 0:
                save_debug_png(raw_h_copy, raw_o_copy, h_u16, o_u16, debug_dir / f"check_{pair_path.name}.png")

        except Exception as e:
            print(f"Errore {pair_path.name}: {e}")

    print("\n✅ Completato.")
    print(f"   Controlla '7_dataset_debug_png'. Se è TROPPO scuro (perdi dettagli),")
    print(f"   abbassa BLACK_CLIP_PERCENTILE nello script (es. da {BLACK_CLIP_PERCENTILE} a 5.0).")

if __name__ == "__main__":
    main()
