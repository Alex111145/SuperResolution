#!/usr/bin/env python3
import os
import sys
import shutil
import numpy as np
from astropy.io import fits
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib
# Backend non interattivo per Linux
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ================= CONFIGURAZIONE AVANZATA =================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# IMPOSTAZIONI STRETCH
USE_LOG_STRETCH = True  
LOWER_PERCENTILE = 1.0  
UPPER_PERCENTILE = 98.0 

# MODIFICA: Salva un PNG ogni X immagini (es. 20)
DEBUG_INTERVAL = 20 

def select_target_directory():
    print("\n" + "="*35)
    print("NORMALIZZAZIONE AVANZATA (LINUX)".center(70))
    print("="*35)
    
    subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs']]
    if not subdirs: return None
    
    print("\nTarget disponibili (con cartella '6_patches_final'):")
    valid_targets = []
    # Ordinamento alfabetico per coerenza
    for i, d in enumerate(sorted(subdirs)):
        if (d / '6_patches_final').exists():
            print(f" {len(valid_targets)+1}: {d.name}")
            valid_targets.append(d)
            
    if not valid_targets:
        print("Nessun target ha le patch estratte.")
        return None

    try:
        idx = int(input("Scelta: ")) - 1
        return valid_targets[idx] if 0 <= idx < len(valid_targets) else None
    except: return None

def robust_normalize(data):
    """
    Normalizza dati astronomici per Deep Learning (0-65535 uint16).
    """
    # 1. Pulizia NaN
    data = np.nan_to_num(data, nan=0.0)
    orig_min, orig_max = np.min(data), np.max(data)

    # 2. Log Stretch
    if USE_LOG_STRETCH:
        offset = np.min(data)
        if offset < 0: data = data - offset + 1e-5
        else: data = data + 1e-5
        data = np.log1p(data)

    # 3. Clipping Percentile
    vmin = np.percentile(data, LOWER_PERCENTILE)
    vmax = np.percentile(data, UPPER_PERCENTILE)
    
    if vmax <= vmin:
        return np.zeros(data.shape, dtype=np.uint16), orig_min, orig_max
    
    # 4. Normalizzazione 0-1
    norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    
    # 5. Output 16-bit
    return (norm * 65535).astype(np.uint16), orig_min, orig_max

def save_debug_png(hr_raw, lr_raw, hr_norm, lr_norm, save_path):
    """Genera un confronto visivo Raw vs Normalized"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Log scale per visualizzazione
    axes[0,0].imshow(np.log1p(np.maximum(hr_raw, 1e-5)), cmap='inferno')
    axes[0,0].set_title(f"Hubble RAW (Log View)")
    
    axes[0,1].imshow(np.log1p(np.maximum(lr_raw, 1e-5)), cmap='viridis')
    axes[0,1].set_title(f"Obs RAW (Log View)")
    
    axes[1,0].imshow(hr_norm, cmap='gray')
    axes[1,0].set_title(f"Hubble INPUT AI (16-bit)")
    
    axes[1,1].imshow(lr_norm, cmap='gray')
    axes[1,1].set_title(f"Obs INPUT AI (16-bit)")
    
    for ax in axes.flat: ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

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

    pairs = sorted(list(input_dir.glob("pair_*")))
    print(f"\nNormalizzazione LOG/STRETCH su {len(pairs)} coppie...")
    print(f"   Input:  {input_dir}")
    print(f"   Output: {output_dir}")
    print(f"   Debug:  {debug_dir}")

    count = 0
    for pair_folder in tqdm(pairs, ncols=100):
        try:
            p_id = pair_folder.name
            
            # File creati dallo step precedente (lowercase su Linux)
            f_hubble = pair_folder / "hubble.fits"
            f_obs = pair_folder / "observatory.fits"
            
            if not f_hubble.exists() or not f_obs.exists(): continue

            # 1. Caricamento
            with fits.open(f_hubble) as h: d_hr = h[0].data
            with fits.open(f_obs) as o:    d_lr = o[0].data

            # 2. Normalizzazione
            img_hr_u16, h_min, h_max = robust_normalize(d_hr)
            img_lr_u16, o_min, o_max = robust_normalize(d_lr)

            # 3. Salvataggio TIFF
            pair_out = output_dir / p_id
            pair_out.mkdir(parents=True, exist_ok=True)
            
            Image.fromarray(img_hr_u16, mode='I;16').save(pair_out / "hubble.tiff")
            Image.fromarray(img_lr_u16, mode='I;16').save(pair_out / "observatory.tiff")
            
            # 4. Debug PNG (MODIFICATO: Ogni 20 foto, infinito)
            if count % DEBUG_INTERVAL == 0:
                save_debug_png(d_hr, d_lr, img_hr_u16, img_lr_u16, debug_dir / f"debug_{p_id}.png")

            count += 1
            
        except Exception as e:
            tqdm.write(f"Errore su {pair_folder.name}: {e}")

    print(f"\nDATASET COMPLETATO!")
    print(f"   Coppie salvate: {count}")
    print(f" VAI NELLA CARTELLA '{debug_dir.name}' E CONTROLLA SE VEDI LE NEBULOSE!")

if __name__ == "__main__":
    main()