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
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import random

# ================= CONFIGURAZIONE =================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# Parametri Normalizzazione
USE_LOG_STRETCH = True
# Percentile per il calcolo dei massimi globali (evita pixel caldi/raggi cosmici)
GLOBAL_PERCENTILE = 99.95 

def select_target_directory():
    print("\n" + "="*50)
    print("NORMALIZZAZIONE GLOBALE (STATISTICHE FISSE)".center(50))
    print("="*50)
    
    subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs']]
    if not subdirs: return None
    
    print("\nTarget disponibili:")
    valid_targets = []
    for i, d in enumerate(sorted(subdirs)):
        if (d / '6_patches_final').exists():
            print(f" {len(valid_targets)+1}: {d.name}")
            valid_targets.append(d)
            
    if not valid_targets:
        print("❌ Nessuna cartella '6_patches_final' trovata.")
        return None

    try:
        idx = int(input("Scelta: ")) - 1
        return valid_targets[idx] if 0 <= idx < len(valid_targets) else None
    except: return None

def get_stats_from_file(path):
    """Legge un FITS e ritorna min e valore al percentile target."""
    try:
        with fits.open(path) as h:
            d = np.nan_to_num(h[0].data, nan=0.0)
            if USE_LOG_STRETCH:
                offset = np.min(d)
                if offset < 0: d = d - offset + 1e-5
                else: d = d + 1e-5
                d = np.log1p(d)
            
            # Ritorna il valore al 99.95% per evitare outlier estremi
            return np.percentile(d, GLOBAL_PERCENTILE)
    except:
        return None

def apply_normalization(path_in, v_max):
    """Applica normalizzazione usando un V_MAX fisso globale."""
    try:
        with fits.open(path_in) as h:
            d = np.nan_to_num(h[0].data, nan=0.0)
            
        if USE_LOG_STRETCH:
            offset = np.min(d)
            if offset < 0: d = d - offset + 1e-5
            else: d = d + 1e-5
            d = np.log1p(d)

        # Normalizzazione Globale: usiamo 0 come min e v_max come max
        # (I valori negativi o > v_max vengono clippati)
        norm = np.clip(d / v_max, 0, 1)
        
        return (norm * 65535).astype(np.uint16)
    except:
        return None

def save_debug_png(hr_u16, lr_u16, save_path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(hr_u16, cmap='gray')
    ax[0].set_title("Hubble (Target)")
    ax[0].axis('off')
    
    ax[1].imshow(lr_u16, cmap='gray')
    ax[1].set_title("Observatory (Input)")
    ax[1].axis('off')
    
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

    all_pairs = sorted(list(input_dir.glob("pair_*")))
    print(f"\n📂 Trovate {len(all_pairs)} coppie.")

    # --- FASE 1: SCANSIONE STATISTICHE GLOBALI ---
    print("\n📊 FASE 1: Calcolo Statistiche Globali (Sampling)...")
    # Campioniamo fino a 500 immagini per velocità (o tutte se sono poche)
    sample_size = min(500, len(all_pairs))
    sample_pairs = random.sample(all_pairs, sample_size)
    
    vals_h = []
    vals_o = []

    for p in tqdm(sample_pairs, desc="   Sampling"):
        v_h = get_stats_from_file(p / "hubble.fits")
        v_o = get_stats_from_file(p / "observatory.fits")
        if v_h is not None: vals_h.append(v_h)
        if v_o is not None: vals_o.append(v_o)

    # Calcoliamo il MAX GLOBALE (usiamo il 95esimo percentile dei massimi locali per robustezza)
    global_max_h = np.percentile(vals_h, 95) if vals_h else 1.0
    global_max_o = np.percentile(vals_o, 95) if vals_o else 1.0
    
    print(f"\n📈 STATISTICHE CALCOLATE:")
    print(f"   Hubble Max Level:      {global_max_h:.4f}")
    print(f"   Observatory Max Level: {global_max_o:.4f}")
    print("   (Questi valori saranno usati per normalizzare TUTTE le immagini)")

    # --- FASE 2: APPLICAZIONE ---
    print("\n💾 FASE 2: Generazione Dataset Normalizzato...")
    count = 0
    
    for p in tqdm(all_pairs, ncols=100):
        try:
            p_id = p.name
            img_h = apply_normalization(p / "hubble.fits", global_max_h)
            img_o = apply_normalization(p / "observatory.fits", global_max_o)
            
            if img_h is None or img_o is None: continue

            # Salvataggio
            pair_out = output_dir / p_id
            pair_out.mkdir(parents=True, exist_ok=True)
            
            Image.fromarray(img_h, mode='I;16').save(pair_out / "hubble.tiff")
            Image.fromarray(img_o, mode='I;16').save(pair_out / "observatory.tiff")
            
            # Debug ogni 50 immagini
            if count % 50 == 0:
                save_debug_png(img_h, img_o, debug_dir / f"check_{p_id}.png")
            
            count += 1
        except Exception as e:
            pass

    print(f"\n✅ DATASET COMPLETATO: {count} coppie.")
    print(f"   Output: {output_dir}")
    print("⚠️  IMPORTANTE: Esegui ora 'python scripts/Modello_2.py' per aggiornare i JSON!")

if __name__ == "__main__":
    main()