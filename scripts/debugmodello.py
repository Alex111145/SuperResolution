import sys
import os
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Setup path
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# Prendi un file a caso dal dataset
# ADATTA "M42" AL TUO TARGET SE DIVERSO
TARGET = "M42" 
TIFF_DIR = ROOT_DATA_DIR / TARGET / "7_dataset_ready" / "pair_000000"

if not TIFF_DIR.exists():
    # Cerca la prima cartella pair disponibile
    search_dir = ROOT_DATA_DIR / TARGET / "7_dataset_ready"
    TIFF_DIR = next(search_dir.glob("pair_*"))

h_path = TIFF_DIR / "hubble.tiff"
o_path = TIFF_DIR / "observatory.tiff"

print(f"🧐 Ispeziono: {TIFF_DIR.name}")

def analyze_image(path, name):
    try:
        # Carica come fa il dataset
        img = Image.open(path)
        data = np.array(img, dtype=np.float32)
        
        print(f"\n--- {name} ---")
        print(f"   Shape: {data.shape}")
        print(f"   Min: {data.min()}")
        print(f"   Max: {data.max()}")
        print(f"   Mean: {data.mean()}")
        
        # Simulazione Dataset (divisione per 65535)
        tensor_val = data / 65535.0
        print(f"   Tensor Value (0-1): Min={tensor_val.min():.4f}, Max={tensor_val.max():.4f}")
        
        if data.max() == 0:
            print("   ❌ ATTENZIONE: L'immagine è completamente NERA!")
        elif tensor_val.max() > 1.0:
            print("   ❌ ATTENZIONE: Valori fuori scala (>1.0)!")
        else:
            print("   ✅ Valori sembrano OK.")
            
    except Exception as e:
        print(f"❌ Errore apertura {name}: {e}")

analyze_image(h_path, "HUBBLE")
analyze_image(o_path, "OBSERVATORY")