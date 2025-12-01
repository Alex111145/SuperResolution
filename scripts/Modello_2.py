"""
STEP 4: PREPARAZIONE SPLIT (TRAIN/VAL/TEST) - TIFF VERSION
Prende le coppie di patch TIFF (Step 4) e crea gli split JSON.
"""
import json
import random
import shutil
import os
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import sys

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

NUM_WORKERS = os.cpu_count() or 4
TRAIN_RATIO = 0.85 
VAL_RATIO = 0.15   
MIN_PAIRS = 10     

def prepare_dataset(target_dir_path):
    target_dir = Path(target_dir_path)
    print(f"\n📊 PREPARAZIONE SPLIT (TIFF) per: {target_dir.name}")
    
    # La sorgente è la cartella dello split manuale fatto nel Dataset Step 5
    # Struttura attesa: 8_dataset_split/train/HR/*.tiff ecc.
    # OPPURE, se usiamo lo Step 4 diretto (7_dataset_ready), adattiamo:
    
    # Caso A: Usiamo 7_dataset_ready (Tutte le coppie normalizzate)
    SOURCE = target_dir / "7_dataset_ready"
    SPLITS = target_dir / "8_dataset_split" / "splits_json" # Salviamo i json qui
    
    if not SOURCE.exists():
        print("❌ Cartella 7_dataset_ready non trovata. Esegui Dataset_step4.")
        return

    # Cerchiamo le cartelle 'pair_XXXXXX'
    pair_folders = sorted(list(SOURCE.glob("pair_*")))
    valid_pairs = []
    
    for p in pair_folders:
        # ORA CERCHIAMO TIFF
        h_file = p / "hubble.tiff"
        o_file = p / "observatory.tiff"
        
        if h_file.exists() and o_file.exists():
            valid_pairs.append({
                "patch_id": p.name, 
                "hubble_path": str(h_file.resolve()), 
                "ground_path": str(o_file.resolve()) # Nota: nel dataset.py uso 'ground_path' per LR
            })

    print(f"   Patch TIFF Valide: {len(valid_pairs)}")
    if len(valid_pairs) < MIN_PAIRS: return

    random.shuffle(valid_pairs)
    n = len(valid_pairs)
    n_tr = int(n * TRAIN_RATIO)
    
    # Creiamo Train e Val (Test opzionale, qui unito a Val o separato)
    # Per semplicità facciamo Train / Val
    train_data = valid_pairs[:n_tr]
    val_data = valid_pairs[n_tr:]
    
    SPLITS.mkdir(parents=True, exist_ok=True)
    
    # Salvataggio JSON
    with open(SPLITS / 'train.json', 'w') as f: json.dump(train_data, f, indent=4)
    with open(SPLITS / 'val.json', 'w') as f: json.dump(val_data, f, indent=4)
    # Usiamo una parte del val come test
    with open(SPLITS / 'test.json', 'w') as f: json.dump(val_data[:50], f, indent=4) # Primi 50 di val per test veloce

    print(f"✅ JSON Creati in: {SPLITS}")
    print(f"   Train: {len(train_data)} | Val: {len(val_data)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        prepare_dataset(Path(sys.argv[1]))
    else:
        print("Ricerca targets...")
        subs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and (d/'7_dataset_ready').exists()]
        for d in subs: prepare_dataset(d)