"""
STEP 5: PREPARAZIONE SPLIT (TRAIN/VAL/TEST)
"""
import json
import random
import shutil
import os
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = Path("/root/SuperResolution")
if not PROJECT_ROOT.exists(): PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

NUM_WORKERS = os.cpu_count() or 16

def copy_worker(args):
    pair, dest_folder = args
    try:
        dest = dest_folder / pair['patch_id']
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pair['ground_path'], dest / "observatory.fits")
        shutil.copy2(pair['hubble_path'], dest / "hubble.fits")
    except: pass

def prepare_dataset(target_dir):
    print(f"\n📊 PREPARAZIONE SPLIT: {target_dir.name}")
    SOURCE = target_dir / "6_patches_aligned"
    SPLITS = SOURCE / "splits"
    
    pairs = sorted(list(SOURCE.glob("pair_*")))
    valid_pairs = []
    
    # Check veloce esistenza
    for p in pairs:
        if (p/"observatory.fits").exists() and (p/"hubble.fits").exists():
            valid_pairs.append({
                "patch_id": p.name, 
                "ground_path": str(p/"observatory.fits"), 
                "hubble_path": str(p/"hubble.fits")
            })

    print(f"   Valide: {len(valid_pairs)}")
    if len(valid_pairs) < 10: return

    random.shuffle(valid_pairs)
    n = len(valid_pairs)
    n_tr = int(n * 0.85) # 85% Train
    n_val = int(n * 0.10) # 10% Val
    
    datasets = {
        'train': valid_pairs[:n_tr],
        'val': valid_pairs[n_tr:n_tr+n_val],
        'test': valid_pairs[n_tr+n_val:]
    }

    for split, data in datasets.items():
        d_path = SPLITS / split
        if d_path.exists(): shutil.rmtree(d_path)
        d_path.mkdir(parents=True)
        
        print(f"   Copia {split} ({len(data)})...")
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as exe:
            exe.map(copy_worker, [(d, d_path) for d in data])

    print("✅ Dataset Pronto.")

if __name__ == "__main__":
    subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs']]
    print("Seleziona Target (0=Tutti):")
    for i, d in enumerate(subdirs): print(f"{i+1}: {d.name}")
    try:
        sel = int(input(">> "))
        if sel == 0: [prepare_dataset(d) for d in subdirs]
        elif 0 < sel <= len(subdirs): prepare_dataset(subdirs[sel-1])
    except: pass