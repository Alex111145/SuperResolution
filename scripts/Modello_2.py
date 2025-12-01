"""
STEP 2: PREPARAZIONE SPLIT (TRAIN/VAL/TEST)
Genera i file JSON necessari per il training.
"""
import json
import random
import os
import sys
from pathlib import Path
from tqdm import tqdm

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# Parametri Split
TRAIN_RATIO = 0.90  # Aumentiamo il train set dato che abbiamo una sola GPU potente
VAL_RATIO = 0.10   

def prepare_dataset(target_dir_path):
    target_dir = Path(target_dir_path)
    print(f"\n📊 PREPARAZIONE SPLIT per: {target_dir.name}")
    
    # Cartella contenente i dati normalizzati (TIFF 16-bit)
    SOURCE = target_dir / "7_dataset_ready_LOG"
    SPLITS = target_dir / "8_dataset_split" / "splits_json"
    
    if not SOURCE.exists():
        # Fallback alla vecchia cartella se la LOG non esiste
        SOURCE = target_dir / "7_dataset_ready"
        if not SOURCE.exists():
            print("❌ Cartella dati (7_dataset_ready_LOG) non trovata. Esegui Dataset_step4.")
            return

    # Scansione coppie
    pair_folders = sorted(list(SOURCE.glob("pair_*")))
    valid_pairs = []
    
    print("   🔍 Scansione dataset...")
    for p in tqdm(pair_folders, ncols=100):
        h_file = p / "hubble.tiff"
        o_file = p / "observatory.tiff"
        
        if h_file.exists() and o_file.exists():
            valid_pairs.append({
                "patch_id": p.name, 
                "hubble_path": str(h_file.resolve()), 
                "ground_path": str(o_file.resolve())
            })

    print(f"   ✅ Patch Valide Trovate: {len(valid_pairs)}")
    if len(valid_pairs) < 10: 
        print("❌ Troppe poche patch per il training.")
        return

    # Shuffle deterministico
    random.seed(42)
    random.shuffle(valid_pairs)
    
    n = len(valid_pairs)
    n_tr = int(n * TRAIN_RATIO)
    
    train_data = valid_pairs[:n_tr]
    val_data = valid_pairs[n_tr:]
    
    SPLITS.mkdir(parents=True, exist_ok=True)
    
    # Salvataggio JSON
    with open(SPLITS / 'train.json', 'w') as f: json.dump(train_data, f, indent=4)
    with open(SPLITS / 'val.json', 'w') as f: json.dump(val_data, f, indent=4)
    
    # Creiamo un test set esplicito (copia di val o subset)
    with open(SPLITS / 'test.json', 'w') as f: json.dump(val_data, f, indent=4)

    print(f"✅ JSON Creati in: {SPLITS}")
    print(f"   Train: {len(train_data)} | Val/Test: {len(val_data)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        prepare_dataset(Path(sys.argv[1]))
    else:
        # Cerca automaticamente
        print("Ricerca targets...")
        subs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir()]
        found = False
        for d in subs: 
            if (d/'7_dataset_ready_LOG').exists() or (d/'7_dataset_ready').exists():
                prepare_dataset(d)
                found = True
        if not found:
            print("❌ Nessun dataset pronto trovato in 'data/'.")