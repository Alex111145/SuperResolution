"""
STEP 4: PREPARAZIONE SPLIT (TRAIN/VAL/TEST)
Prende le coppie di patch tagliate dallo Step 3 e crea gli split.
"""
import json
import random
import shutil
import os
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import sys

# ================= CONFIGURAZIONE PATH PORTABILE =================
# La radice del progetto è sempre la cartella sopra lo script (assumendo 'scripts/...')
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

NUM_WORKERS = os.cpu_count() or 16
# Percentuali Split
TRAIN_RATIO = 0.85 
VAL_RATIO = 0.10   
MIN_PAIRS = 10     

# ================= FUNZIONI DI LAVORO =================

def copy_worker(args):
    """Copia la coppia di patch in una cartella split specifica."""
    pair, dest_folder = args
    try:
        dest = dest_folder / pair['patch_id']
        dest.mkdir(parents=True, exist_ok=True)
        
        # Le patch sono chiamate hubble.fits e observatory.fits all'interno delle cartelle pair_XXX
        # Usiamo Path per la compatibilità
        shutil.copy2(pair['hubble_path'], dest / "hubble.fits")
        shutil.copy2(pair['observatory_path'], dest / "observatory.fits")
    except Exception as e:
        # print(f"Errore copia {pair['patch_id']}: {e}")
        pass

def prepare_dataset(target_dir_path):
    """Prepara il dataset creando gli split Train/Val/Test."""
    target_dir = Path(target_dir_path)
    print(f"\n📊 PREPARAZIONE SPLIT FINALE per: {target_dir.name}")
    
    # La sorgente è l'output dello Step 3
    SOURCE = target_dir / "6_patches_final"
    SPLITS = SOURCE / "splits"
    
    pairs = sorted(list(SOURCE.glob("pair_*")))
    valid_pairs = []
    
    # Verifica che le patch contengano i due file attesi
    for p in pairs:
        hubble_file = p / "hubble.fits"
        obs_file = p / "observatory.fits"
        
        if hubble_file.exists() and obs_file.exists():
            # Memorizza i path come stringhe assolute per la sicurezza
            valid_pairs.append({
                "patch_id": p.name, 
                "hubble_path": str(hubble_file.resolve()), 
                "observatory_path": str(obs_file.resolve())
            })

    print(f"   Patch Valide trovate: {len(valid_pairs)}")
    if len(valid_pairs) < MIN_PAIRS: 
        print(f"   ⚠️ ATTENZIONE: Troppe poche patch ({len(valid_pairs)}). Split saltato.")
        return

    random.shuffle(valid_pairs)
    n = len(valid_pairs)
    n_tr = int(n * TRAIN_RATIO) 
    n_val = int(n * VAL_RATIO)
    
    datasets = {
        'train': valid_pairs[:n_tr],
        'val': valid_pairs[n_tr:n_tr+n_val],
        'test': valid_pairs[n_tr+n_val:]
    }

    if SPLITS.exists(): shutil.rmtree(SPLITS)
    
    for split, data in datasets.items():
        d_path = SPLITS / split
        d_path.mkdir(parents=True)
        
        print(f"   Copia {split} ({len(data)} coppie)...")
        tasks = [(d, d_path) for d in data]
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as exe:
            list(tqdm(exe.map(copy_worker, tasks), total=len(tasks), desc=f"   Splitting {split}"))
        
        with open(d_path.parent / f'{split}_list.json', 'w') as f:
            json.dump([d['patch_id'] for d in data], f, indent=4)

    print(f"\n✅ Dataset Pronto. Totale coppie: {n}. Split salvati in {SPLITS.name}")

# ================= MAIN EXECUTION =================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Esecuzione per un singolo target passato come argomento (tipico del subprocess)
        prepare_dataset(Path(sys.argv[1]))
    else:
        # Esecuzione con menu (fallback)
        print("\n" + "📂"*35)
        print("SELEZIONE CARTELLA TARGET (Split Finale)".center(70))
        print("📂"*35)
        # Usa il ROOT_DATA_DIR definito in modo portabile
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs', '__pycache__']]
        if not subdirs:
            print("❌ Nessuna sottocartella target trovata.")
        
        print("\nCartelle target disponibili:")
        print(f"   0: ✨ Processa TUTTI i {len(subdirs)} target")
        for i, d in enumerate(subdirs): 
            print(f"   {i+1}: {d.name}")
        
        try:
            choice = int(input("\n👉 Seleziona un numero (0=Tutti): ").strip())
            if choice == 0: 
                for d in subdirs: prepare_dataset(d)
            elif 0 < choice <= len(subdirs): 
                prepare_dataset(subdirs[choice-1])
            else:
                print("❌ Scelta non valida.")
        except ValueError: 
            print("❌ Input non valido.")