"""
STEP 4: PREPARAZIONE SPLIT FISICI (TRAIN/VAL/TEST)
Questo script prende le coppie di patch e le copia fisicamente
nelle cartelle splits/train, splits/val, splits/test.
NON GENERA FILE JSON.
"""
import random
import shutil
import os
import sys
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# ================= CONFIGURAZIONE PATH =================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

NUM_WORKERS = os.cpu_count() or 8
# Percentuali Split
TRAIN_RATIO = 0.85 
VAL_RATIO = 0.10 
# Il resto (0.05) va in TEST
MIN_PAIRS = 10     

# ================= FUNZIONI DI LAVORO =================

def copy_worker(args):
    """
    Copia la coppia di patch nella cartella di destinazione (es. splits/train/pair_001).
    """
    pair_source_path, dest_root_folder = args
    try:
        # Crea la cartella specifica per la patch (es. splits/train/pair_123)
        dest_patch_folder = dest_root_folder / pair_source_path.name
        dest_patch_folder.mkdir(parents=True, exist_ok=True)
        
        # Copia i file fits
        shutil.copy2(pair_source_path / "hubble.fits", dest_patch_folder / "hubble.fits")
        shutil.copy2(pair_source_path / "observatory.fits", dest_patch_folder / "observatory.fits")
    except Exception as e:
        # Ignora errori su file corrotti per non bloccare tutto
        pass

def prepare_dataset(target_dir_path):
    """Organizza le cartelle Train/Val/Test senza usare JSON."""
    target_dir = Path(target_dir_path)
    print(f"\n📊 ORGANIZZAZIONE CARTELLE SPLIT per: {target_dir.name}")
    
    # Cartelle Input e Output
    SOURCE = target_dir / "6_patches_final"
    SPLITS = SOURCE / "splits" # Ora i dati vanno dentro 6_patches_final/splits
    
    # 1. Trova tutte le cartelle "pair_*"
    if not SOURCE.exists():
        print(f"❌ Errore: Cartella sorgente non trovata: {SOURCE}")
        return

    pairs = sorted(list(SOURCE.glob("pair_*")))
    valid_pairs = []
    
    # 2. Validazione rapida
    print("   🔍 Scansione patch valide...")
    for p in pairs:
        if (p / "hubble.fits").exists() and (p / "observatory.fits").exists():
            valid_pairs.append(p) # Salviamo direttamente il Path object

    print(f"   Patch Valide trovate: {len(valid_pairs)}")
    if len(valid_pairs) < MIN_PAIRS: 
        print(f"   ⚠️ ATTENZIONE: Troppe poche patch ({len(valid_pairs)}). Operazione annullata.")
        return

    # 3. Mescolamento e Calcolo Indici
    random.shuffle(valid_pairs)
    n = len(valid_pairs)
    n_tr = int(n * TRAIN_RATIO) 
    n_val = int(n * VAL_RATIO)
    
    # Dizionario che mappa il nome dello split alla lista di path sorgente
    split_groups = {
        'train': valid_pairs[:n_tr],
        'val': valid_pairs[n_tr:n_tr+n_val],
        'test': valid_pairs[n_tr+n_val:]
    }

    # 4. Pulizia Vecchi Split (Reset Totale)
    if SPLITS.exists(): 
        print("   🧹 Pulizia vecchi split...")
        shutil.rmtree(SPLITS)
    
    # 5. Esecuzione Copia
    for split_name, patch_list in split_groups.items():
        dest_path = SPLITS / split_name # es. .../splits/train
        dest_path.mkdir(parents=True, exist_ok=True)
        
        if not patch_list:
            continue
            
        print(f"   📂 Creazione {split_name.upper()} ({len(patch_list)} coppie)...")
        
        # Prepara i task per il ThreadPool: (path_sorgente, cartella_destinazione_generale)
        tasks = [(p, dest_path) for p in patch_list]
        
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as exe:
            list(tqdm(exe.map(copy_worker, tasks), total=len(tasks), desc=f"   Copia in {split_name}", unit="pair"))

    print(f"\n✅ Dataset Organizzato in Cartelle Fisiche in: {SPLITS}")
    print(f"   ├── train/ ({len(split_groups['train'])})")
    print(f"   ├── val/   ({len(split_groups['val'])})")
    print(f"   └── test/  ({len(split_groups['test'])})")

# ================= MAIN EXECUTION =================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Esecuzione diretta
        prepare_dataset(Path(sys.argv[1]))
    else:
        # Menu interattivo
        print("\n" + "📂"*35)
        print("SELEZIONE CARTELLA TARGET (Creazione Cartelle Split)".center(70))
        print("📂"*35)
        
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs', '__pycache__']]
        
        if not subdirs:
            print("❌ Nessuna sottocartella target trovata in data/.")
        else:
            print("\nTarget disponibili:")
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