"""
STEP 5: PREPARAZIONE SPLIT (TRAIN/VAL/TEST) - TIFF VERSION
Crea le cartelle fisiche train/val/test e genera i relativi file JSON.
"""
import json
import random
import sys
import shutil
from pathlib import Path
from tqdm import tqdm

# Configurazione Path
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

TRAIN_RATIO = 0.85 
MIN_PAIRS = 10     

def prepare_dataset(target_dir_path):
    target_dir = Path(target_dir_path)
    print(f"\n📊 ANALISI TARGET: {target_dir.name}")
    
    # Cerca la cartella creata dallo Step 4 (Normalization)
    SOURCE = target_dir / "7_dataset_ready"
    OUTPUT_SPLIT_DIR = target_dir / "8_dataset_split"
    JSON_DIR = OUTPUT_SPLIT_DIR / "splits_json"
    
    if not SOURCE.exists():
        print(f"   ❌ ERRORE: Cartella '7_dataset_ready' NON trovata in {target_dir.name}.")
        if (target_dir / "6_patches_final").exists():
             print("      (Trovata '6_patches_final'. Esegui 'Dataset_step4_normalization.py' per creare i TIFF!)")
        return

    # Cerchiamo le cartelle 'pair_XXXXXX' contenenti i TIFF
    pair_folders = sorted(list(SOURCE.glob("pair_*")))
    
    if not pair_folders:
        print(f"   ❌ ERRORE: Nessuna cartella 'pair_*' trovata in {SOURCE}.")
        return

    valid_pairs = []
    print(f"   🔍 Scansione {len(pair_folders)} coppie...")

    for p in pair_folders:
        h_file = p / "hubble.tiff"
        o_file = p / "observatory.tiff"
        
        if h_file.exists() and o_file.exists():
            valid_pairs.append({
                "patch_id": p.name, 
                "hubble_src": h_file,       # Path sorgente (Path object)
                "ground_src": o_file        # Path sorgente (Path object)
            })

    print(f"   ✅ Patch TIFF Valide: {len(valid_pairs)}")
    if len(valid_pairs) < MIN_PAIRS: 
        print("   ⚠️ Troppe poche patch per il training. Split annullato.")
        return

    # Shuffle e Split
    random.shuffle(valid_pairs)
    n = len(valid_pairs)
    n_tr = int(n * TRAIN_RATIO)
    
    # Creazione liste logiche
    train_data = valid_pairs[:n_tr]
    val_data = valid_pairs[n_tr:]
    # Usiamo i primi 50 del validation come test rapido
    test_data = val_data[:50]

    # --- CREAZIONE CARTELLE FISICHE ---
    print(f"   📂 Creazione cartelle in {OUTPUT_DIR}...")
    
    # Pulisci se esiste già per evitare mix
    if OUTPUT_SPLIT_DIR.exists():
        shutil.rmtree(OUTPUT_SPLIT_DIR)
    
    OUTPUT_SPLIT_DIR.mkdir(parents=True)
    JSON_DIR.mkdir(parents=True)

    # Dizionario per ciclare sui tre set
    datasets = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

    json_paths = {}

    for split_name, data_list in datasets.items():
        # Crea cartella specifica (es. 8_dataset_split/train)
        split_dir = OUTPUT_SPLIT_DIR / split_name
        split_dir.mkdir(exist_ok=True)
        
        json_list = []
        
        print(f"      Copia {split_name} ({len(data_list)} immagini)...")
        
        for item in tqdm(data_list, desc=f"Writing {split_name}", ncols=80):
            # Crea sottocartella per la coppia (es. 8_dataset_split/train/pair_001)
            dest_pair_dir = split_dir / item['patch_id']
            dest_pair_dir.mkdir(exist_ok=True)
            
            # Destinazioni finali
            dest_h = dest_pair_dir / "hubble.tiff"
            dest_o = dest_pair_dir / "observatory.tiff"
            
            # Copia fisica
            shutil.copy2(item['hubble_src'], dest_h)
            shutil.copy2(item['ground_src'], dest_o)
            
            # Aggiungi al JSON il path RELATIVO o ASSOLUTO
            # Qui usiamo path assoluti per sicurezza nel training
            json_list.append({
                "patch_id": item['patch_id'],
                "hubble_path": str(dest_h.resolve()),
                "ground_path": str(dest_o.resolve())
            })
            
        # Salva il JSON corrispondente
        json_path = JSON_DIR / f"{split_name}.json"
        with open(json_path, 'w') as f:
            json.dump(json_list, f, indent=4)
        json_paths[split_name] = json_path

    print(f"\n   ✅ COMPLETATO!")
    print(f"      Cartelle create in: {OUTPUT_SPLIT_DIR}")
    print(f"      JSON creati in:     {JSON_DIR}")
    print(f"      Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        prepare_dataset(Path(sys.argv[1]))
    else:
        print("Ricerca targets...")
        print(f"📂 ROOT DATA DIR: {ROOT_DATA_DIR}")
        
        if not ROOT_DATA_DIR.exists():
            print("❌ La cartella 'data' non esiste nella root del progetto!")
            sys.exit(1)
            
        subs = []
        for d in ROOT_DATA_DIR.iterdir():
            if d.is_dir():
                if (d / '7_dataset_ready').exists():
                    subs.append(d)
        
        if not subs:
            print("❌ Nessun target con '7_dataset_ready' trovato.")
        else:
            for d in subs: 
                prepare_dataset(d)