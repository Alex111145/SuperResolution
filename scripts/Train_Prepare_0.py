import json
import sys
import random
from pathlib import Path

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

def prepare_full_dataset(target_dir_path):
    target_dir = Path(target_dir_path)
    print(f"\n🌍 PREPARAZIONE DATASET COMPLETO (TRAIN): {target_dir.name}")
    
    SOURCE = target_dir / "7_dataset_ready_LOG"
    if not SOURCE.exists(): SOURCE = target_dir / "7_dataset_ready"
    SPLITS = target_dir / "8_dataset_split" / "splits_json"
    
    if not SOURCE.exists():
        print(f"❌ Sorgente dati non trovata.")
        return

    # 1. Trova TUTTE le coppie
    pair_folders = sorted([d for d in SOURCE.glob("pair_*") if d.is_dir()])
    
    if not pair_folders:
        print(f"❌ Nessuna coppia trovata.")
        return
    
    print(f"   📊 Trovate {len(pair_folders)} patch totali.")

    # 2. Costruisci entries
    dataset_entries = []
    for pair in pair_folders:
        h_file = pair / "hubble.tiff"
        o_file = pair / "observatory.tiff"
        if h_file.exists() and o_file.exists():
            dataset_entries.append({
                "patch_id": pair.name,
                "hubble_path": str(h_file.resolve()),
                "ground_path": str(o_file.resolve())
            })
    
    # 3. Shuffle e Split (80% Train, 10% Val, 10% Test)
    random.shuffle(dataset_entries)
    
    total = len(dataset_entries)
    n_train = int(total * 0.8)
    n_val = int(total * 0.1)
    
    train_data = dataset_entries[:n_train]
    val_data = dataset_entries[n_train:n_train+n_val]
    test_data = dataset_entries[n_train+n_val:]
    
    print(f"   ✂️  Split: Train={len(train_data)} | Val={len(val_data)} | Test={len(test_data)}")

    SPLITS.mkdir(parents=True, exist_ok=True)
    
    with open(SPLITS / 'train.json', 'w') as f: json.dump(train_data, f, indent=4)
    with open(SPLITS / 'val.json', 'w') as f: json.dump(val_data, f, indent=4)
    with open(SPLITS / 'test.json', 'w') as f: json.dump(test_data, f, indent=4)

    print(f"✅ JSON Generati in {SPLITS.resolve()}")
    print("-" * 40)

def select_target_from_menu():
    print("\n--- 🌍 SELEZIONE DATASET PER TRAINING ---")
    available_targets = []
    for d in sorted(ROOT_DATA_DIR.iterdir()):
        if d.is_dir():
            if (d / '7_dataset_ready_LOG').exists() or (d / '7_dataset_ready').exists():
                available_targets.append(d)

    if not available_targets:
        sys.exit("❌ Nessun dato trovato.")

    for i, target in enumerate(available_targets):
        print(f"  [{i+1}] {target.name}")
    
    try:
        idx = int(input("Scelta: ")) - 1
        return available_targets[idx]
    except: sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        prepare_full_dataset(Path(sys.argv[1]))
    else:
        prepare_full_dataset(select_target_from_menu())