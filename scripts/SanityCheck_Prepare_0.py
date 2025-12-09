import json
import sys
from pathlib import Path

# --- CONFIGURAZIONE PATH ---
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

def prepare_sanity_dataset(target_dir_path):
    """
    Prepara il dataset ESCLUSIVAMENTE per il Sanity Check (Overfitting).
    Prende solo la prima immagine e la duplica per Train/Val/Test.
    """
    target_dir = Path(target_dir_path)
    print(f"\n🧪 PREPARAZIONE DATASET SANITY CHECK: {target_dir.name}")
    
    # Cerca la cartella sorgente
    SOURCE = target_dir / "7_dataset_ready_LOG"
    if not SOURCE.exists():
        SOURCE = target_dir / "7_dataset_ready"

    # --- MODIFICA: Cartella dedicata per evitare conflitti con il training vero ---
    SPLITS = target_dir / "8_sanity_split" / "splits_json"
    
    if not SOURCE.exists():
        print(f"❌ Nessuna directory dati sorgente trovata in {target_dir.name}.")
        return

    pair_folders = sorted([d for d in SOURCE.glob("pair_*") if d.is_dir()])
    
    if not pair_folders:
        print(f"❌ Nessuna coppia trovata in {SOURCE.name}.")
        return

    # --- MODIFICA RADICALE PER SANITY CHECK ---
    selected_pair = pair_folders[0] 
    
    h_file = selected_pair / "hubble.tiff"
    o_file = selected_pair / "observatory.tiff"
    
    if not (h_file.exists() and o_file.exists()):
        print(f"❌ La coppia ({selected_pair.name}) è incompleta.")
        return

    single_entry = [{
        "patch_id": selected_pair.name, 
        "hubble_path": str(h_file.resolve()), 
        "ground_path": str(o_file.resolve())
    }]

    print(f"   ⚠️ MODE: SANITY CHECK (OVERFITTING)")
    print(f"   🎯 Immagine Singola: **{selected_pair.name}**")

    SPLITS.mkdir(parents=True, exist_ok=True)
    
    with open(SPLITS / 'train.json', 'w') as f: json.dump(single_entry, f, indent=4)
    with open(SPLITS / 'val.json', 'w') as f: json.dump(single_entry, f, indent=4)
    with open(SPLITS / 'test.json', 'w') as f: json.dump(single_entry, f, indent=4)

    print(f"✅ Dataset Sanity Check generato in {SPLITS.resolve()}.")
    print("-" * 40)

def select_target_from_menu():
    print("\n--- 🎯 SELEZIONE DATI PER SANITY CHECK ---")
    available_targets = []
    for d in sorted(ROOT_DATA_DIR.iterdir()):
        if d.is_dir():
            has_ready_log = (d / '7_dataset_ready_LOG').exists()
            has_ready = (d / '7_dataset_ready').exists()
            if has_ready_log or has_ready:
                available_targets.append(d)

    if not available_targets:
        print(f"❌ Nessun dato trovato in {ROOT_DATA_DIR}.")
        sys.exit(1)

    for i, target in enumerate(available_targets):
        print(f"  [{i+1}] {target.name}")
    print("  [0] Esci")

    while True:
        try:
            choice = input("Scelta: ").strip()
            if choice == '0': sys.exit(0)
            choice_index = int(choice) - 1
            if 0 <= choice_index < len(available_targets):
                return available_targets[choice_index]
        except ValueError: pass

if __name__ == "__main__":
    if len(sys.argv) > 1:
        prepare_sanity_dataset(Path(sys.argv[1]))
    else:
        prepare_sanity_dataset(select_target_from_menu())