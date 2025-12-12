import sys
import torch
import numpy as np
import shutil
import json
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from typing import List, Optional, Dict

# ================= CONFIGURAZIONE SISTEMA E PATH =================
CURRENT_SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT.parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# Aggiungiamo la root al sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"🔍 PATH INFO:")
print(f"   Project Root: {PROJECT_ROOT}")

# Verifica esistenza src
if not (PROJECT_ROOT / "src").exists():
    sys.exit("❌ ERRORE CRITICO: Cartella 'src' non trovata nella root.")

# ================= IMPORT MODULI PROGETTO =================
print("⏳ Importazione moduli...")

try:
    from src.architecture_train import TrainHybridModel
    from src.dataset import AstronomicalDataset
    from src.metrics_train import TrainMetrics
    print("✅ Moduli importati correttamente.")
except ImportError as e:
    # Qui potrebbero esserci problemi di path/dipendenze per RRDBNet/HAT
    print(f"\n❌ ERRORE IMPORTAZIONE MODULI PROGETTO (Controlla sys.path e dipendenze): {e}")
    sys.exit(1)


# ================= FUNZIONI DI UTILITÀ =================
torch.backends.cudnn.benchmark = True

def save_as_tiff16(tensor, path):
    """Salva un tensore PyTorch come immagine TIFF a 16 bit."""
    arr = tensor.squeeze().float().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    arr_u16 = (arr * 65535).astype(np.uint16)
    Image.fromarray(arr_u16, mode='I;16').save(path)

def get_target_list_from_model_name(model_folder_name: str) -> List[str]:
    """
    Estrae la lista dei target dal nome della cartella di output.
    Esempio: 'M33_M42_M8_DDP' -> ['M33', 'M42', 'M8']
    """
    base_name = model_folder_name.replace("_DDP", "")
    return [t.strip() for t in base_name.split('_') if t.strip()]

def get_available_targets(output_root: Path) -> List[str]:
    """Elenca le cartelle in 'outputs'."""
    if not output_root.is_dir():
        print(f"⚠️ La cartella '{output_root.name}' non esiste.")
        return []
    targets = [p.name for p in output_root.iterdir() if p.is_dir()]
    return sorted(targets)

def select_target_from_menu(targets: List[str]) -> Optional[str]:
    """Menu interattivo."""
    if not targets:
        print("❌ Nessun training trovato.")
        return None
    
    print("\n--- SELEZIONE MODELLO DA TESTARE (Cartelle in outputs/) ---")
    for i, target in enumerate(targets):
        targets_in_model = get_target_list_from_model_name(target)
        if len(targets_in_model) > 1:
            print(f"[{i+1}] {target} (AGGREGATO su {len(targets_in_model)} target)")
        else:
            print(f"[{i+1}] {target}")
    
    while True:
        try:
            choice = input("\nSeleziona il numero (o Invio per uscire): ")
            if not choice:
                return None 
            choice_index = int(choice) - 1
            if 0 <= choice_index < len(targets):
                return targets[choice_index]
            else:
                print("⚠️ Numero non valido.")
        except ValueError:
            print("⚠️ Inserisci un numero valido.")

def aggregate_test_data(target_names: List[str]) -> Path:
    """
    Aggrega i dati di test (o validazione) per tutti i target specificati
    e salva il risultato in un unico JSON temporaneo.
    """
    all_test_data: List[Dict] = []
    
    print(f"   📑 Aggregazione dati di test per {len(target_names)} target...")
    
    for target_name in target_names:
        splits_dir = ROOT_DATA_DIR / target_name / "8_dataset_split" / "splits_json"
        
        # Priorità a test.json, altrimenti usa val.json
        test_path = splits_dir / "test.json"
        if not test_path.exists():
            test_path = splits_dir / "val.json"
            if test_path.exists():
                print(f"     -> Usando 'val.json' per {target_name}.")
            else:
                print(f"     ❌ Nessun file test/val trovato per {target_name}. Salto.")
                continue

        with open(test_path, 'r') as f:
            all_test_data.extend(json.load(f))
    
    if not all_test_data:
        raise FileNotFoundError("Nessun dato di test aggregato trovato.")
        
    # Salvataggio in JSON temporaneo per il dataset
    temp_dir = OUTPUT_ROOT / "temp_test_data"
    temp_dir.mkdir(exist_ok=True)
    temp_json_path = temp_dir / f"aggregated_test_{'_'.join(target_names)}.json"
    
    with open(temp_json_path, 'w') as f:
        json.dump(all_test_data, f)
        
    print(f"   ✅ Totale coppie di test aggregate: {len(all_test_data)}")
    return temp_json_path

# ================= LOGICA DI INFERENZA =================

def run_test(target_model_folder: str):
    # 1. SCELTA DEVICE
    print("\n⚠️ 	ATTENZIONE: Se il training è attivo, la GPU è occupata.")
    use_cpu = input(" 	 Vuoi usare la CPU per evitare crash? (s/n) [default: s]: ").strip().lower()
    
    if use_cpu == 'n' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(" 	 👉 Tento di usare la GPU...")
    else:
        device = torch.device('cpu')
        print(" 	 👉 Uso la CPU (Lento ma sicuro).")

    # 2. PERCORSI
    OUTPUT_DIR = OUTPUT_ROOT / target_model_folder / "test_results_tiff"
    CHECKPOINT_DIR = OUTPUT_ROOT / target_model_folder / "checkpoints"
    
    # Identificazione della lista dei target che compongono questo modello
    data_target_names = get_target_list_from_model_name(target_model_folder)
    target_base_name = target_model_folder.replace("_DDP", "")
    
    # Cerca il checkpoint migliore
    checkpoints = list(CHECKPOINT_DIR.glob("best_train_model.pth"))
    if not checkpoints: checkpoints = list(CHECKPOINT_DIR.glob("*.pth"))
    
    if not checkpoints:
        print(f"❌ Nessun checkpoint trovato in {CHECKPOINT_DIR}")
        return
    
    CHECKPOINT_PATH = checkpoints[0]
    
    print(f"\n🧪 INFERENZA SU MODELLO: {target_model_folder}")
    print(f"   Target Addestrati: {data_target_names}")
    print(f"   Checkpoint: {CHECKPOINT_PATH.name}")

    # 3. IDENTIFICAZIONE E AGGREGAZIONE DATASET DI TEST
    try:
        temp_json_path = aggregate_test_data(data_target_names)
    except FileNotFoundError as e:
        print(f"❌ ERRORE: {e}")
        return

    # 4. PREPARAZIONE OUTPUT
    (OUTPUT_DIR / "tiff_science").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "png_preview").mkdir(parents=True, exist_ok=True)

    # 5. CARICAMENTO DATASET
    test_ds = AstronomicalDataset(temp_json_path, base_path=PROJECT_ROOT, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    
    # 6. CARICAMENTO MODELLO
    print("   Caricamento Architettura...")
    # IMPORTANTE: smoothing='none' per mantenere l'output pulito per la valutazione
    model = TrainHybridModel(smoothing='none', device=device, output_size=512).to(device)

    try:
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            # Rimuove il prefisso 'module.' aggiunto dal DDP
            name = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("   ✅ Pesi caricati.")
    except Exception as e:
        print(f"   ❌ Errore caricamento pesi: {e}")
        return

    model.eval()
    metrics = TrainMetrics()
    
    print(f"   🚀 Elaborazione {len(test_ds)} immagini aggregate...")
    
    # 7. LOOP DI INFERENZA
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            
            # Gestione Autocast sicura (solo per CUDA)
            if device.type == 'cuda' and hasattr(torch.amp, 'autocast'):
                with torch.amp.autocast('cuda'):
                    sr = model(lr)
            else:
                sr = model(lr)
            
            # Calcolo metriche
            metrics.update(sr.float(), hr.float())
            
            # Salvataggio dati scientifici (TIFF 16-bit)
            save_as_tiff16(sr, OUTPUT_DIR / "tiff_science" / f"sr_{i:04d}.tiff")
            
            # Salvataggio anteprima (PNG)
            lr_up = torch.nn.functional.interpolate(lr, size=(512,512), mode='nearest')
            comp = torch.cat((lr_up, sr, hr), dim=3).clamp(0,1) # LR | SR | HR
            save_image(comp, OUTPUT_DIR / "png_preview" / f"comp_{i:04d}.png")

    # 8. RISULTATI
    res = metrics.compute()
    print("\n📊 RISULTATI MEDI (Sul dataset aggregato):")
    print(f"   PSNR: {res['psnr']:.2f} dB")
    print(f"   SSIM: {res['ssim']:.4f}")

    # 9. PULIZIA E ZIP
    print("\n📦 Creazione ZIP e Pulizia...")
    
    # Pulisci il JSON temporaneo creato
    if temp_json_path.exists():
        temp_json_path.unlink()

    zip_root_dir = OUTPUT_ROOT / target_model_folder
    
    def create_zip(folder_name, suffix):
        path_to_zip = OUTPUT_DIR / folder_name
        if path_to_zip.exists():
            zip_filename = zip_root_dir / f"{target_base_name}_{suffix}"
            shutil.make_archive(
                base_name=str(zip_filename), 
                format='zip', 
                root_dir=OUTPUT_DIR.parent, 
                base_dir=Path("test_results_tiff") / folder_name
            )
            print(f"   ✅ {suffix}: {zip_filename.name}.zip")

    create_zip("tiff_science", "results_tiff")
    create_zip("png_preview", "preview_png")
    
    print(f"\n📂 Finito. Risultati in {OUTPUT_DIR.parent}")


if __name__ == "__main__":
    targets = get_available_targets(OUTPUT_ROOT)
    selected_target_folder = select_target_from_menu(targets)
    if selected_target_folder:
        run_test(selected_target_folder)
    else:
        print("\n🚫 Operazione annullata.")