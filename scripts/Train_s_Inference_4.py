import sys
import torch
import numpy as np
import shutil
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from typing import List, Optional

# ================= CONFIGURAZIONE SISTEMA E PATH =================
CURRENT_SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT.parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# Aggiungiamo la root al sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"🔍 PATH INFO:")
print(f"   Project Root: {PROJECT_ROOT}")

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
    print(f"\n❌ ERRORE IMPORTAZIONE: {e}")
    sys.exit(1)


# ================= FUNZIONI DI UTILITÀ =================
torch.backends.cudnn.benchmark = True

def save_as_tiff16(tensor, path):
    """Salva un tensore PyTorch come immagine TIFF a 16 bit."""
    arr = tensor.squeeze().float().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    arr_u16 = (arr * 65535).astype(np.uint16)
    Image.fromarray(arr_u16, mode='I;16').save(path)

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
    
    print("\n--- SELEZIONE MODELLO DA TESTARE ---")
    for i, target in enumerate(targets):
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

# ================= LOGICA DI INFERENZA =================

def run_test(target_model_folder: str):
    # 1. SCELTA DEVICE (SOLUZIONE AL TUO PROBLEMA)
    print("\n⚠️  ATTENZIONE: Se il training è attivo, la GPU è occupata.")
    use_cpu = input("   Vuoi usare la CPU per evitare crash? (s/n) [default: s]: ").strip().lower()
    
    if use_cpu == 'n':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("   👉 Tento di usare la GPU (Potrebbe crashare per OOM)...")
    else:
        device = torch.device('cpu')
        print("   👉 Uso la CPU (Lento ma sicuro durante il training).")

    # 2. PERCORSI
    OUTPUT_DIR = OUTPUT_ROOT / target_model_folder / "test_results_tiff"
    CHECKPOINT_DIR = OUTPUT_ROOT / target_model_folder / "checkpoints"
    
    checkpoints = list(CHECKPOINT_DIR.glob("best_train_model.pth"))
    if not checkpoints: checkpoints = list(CHECKPOINT_DIR.glob("*.pth"))
    
    if not checkpoints:
        print(f"❌ Nessun checkpoint trovato in {CHECKPOINT_DIR}")
        return
    
    CHECKPOINT_PATH = checkpoints[0]
    print(f"\n🧪 INFERENZA SU: {target_model_folder}")
    print(f"   Checkpoint: {CHECKPOINT_PATH.name}")

    # 3. IDENTIFICAZIONE DATASET
    if "_DDP" in target_model_folder:
        data_target_name = target_model_folder.replace("_DDP", "")
    else:
        data_target_name = target_model_folder
        
    ROOT_DATA_DIR = PROJECT_ROOT / "data" 
    splits_dir = ROOT_DATA_DIR / data_target_name / "8_dataset_split" / "splits_json" 
    test_json = splits_dir / "test.json"
    
    if not test_json.exists():
        test_json = splits_dir / "val.json"
        print("⚠️ 'test.json' non trovato. Uso 'val.json'.")

    # 4. PREPARAZIONE OUTPUT
    (OUTPUT_DIR / "tiff_science").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "png_preview").mkdir(parents=True, exist_ok=True)

    # 5. CARICAMENTO DATASET
    test_ds = AstronomicalDataset(test_json, base_path=PROJECT_ROOT, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    
    # 6. CARICAMENTO MODELLO
    print("   Caricamento Architettura...")
    # IMPORTANTE: smoothing='none' per corrispondere al training
    model = TrainHybridModel(smoothing='none', device=device, output_size=512).to(device)

    try:
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("   ✅ Pesi caricati.")
    except Exception as e:
        print(f"   ❌ Errore caricamento pesi: {e}")
        return

    model.eval()
    metrics = TrainMetrics()
    
    print(f"   🚀 Elaborazione {len(test_ds)} immagini...")
    
    # 7. LOOP DI INFERENZA
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            
            # Gestione AMP (Autocast) sicura
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    sr = model(lr)
            else:
                # Su CPU niente autocast CUDA
                sr = model(lr)
            
            metrics.update(sr.float(), hr.float())
            
            save_as_tiff16(sr, OUTPUT_DIR / "tiff_science" / f"sr_{i:04d}.tiff")
            
            lr_up = torch.nn.functional.interpolate(lr, size=(512,512), mode='nearest')
            comp = torch.cat((lr_up, sr, hr), dim=3).clamp(0,1)
            save_image(comp, OUTPUT_DIR / "png_preview" / f"comp_{i:04d}.png")

    # 8. RISULTATI
    res = metrics.compute()
    print("\n📊 RISULTATI MEDI:")
    print(f"   PSNR: {res['psnr']:.2f} dB")
    print(f"   SSIM: {res['ssim']:.4f}")

    # 9. ZIP
    print("\n📦 Creazione ZIP...")
    base_zip_name = data_target_name
    zip_root_dir = OUTPUT_ROOT / target_model_folder
    
    def create_zip(folder_name, suffix):
        path_to_zip = OUTPUT_DIR / folder_name
        if path_to_zip.exists():
            zip_filename = zip_root_dir / f"{base_zip_name}_{suffix}"
            shutil.make_archive(
                base_name=str(zip_filename), 
                format='zip', 
                root_dir=OUTPUT_DIR.parent, 
                base_dir=Path("test_results_tiff") / folder_name
            )
            print(f"   ✅ {suffix}: {zip_filename.name}.zip")

    create_zip("tiff_science", "results_tiff")
    create_zip("png_preview", "preview_png")
    print(f"\n📂 Finito.")


if __name__ == "__main__":
    targets = get_available_targets(OUTPUT_ROOT)
    selected_target = select_target_from_menu(targets)
    if selected_target:
        run_test(selected_target)
    else:
        print("\n🚫 Operazione annullata.")
