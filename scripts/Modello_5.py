import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import shutil # NUOVO: Import per la creazione degli archivi ZIP

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

CURRENT_SCRIPT = Path(__file__).resolve()
# Assuming the script is in a subdirectory (e.g., 'scripts') of the project root
PROJECT_ROOT = CURRENT_SCRIPT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Nuove costanti per la gestione dei percorsi
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

try:
    from src.architecture import HybridSuperResolutionModel
    from src.dataset import AstronomicalDataset
    from src.metrics import Metrics
except ImportError:
    # Aggiungo un messaggio di errore più chiaro se l'import fallisce
    sys.exit("❌ Errore Import src. Assicurati che le cartelle 'src' e 'scripts' siano nella stessa root e che i moduli esistano.")


def save_as_tiff16(tensor, path):
    """Salva un tensore PyTorch come immagine TIFF a 16 bit."""
    arr = tensor.squeeze().float().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    arr_u16 = (arr * 65535).astype(np.uint16)
    Image.fromarray(arr_u16, mode='I;16').save(path)


def get_available_targets(output_root: Path) -> list[str]:
    """Elenca le sottocartelle in 'outputs', considerate come target."""
    if not output_root.is_dir():
        print(f"⚠️ La cartella '{output_root.name}' non esiste. Creala e inserisci i target.")
        return []
    
    targets = [p.name for p in output_root.iterdir() if p.is_dir()]
    return sorted(targets)


def select_target_from_menu(targets: list[str]) -> str | None:
    """Mostra un menu e chiede all'utente di selezionare un target."""
    if not targets:
        print("❌ Nessun target (cartella) trovato nella cartella 'outputs'.")
        return None
    
    print("\n--- SELEZIONE TARGET ---")
    for i, target in enumerate(targets):
        print(f"[{i+1}] {target}")
    
    while True:
        try:
            choice = input("\nSeleziona il numero del target da testare (o premi Invio per uscire): ")
            if not choice:
                return None 
            choice_index = int(choice) - 1
            if 0 <= choice_index < len(targets):
                return targets[choice_index]
            else:
                print(f"⚠️ Selezione non valida. Inserisci un numero tra 1 e {len(targets)}.")
        except ValueError:
            print("⚠️ Input non valido. Inserisci un numero.")


def run_test(target_name: str):
    """Esegue il test (inferenza) per il target specificato e crea gli archivi ZIP."""
    
    # Percorsi dinamici basati sul target selezionato
    OUTPUT_DIR = OUTPUT_ROOT / target_name / "test_results_tiff"
    CHECKPOINT_PATH = OUTPUT_ROOT / target_name / "checkpoints" / "best_model.pth"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🧪 INFERENZA: {target_name}")
    
    if not CHECKPOINT_PATH.exists():
        print(f"❌ Modello non trovato: {CHECKPOINT_PATH}")
        print("   Assicurati che: ")
        print("   1. La cartella 'checkpoint' esista sotto 'outputs/{target_name}'.")
        print("   2. Il file si chiami esattamente 'best_model.pth' (attenzione a maiuscole/minuscole).")
        return

    (OUTPUT_DIR / "tiff_science").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "png_preview").mkdir(parents=True, exist_ok=True)

    # LOGICA PER IL NOME DELLA CARTELLA DATI
    # Estrai la parte base del nome del target (es. M1_Worker_HAT_Medium -> M1)
    if '_' in target_name:
        data_target_name = target_name.split('_')[0]
    else:
        data_target_name = target_name

    # Percorso per i file JSON degli split del dataset usando il nome dati corretto
    ROOT_DATA_DIR = PROJECT_ROOT / "data" 
    splits_dir = ROOT_DATA_DIR / data_target_name / "8_dataset_split" / "splits_json" 
    test_json = splits_dir / "test.json"
    
    if not test_json.exists():
        test_json = splits_dir / "val.json"
        if not test_json.exists():
            print(f"❌ Nessun file 'test.json' o 'val.json' trovato per {data_target_name} in {splits_dir}")
            return
        print("⚠️ Uso Validation set (val.json) dato che 'test.json' non è stato trovato.")

    test_ds = AstronomicalDataset(test_json, base_path=PROJECT_ROOT, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
    
    print("   Caricamento Modello...")
    model = HybridSuperResolutionModel(smoothing='balanced', device=device, output_size=512).to(device)
    
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print("   ✅ Pesi caricati")
    except Exception as e:
        print(f"   ❌ Errore nel caricamento dei pesi. Il file potrebbe essere corrotto o non essere un dizionario di stato valido: {e}")
        return

    model.eval()
    metrics = Metrics()
    
    print(f"   🚀 Generazione di {len(test_ds)} immagini...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            
            with torch.cuda.amp.autocast():
                sr = model(lr)
            
            metrics.update(sr.float(), hr.float())
            
            # Salva l'immagine Super Risolta (SR) in TIFF 16-bit
            save_as_tiff16(sr, OUTPUT_DIR / "tiff_science" / f"sr_{i:04d}.tiff")
            
            # Genera l'immagine di preview PNG (LR upscalato, SR, HR)
            lr_up = torch.nn.functional.interpolate(lr, size=(512,512), mode='nearest')
            comp = torch.cat((lr_up, sr, hr), dim=3).clamp(0,1)
            save_image(comp, OUTPUT_DIR / "png_preview" / f"comp_{i:04d}.png")

    res = metrics.compute()
    print("\n📊 RISULTATI AGGREGATI:")
    print(f"   PSNR: {res['psnr']:.2f} dB")
    print(f"   SSIM: {res['ssim']:.4f}")
    print(f"📂 Output salvati in: {OUTPUT_DIR}")

    # --- NUOVA LOGICA PER LA CREAZIONE DEGLI ARCHIVI ZIP ---
    print("\n📦 Creazione archivi ZIP...")
    
    # Usiamo il nome del target dati come base per il nome del file ZIP (es. M1)
    base_zip_name = data_target_name 
    # Percorso dove salvare i file ZIP (nella cartella del target specifico del modello)
    zip_root_dir = OUTPUT_ROOT / target_name

    # 1. ZIP per tiff_science (risultati scientifici)
    tiff_dir_to_zip = OUTPUT_DIR / "tiff_science"
    if tiff_dir_to_zip.exists():
        try:
            # zip_name sarà [M1]_tiff_science
            zip_name_path = zip_root_dir / f"{base_zip_name}_tiff_science"
            shutil.make_archive(
                base_name=str(zip_name_path), 
                format='zip', 
                root_dir=OUTPUT_DIR.parent, # Directory da cui partire (outputs/M1_Worker_HAT_Medium/)
                base_dir=Path("test_results_tiff") / "tiff_science" # Sottodirectory da zippare
            )
            print(f"   ✅ Archivio ZIP creato per TIFFs: {zip_name_path.name}.zip")
        except Exception as e:
            print(f"   ❌ Errore ZIP TIFFs: {e}")

    # 2. ZIP per png_preview (anteprime)
    png_dir_to_zip = OUTPUT_DIR / "png_preview"
    if png_dir_to_zip.exists():
        try:
            # zip_name sarà [M1]_png_preview
            zip_name_path = zip_root_dir / f"{base_zip_name}_png_preview"
            shutil.make_archive(
                base_name=str(zip_name_path), 
                format='zip', 
                root_dir=OUTPUT_DIR.parent, 
                base_dir=Path("test_results_tiff") / "png_preview"
            )
            print(f"   ✅ Archivio ZIP creato per PNGs: {zip_name_path.name}.zip")
        except Exception as e:
            print(f"   ❌ Errore ZIP PNGs: {e}")


if __name__ == "__main__":
    targets = get_available_targets(OUTPUT_ROOT)
    selected_target = select_target_from_menu(targets)
    
    if selected_target:
        run_test(selected_target)
    else:
        print("\n🚫 Operazione annullata. Nessun test eseguito.")
