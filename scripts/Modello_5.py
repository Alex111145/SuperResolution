import sys
import torch
import numpy as np
import argparse
import json
import os
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

# Setup Percorsi
CURRENT_SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
ROOT_DATA_DIR = PROJECT_ROOT / "data"

try:
    # Import corretto
    from src.architecture import TrainHybridModel
    from src.dataset import AstronomicalDataset
    from src.metrics import Metrics
except ImportError:
    sys.exit("❌ Errore Import src. Esegui dalla root del progetto.")

def save_as_tiff16(tensor, path):
    """Salva tensore Float [0-1] come TIFF 16-bit."""
    arr = tensor.squeeze().float().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    arr_u16 = (arr * 65535).astype(np.uint16)
    Image.fromarray(arr_u16, mode='I;16').save(path)

def create_temp_json_from_dirs(lr_dir, hr_dir):
    """Genera lista coppie scansionando le cartelle TIFF."""
    lr_files = sorted(list(lr_dir.glob("*.tiff")) + list(lr_dir.glob("*.tif")))
    pairs = []
    
    print(f"📂 Scansione TIFF...")
    print(f"   LR: {lr_dir}")
    print(f"   HR: {hr_dir}")
    
    for lr_path in lr_files:
        hr_path = hr_dir / lr_path.name
        if hr_path.exists():
            pairs.append({
                "ground_path": str(lr_path),
                "hubble_path": str(hr_path)
            })
    
    if not pairs:
        sys.exit("❌ Nessun file corrispondente trovato.")
        
    temp_json = PROJECT_ROOT / f"temp_inf_{np.random.randint(0,9999)}.json"
    with open(temp_json, 'w') as f: json.dump(pairs, f)
    return temp_json

def select_inference_target():
    """Scansiona la cartella outputs e propone i target disponibili."""
    out_dir = PROJECT_ROOT / "outputs"
    if not out_dir.exists():
        print("❌ Cartella outputs non trovata.")
        return None
    
    # Cerca cartelle che sembrano training run (finiscono con _SwinIR_FULL_TRAIN)
    # Esempio: "M42_SwinIR_FULL_TRAIN" -> Target è "M42"
    candidates = set()
    for d in out_dir.iterdir():
        if d.is_dir() and "_SwinIR_FULL_TRAIN" in d.name:
            t_name = d.name.replace("_SwinIR_FULL_TRAIN", "")
            candidates.add(t_name)
            
    sorted_targets = sorted(list(candidates))
    
    if not sorted_targets:
        print("❌ Nessun modello allenato trovato in outputs (cerco cartelle *_SwinIR_FULL_TRAIN).")
        return None

    print("\n🎯 SELEZIONA MODELLO (Trovati in outputs):")
    for i, t in enumerate(sorted_targets):
        print(f" {i+1}: {t}")
        
    try:
        raw = input("Scelta (Invio per uscire): ")
        if not raw: return None
        sel = int(raw) - 1
        if 0 <= sel < len(sorted_targets):
            return sorted_targets[sel]
    except ValueError:
        pass
        
    return None

def run_test(args):
    # Forza CPU
    device = torch.device('cpu')
    print(f"\n🧪 INFERENZA SWINIR & METRICHE (CPU MODE)")
    
    target = None
    json_path = None
    output_base = None
    checkpoint_path = None

    # 1. Determina Target e Percorsi
    if args.lr_dir and args.hr_dir:
        # Modalità Custom (Cartelle manuali)
        print("   🔹 Modalità Custom (Cartelle Manuali)")
        lr_path = Path(args.lr_dir)
        hr_path = Path(args.hr_dir)
        json_path = create_temp_json_from_dirs(lr_path, hr_path)
        output_base = PROJECT_ROOT / "outputs" / "custom_inference"
        checkpoint_path = Path(args.model_path) if args.model_path else None
    else:
        # Modalità Standard (Menu o Argomento)
        target = args.target
        if not target:
            target = select_inference_target()
        
        if not target:
            print("❌ Nessun target selezionato. Esco.")
            return

        print(f"   🔹 Target Selezionato: {target}")
        
        # Percorsi Dataset
        splits_dir = ROOT_DATA_DIR / target / "8_dataset_split" / "splits_json"
        json_path = splits_dir / "test.json"
        if not json_path.exists():
            json_path = splits_dir / "val.json"
            print("⚠️  Test set non trovato, uso Validation set.")
        
        if not json_path.exists():
            sys.exit(f"❌ File JSON non trovato: {json_path}")

        # Percorsi Output e Modello
        output_base = PROJECT_ROOT / "outputs" / target / "test_results_tiff"
        
        if args.model_path:
            checkpoint_path = Path(args.model_path)
        else:
            # Cerca il modello nella cartella di training standard
            checkpoint_path = PROJECT_ROOT / "outputs" / f"{target}_SwinIR_FULL_TRAIN" / "checkpoints" / "best_train_model.pth"

    if not checkpoint_path or not checkpoint_path.exists():
        sys.exit(f"❌ Modello non trovato: {checkpoint_path}")

    # Output Dir
    (output_base / "tiff_science").mkdir(parents=True, exist_ok=True)
    (output_base / "png_preview").mkdir(parents=True, exist_ok=True)

    # Dataset
    # Augment=False fondamentale per test/inferenza
    test_ds = AstronomicalDataset(json_path, base_path=PROJECT_ROOT, augment=False)
    # Pochi worker su CPU per evitare overhead
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    
    # Load Model
    print(f"   📂 Caricamento pesi: {checkpoint_path.name}")
    model = TrainHybridModel(output_size=512, device='cpu').to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    metrics = Metrics()
    
    print(f"   🚀 Elaborazione {len(test_ds)} immagini su CPU...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            
            # Inferenza (Senza Autocast per CPU)
            sr = model(lr)
            
            # Calcolo metriche
            metrics.update(sr.float(), hr.float())
            
            # Salvataggio TIFF 16-bit (Scientifico)
            save_as_tiff16(sr, output_base / "tiff_science" / f"sr_{i:04d}.tiff")
            
            # Salvataggio Preview PNG (Confronto visivo)
            lr_up = torch.nn.functional.interpolate(lr, size=(512,512), mode='bicubic')
            comp = torch.cat((lr_up, sr, hr), dim=3).clamp(0,1)
            save_image(comp, output_base / "png_preview" / f"comp_{i:04d}.png")

    # REPORT
    res = metrics.compute()
    print("\n📊 RISULTATI FINALI:")
    print(f"   SSIM: {res['ssim']:.4f}")
    print(f"   FSIM: {res['fsim']:.4f}")
    print(f"📂 Output salvati in: {output_base}")

    # Pulizia temp json se creato
    if args.lr_dir and args.hr_dir and json_path:
        try: os.remove(json_path)
        except: pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default rimosso per permettere il menu se vuoto
    parser.add_argument('--target', type=str, help='Nome Target (es: M42)')
    parser.add_argument('--lr_dir', type=str, help='Cartella Input TIFF')
    parser.add_argument('--hr_dir', type=str, help='Cartella Ground Truth TIFF')
    parser.add_argument('--model_path', type=str, help='Path manuale modello')
    
    args = parser.parse_args()
    run_test(args)