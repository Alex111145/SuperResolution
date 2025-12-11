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

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Setup Percorsi
CURRENT_SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
ROOT_DATA_DIR = PROJECT_ROOT / "data"

try:
    # Usa architecture_train per caricare la classe corretta per SwinIR
    from src.architecture_train import TrainHybridModel
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

def run_test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🧪 INFERENZA SWINIR & METRICHE (SSIM, FSIM)")
    
    # Selezione Modalità
    if args.lr_dir and args.hr_dir:
        # Modalità Custom
        lr_path = Path(args.lr_dir)
        hr_path = Path(args.hr_dir)
        json_path = create_temp_json_from_dirs(lr_path, hr_path)
        output_base = PROJECT_ROOT / "outputs" / "custom_inference"
        checkpoint_path = Path(args.model_path) if args.model_path else None
    else:
        # Modalità Standard (Cerca output del Train_Worker)
        target = args.target if args.target else "M42"
        splits_dir = ROOT_DATA_DIR / target / "8_dataset_split" / "splits_json"
        json_path = splits_dir / "test.json"
        if not json_path.exists():
            json_path = splits_dir / "val.json"
            print("⚠️ Test set non trovato, uso Validation set.")
            
        output_base = PROJECT_ROOT / "outputs" / target / "test_results_tiff"
        
        if args.model_path:
            checkpoint_path = Path(args.model_path)
        else:
            # Cerca nel path generato da Train_Worker
            checkpoint_path = PROJECT_ROOT / "outputs" / f"{target}_SwinIR_FULL_TRAIN" / "checkpoints" / "best_train_model.pth"

    if not checkpoint_path or not checkpoint_path.exists():
        sys.exit(f"❌ Modello non trovato: {checkpoint_path}")

    # Output Dir
    (output_base / "tiff_science").mkdir(parents=True, exist_ok=True)
    (output_base / "png_preview").mkdir(parents=True, exist_ok=True)

    # Dataset
    test_ds = AstronomicalDataset(json_path, base_path=PROJECT_ROOT, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
    
    # Load Model
    print(f"   Caricamento pesi: {checkpoint_path.name}")
    model = TrainHybridModel(output_size=512).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    metrics = Metrics()
    
    print(f"   🚀 Elaborazione {len(test_ds)} immagini...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            
            with torch.cuda.amp.autocast():
                sr = model(lr)
            
            metrics.update(sr.float(), hr.float())
            
            save_as_tiff16(sr, output_base / "tiff_science" / f"sr_{i:04d}.tiff")
            
            lr_up = torch.nn.functional.interpolate(lr, size=(512,512), mode='bicubic')
            comp = torch.cat((lr_up, sr, hr), dim=3).clamp(0,1)
            save_image(comp, output_base / "png_preview" / f"comp_{i:04d}.png")

    # REPORT (Senza PSNR)
    res = metrics.compute()
    print("\n📊 RISULTATI FINALI:")
    print(f"   SSIM: {res['ssim']:.4f}")
    print(f"   FSIM: {res['fsim']:.4f}")
    print(f"📂 Output salvati in: {output_base}")

    if args.lr_dir and args.hr_dir:
        try: os.remove(json_path)
        except: pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default="M42", help='Nome Target')
    parser.add_argument('--lr_dir', type=str, help='Cartella Input TIFF')
    parser.add_argument('--hr_dir', type=str, help='Cartella Ground Truth TIFF')
    parser.add_argument('--model_path', type=str, help='Path manuale modello')
    
    args = parser.parse_args()
    run_test(args)