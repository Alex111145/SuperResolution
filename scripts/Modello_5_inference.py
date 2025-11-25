"""
MODELLO - STEP 5: GENERAZIONE E TEST (A6000 EDITION)
Questo script prende il dataset di TEST, genera le immagini Super-Risolte
e le salva su disco (sia FITS che PNG di confronto).
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from astropy.io import fits
from tqdm import tqdm
import json

# ============================================================
# 0. CONFIGURAZIONI & PATH (RUNPOD)
# ============================================================
# Abilita TF32 per velocità
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

CURRENT_SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = Path("/root/SuperResolution")
if not PROJECT_ROOT.exists(): PROJECT_ROOT = CURRENT_SCRIPT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# --- CONFIGURAZIONE INPUT (MODIFICA QUI SE SERVE) ---
TARGET_NAME = "M33"  # La cartella su cui testare
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / TARGET_NAME / "checkpoints" / "best.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / TARGET_NAME / "test_results"

SCALE = 6.4        # (512 / 80)
SMOOTHING = 'balanced'
# ============================================================

# Import Moduli
try:
    from src.architecture import HybridSuperResolutionModel
    from src.metrics import Metrics
    from src.dataset import AstronomicalDataset
except ImportError:
    sys.exit("❌ Errore Import: Assicurati di essere nella root corretta.")

# ============================================================
# 1. UTILS
# ============================================================
def create_test_json(split_dir):
    test_list = []
    # Cerca le coppie nel set di test
    test_dir = split_dir / "test"
    if not test_dir.exists(): return None
    
    for p in sorted(test_dir.glob("pair_*")):
        lr = p / "observatory.fits"
        hr = p / "hubble.fits"
        if lr.exists() and hr.exists():
            test_list.append({"patch_id": p.name, "ground_path": str(lr), "hubble_path": str(hr)})

    ft = split_dir / "test_temp.json"
    with open(ft, 'w') as f: json.dump(test_list, f)
    return ft

# ============================================================
# 2. MAIN RUN
# ============================================================
def run_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🧪 START TEST GENERATION: {TARGET_NAME}")
    print(f"   💾 Output Folder: {OUTPUT_DIR}")
    
    # Setup Cartelle Output
    (OUTPUT_DIR / "fits").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "comparison_png").mkdir(parents=True, exist_ok=True)

    # 1. Check Dati
    target_dir = ROOT_DATA_DIR / TARGET_NAME
    splits_dir = target_dir / "6_patches_aligned" / "splits"
    
    if not CHECKPOINT_PATH.exists():
        sys.exit(f"❌ Checkpoint mancante: {CHECKPOINT_PATH}")
        
    json_file = create_test_json(splits_dir)
    if not json_file:
        sys.exit("❌ Cartella 'test' non trovata negli splits.")

    # 2. Dataset Loader
    test_ds = AstronomicalDataset(json_file, base_path=PROJECT_ROOT, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
    
    # 3. Carica Modello
    print("   🔧 Caricamento Modello...")
    model = HybridSuperResolutionModel(
        smoothing=SMOOTHING, 
        device=device,
        output_size=512
    ).to(device)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 4. Loop di Generazione
    metrics_calc = Metrics()
    print(f"   🚀 Generazione di {len(test_loader)} immagini...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            
            # INFERENZA
            with torch.amp.autocast('cuda'):
                sr = model(lr)
            
            # Calcolo Metriche
            metrics_calc.update(sr, hr)
            
            # --- SALVATAGGIO RISULTATI ---
            
            # A. Salva FITS Scientifico (Solo il risultato AI)
            # .squeeze() toglie dimensioni batch/channel -> [512, 512]
            sr_numpy = sr.squeeze().float().cpu().numpy()
            fits_path = OUTPUT_DIR / "fits" / f"result_{i:04d}.fits"
            fits.PrimaryHDU(data=sr_numpy).writeto(fits_path, overwrite=True)
            
            # B. Salva PNG Confronto (Input | AI | Target)
            # Upscale semplice dell'input per metterlo a confronto (Nearest Neighbor)
            lr_up = F.interpolate(lr, size=(512,512), mode='nearest')
            
            # Unisci le 3 immagini in una striscia orizzontale
            comparison = torch.cat((lr_up, sr, hr), dim=3)
            
            png_path = OUTPUT_DIR / "comparison_png" / f"compare_{i:04d}.png"
            save_image(comparison, png_path, normalize=False)

    # 5. Risultati Finali
    final_metrics = metrics_calc.compute()
    print("\n" + "="*50)
    print("📊 REPORT FINALE")
    print("="*50)
    print(f"   PSNR Medio: {final_metrics['psnr']:.2f} dB")
    print(f"   SSIM Medio: {final_metrics['ssim']:.4f}")
    print(f"   📂 File salvati in: {OUTPUT_DIR}")
    
    # Pulizia
    if json_file.exists(): json_file.unlink()

if __name__ == "__main__":
    run_test()