import sys
import os
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
import json
import csv
import warnings
from pathlib import Path
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from typing import List

# === PATCH TORCHVISION (Necessaria per HAT) ===
import torchvision.transforms.functional as TF_functional
sys.modules['torchvision.transforms.functional_tensor'] = TF_functional


CURRENT_SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# Aggiunge la root del progetto al path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


try:
    from models.hybridmodels import HybridHATRealESRGAN
    from dataset.astronomical_dataset import AstronomicalDataset
    from utils.metrics import TrainMetrics, ssim_torch
except ImportError as e:
    sys.exit(f" Errore Import: {e}. Assicurati di essere nella root del progetto.")

# Ottimizzazione
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", category=UserWarning)

def save_as_tiff16(tensor, path):
    """
    Salva il tensore (normalizzato 0-1) come immagine TIFF a 16-bit.
    """
    arr = tensor.squeeze().float().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    # Conversione a 16-bit (0-65535)
    arr_u16 = (arr * 65535).astype(np.uint16)
    Image.fromarray(arr_u16, mode='I;16').save(path)

def detect_hybrid_params(state_dict):
    """
    Analizza i pesi salvati per dedurre la configurazione.
    Defaults impostati sulla versione 'Small' (Soft) usata nel training.
    """
  
    params = {
        'img_size': 128,
        'in_chans': 1,
        'embed_dim': 90,      
        'depths': (6, 6, 6, 6), 
        'num_heads': (6, 6, 6, 6),
        'window_size': 8,       
        'upscale': 4,
        'num_rrdb': 12,        
        'num_feat': 48,         
        'num_grow_ch': 24     
    }
    
    print(" Analisi parametri dal checkpoint...")
    
    # 1. Rileva embed_dim
    if 'hat.conv_first.weight' in state_dict:
        params['embed_dim'] = state_dict['hat.conv_first.weight'].shape[0]

    # 2. Rileva num_feat (Adattatore)
    if 'conv_adapt.weight' in state_dict:
        params['num_feat'] = state_dict['conv_adapt.weight'].shape[0]

    # 3. Rileva num_grow_ch (RRDB)
    if 'rrdb_trunk.0.rdb1.conv1.weight' in state_dict:
        params['num_grow_ch'] = state_dict['rrdb_trunk.0.rdb1.conv1.weight'].shape[0]

    # 4. Rileva numero blocchi RRDB
    max_rrdb = -1
    for k in state_dict.keys():
        if k.startswith('rrdb_trunk.'):
            try:
                idx = int(k.split('.')[1])
                max_rrdb = max(max_rrdb, idx)
            except: pass
    if max_rrdb >= 0:
        params['num_rrdb'] = max_rrdb + 1

    # 5. Rileva numero stadi HAT (depths)
    max_stage = -1
    for k in state_dict.keys():
        if k.startswith('hat.layers.'):
            try:
                idx = int(k.split('.')[2])
                max_stage = max(max_stage, idx)
            except: pass
            
    if max_stage >= 0:
        num_stages = max_stage + 1
        # Aggiorna depths e heads in base agli stadi trovati
        params['depths'] = tuple([6] * num_stages)
        params['num_heads'] = tuple([6] * num_stages)

    print(f"   • Rilevato: Embed={params['embed_dim']}, RRDB={params['num_rrdb']}, Feat={params['num_feat']}, Stadi={len(params['depths'])}")
    return params

def get_available_targets(output_root: Path) -> List[str]:
    """Restituisce la lista delle cartelle presenti in 'outputs'."""
    if not output_root.is_dir(): return []
    return sorted([p.name for p in output_root.iterdir() if p.is_dir()])

def run_test(target_model_folder: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device in uso: {device}")
    print(" Modalità: Inferenza Hybrid HAT+ESRGAN (Configurazione Soft)")

    # --- PERCORSI FILES ---
    target_path = OUTPUT_ROOT / target_model_folder
    
    BASE_RESULTS = target_path / "test_results_hybrid"
    TIFF_DIR = BASE_RESULTS / "tiff_16bit"
    PNG_DIR = BASE_RESULTS / "comparison_png"
    CHECKPOINT_DIR = target_path / "checkpoints"
    
    # Fallback se non esiste cartella checkpoints standard
    if not CHECKPOINT_DIR.exists(): CHECKPOINT_DIR = target_path

    TIFF_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- RICERCA CHECKPOINT ---
    checkpoints = list(CHECKPOINT_DIR.glob("best_hybrid_model.pth"))
    if not checkpoints:
        pths = list(CHECKPOINT_DIR.glob("hybrid_epoch_*.pth"))
        if pths:
            checkpoints = [sorted(pths, key=lambda x: int(x.stem.split('_')[-1]))[-1]]
    
    if not checkpoints:
        checkpoints = list(CHECKPOINT_DIR.glob("*.pth"))

    if not checkpoints:
        print(f" Nessun checkpoint trovato in {CHECKPOINT_DIR}")
        return
    
    CHECKPOINT_PATH = checkpoints[0]
    print(f" Caricamento checkpoint: {CHECKPOINT_PATH.name}")

    # --- CARICAMENTO MODELLO ---
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'net_g' in checkpoint:
            state_dict = checkpoint['net_g']
        else:
            state_dict = checkpoint

        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # Auto-detection (partendo dai default Soft)
        model_params = detect_hybrid_params(clean_state_dict)
        
        model = HybridHATRealESRGAN(**model_params).to(device)
        model.load_state_dict(clean_state_dict, strict=True)
        print(" Pesi caricati correttamente.")
        
    except Exception as e:
        print(f" Errore caricamento modello: {e}")
        return

    model.eval()

    # --- DATASET ---
    clean_name = target_model_folder.replace("_Hybrid", "").replace("_SR", "")
    targets_names = clean_name.split("_")
    
    all_test_data = []
    print("\n Ricerca Dataset Test...")
    
    for t in targets_names:
        json_path = ROOT_DATA_DIR / t / "8_dataset_split" / "splits_json" / "test.json"
        
        if json_path.exists():
            print(f"   -> Trovato: {t}")
            with open(json_path, 'r') as f:
                data = json.load(f)
                all_test_data.extend(data)
        else:
            print(f"     Non trovato JSON per: {t}")

    if not all_test_data:
        print(" Nessun dato di test trovato.")
        return

    TEMP_JSON = BASE_RESULTS / "temp_test_infer.json"
    with open(TEMP_JSON, 'w') as f: json.dump(all_test_data, f)

    test_ds = AstronomicalDataset(str(TEMP_JSON), base_path=PROJECT_ROOT, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    
    metrics = TrainMetrics()
    csv_log_path = BASE_RESULTS / "test_metrics.csv"
    
    print(f"\n Inizio inferenza su {len(test_ds)} immagini...")

    # --- LOOP DI INFERENZA ---
    with open(csv_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "PSNR", "SSIM"])

        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader, desc="Processing", unit="img")):
                lr = batch['lr'].to(device)
                hr = batch['hr'].to(device)
                
                # 1. Super-Resolution
                sr = model(lr)
                sr_clamped = torch.clamp(sr, 0, 1)
                
                # 2. Aggiornamento Metriche Globali
                metrics.update(sr_clamped, hr)
                
                # 3. Calcolo Metriche Locali (per CSV)
                mse_val = F.mse_loss(sr_clamped, hr)
                psnr_val = 10 * torch.log10(1.0 / (mse_val + 1e-8)).item()
                ssim_val = ssim_torch(sr_clamped, hr).item()
                
                writer.writerow([f"img_{i:04d}", f"{psnr_val:.4f}", f"{ssim_val:.4f}"])

                # 4. Creazione "Tris" (LR | SR | HR)
                lr_up = F.interpolate(lr, size=sr_clamped.shape[2:], mode='nearest')
                comparison = torch.cat((lr_up, sr_clamped, hr), dim=3)
                
                # 5. Salvataggi
                save_as_tiff16(sr_clamped, TIFF_DIR / f"test_{i:04d}_sr.tiff")
                vutils.save_image(comparison, PNG_DIR / f"test_{i:04d}_tris.png")

    if TEMP_JSON.exists(): os.remove(TEMP_JSON)

    results = metrics.compute()
    print("\n" + "="*50)
    print(f" TEST COMPLETATO.")
    print(f" PSNR Medio: {results['psnr']:.2f} dB")
    print(f" SSIM Medio: {results['ssim']:.4f}")
    print(f" TIFF salvati in: .../{TIFF_DIR.parent.name}/{TIFF_DIR.name}")
    print(f" PNG salvati in:  .../{PNG_DIR.parent.name}/{PNG_DIR.name}")
    print(f" CSV Metriche:    {csv_log_path.name}")
    print("="*50)

if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    available = get_available_targets(OUTPUT_ROOT)
    
    if available:

        print("Cartelle output disponibili:")
        for idx, name in enumerate(available):
            print(f"   [{idx}] {name}")
            
        sel = input("\nSeleziona numero o nome cartella: ").strip()
        
        target = None
        if sel.isdigit():
            idx = int(sel)
            if 0 <= idx < len(available):
                target = available[idx]
        elif sel in available:
            target = sel
            
        if target:
            run_test(target)
        else:
            print(" Selezione non valida.")
    else:
        print(" Nessuna cartella trovata in 'outputs'. Assicurati di aver fatto il training.")
