import sys
import os
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
import json
import warnings
from pathlib import Path
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from typing import List

import torchvision.transforms.functional as TF_functional
sys.modules['torchvision.transforms.functional_tensor'] = TF_functional


CURRENT_SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
ROOT_DATA_DIR = PROJECT_ROOT / "data"


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


try:
    from models.hybridmodels import HybridHATRealESRGAN
    from dataset.astronomical_dataset import AstronomicalDataset
    from utils.metrics import TrainMetrics
except ImportError as e:
    sys.exit(f"❌ Errore Import: {e}. Assicurati di essere nella root del progetto.")

# Ottimizzazione
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", category=UserWarning)

def save_as_tiff16(tensor, path):
    """
    Salva il tensore (normalizzato 0-1) come immagine TIFF a 16-bit.
    """
    arr = tensor.squeeze().float().cpu().numpy()
    arr = np.clip(arr, 0, 1)

    arr_u16 = (arr * 65535).astype(np.uint16)
    Image.fromarray(arr_u16, mode='I;16').save(path)

def detect_hybrid_params(state_dict):
    """
    Analizza i pesi salvati per dedurre la configurazione del modello (Small vs Large).
    """
    params = {
        'img_size': 128,
        'in_chans': 1,
        'embed_dim': 180,       
        'depths': (6, 6, 6, 6, 6, 6),
        'num_heads': (6, 6, 6, 6, 6, 6),
        'window_size': 8,      
        'upscale': 4,
        'num_rrdb': 23,       
        'num_feat': 64,     
        'num_grow_ch': 32       
    }
    
    print("🔍 Analisi parametri dal checkpoint...")
    

    if 'hat.conv_first.weight' in state_dict:
        e_dim = state_dict['hat.conv_first.weight'].shape[0]
        params['embed_dim'] = e_dim
        print(f"   • Embed Dim rilevata: {e_dim}")


    if 'conv_adapt.weight' in state_dict:
        n_feat = state_dict['conv_adapt.weight'].shape[0]
        params['num_feat'] = n_feat
        print(f"   • Num Feat rilevati: {n_feat}")


    if 'rrdb_trunk.0.rdb1.conv1.weight' in state_dict:
        g_ch = state_dict['rrdb_trunk.0.rdb1.conv1.weight'].shape[0]
        params['num_grow_ch'] = g_ch
        print(f"   • Growth Channel rilevati: {g_ch}")

    max_rrdb_idx = -1
    for k in state_dict.keys():
        if k.startswith('rrdb_trunk.'):
            try:
          
                idx = int(k.split('.')[1])
                if idx > max_rrdb_idx:
                    max_rrdb_idx = idx
            except: pass
    if max_rrdb_idx >= 0:
        params['num_rrdb'] = max_rrdb_idx + 1
        print(f"   • Blocchi RRDB rilevati: {params['num_rrdb']}")

    max_stage_idx = -1
    for k in state_dict.keys():
        if k.startswith('hat.layers.'):
            try:
                idx = int(k.split('.')[2])
                if idx > max_stage_idx:
                    max_stage_idx = idx
            except: pass
            
    if max_stage_idx >= 0:
        num_stages = max_stage_idx + 1
        params['depths'] = tuple([6] * num_stages)
        params['num_heads'] = tuple([6] * num_stages)
        print(f"   • Stadi HAT rilevati: {num_stages} (Depths: {params['depths']})")

 
    
    return params

def get_available_targets(output_root: Path) -> List[str]:
    """Restituisce la lista delle cartelle presenti in 'outputs'."""
    if not output_root.is_dir():
        return []
    return sorted([p.name for p in output_root.iterdir() if p.is_dir()])

def run_test(target_model_folder: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device in uso: {device}")
    print("📋 Modalità: Inferenza Hybrid HAT+ESRGAN con output Tris (LR|SR|HR)")


    target_path = OUTPUT_ROOT / target_model_folder
    
    BASE_RESULTS = target_path / "test_results_hybrid"
    TIFF_DIR = BASE_RESULTS / "tiff_16bit"
    PNG_DIR = BASE_RESULTS / "comparison_png"
    CHECKPOINT_DIR = target_path / "checkpoints"
    

    if not CHECKPOINT_DIR.exists():
        CHECKPOINT_DIR = target_path

    TIFF_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    
 
    checkpoints = list(CHECKPOINT_DIR.glob("best_hybrid_model.pth"))
    if not checkpoints:
        pths = list(CHECKPOINT_DIR.glob("hybrid_epoch_*.pth"))
        if pths:
   
            checkpoints = [sorted(pths, key=lambda x: int(x.stem.split('_')[-1]))[-1]]
    
    if not checkpoints:
      
        checkpoints = list(CHECKPOINT_DIR.glob("*.pth"))

    if not checkpoints:
        print(f"❌ Nessun checkpoint trovato in {CHECKPOINT_DIR}")
        return
    
    CHECKPOINT_PATH = checkpoints[0]
    print(f"📦 Caricamento checkpoint: {CHECKPOINT_PATH.name}")

    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'net_g' in checkpoint:
            state_dict = checkpoint['net_g']
        else:
            state_dict = checkpoint

        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
      
        model_params = detect_hybrid_params(clean_state_dict)
        
        model = HybridHATRealESRGAN(**model_params).to(device)

 
        model.load_state_dict(clean_state_dict, strict=True)
        print("✓ Pesi caricati correttamente (Strict Mode).")
        
    except Exception as e:
        print(f"❌ Errore critico nel caricamento del modello: {e}")
        print("   Suggerimento: Verifica che il file models/hybridmodels.py sia aggiornato.")
        return

    model.eval()

 
    clean_name = target_model_folder.replace("_Hybrid", "").replace("_SR", "")
    targets_names = clean_name.split("_")
    
    all_test_data = []
    print("\n📂 Ricerca Dataset Test...")
    
    for t in targets_names:
       
        json_path = ROOT_DATA_DIR / t / "8_dataset_split" / "splits_json" / "test.json"
        
        if json_path.exists():
            print(f"   -> Trovato: {t}")
            with open(json_path, 'r') as f:
                data = json.load(f)
                
                for item in data: item['target_name'] = t 
                all_test_data.extend(data)
        else:
        
            print(f"   ⚠️  Non trovato JSON per: {t}")

    if not all_test_data:
        print("❌ Nessun dato di test trovato. Verifica le cartelle in 'data/'.")
        return

    TEMP_JSON = BASE_RESULTS / "temp_test_infer.json"
    with open(TEMP_JSON, 'w') as f:
        json.dump(all_test_data, f)

    test_ds = AstronomicalDataset(str(TEMP_JSON), base_path=PROJECT_ROOT, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    
    metrics = TrainMetrics()
    
    print(f"\n🚀 Inizio inferenza su {len(test_ds)} immagini...")


    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Upscaling", unit="img")):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            
   
            sr = model(lr)
            sr_clamped = torch.clamp(sr, 0, 1)
         
            lr_up = F.interpolate(lr, size=sr_clamped.shape[2:], mode='nearest')
            comparison = torch.cat((lr_up, sr_clamped, hr), dim=3)
            
        
            save_as_tiff16(sr_clamped, TIFF_DIR / f"test_{i:04d}_sr.tiff")
         
            vutils.save_image(comparison, PNG_DIR / f"test_{i:04d}_tris.png")
            

            metrics.update(sr_clamped, hr)

   
    if TEMP_JSON.exists(): os.remove(TEMP_JSON)

    results = metrics.compute()
    print("\n" + "="*50)
    print(f"✅ TEST COMPLETATO.")
    print(f"📊 PSNR Medio: {results['psnr']:.2f} dB")
    print(f"📊 SSIM Medio: {results['ssim']:.4f}")
    print(f"💾 TIFF salvati in: .../{TIFF_DIR.parent.name}/{TIFF_DIR.name}")
    print(f"🖼️  PNG salvati in:  .../{PNG_DIR.parent.name}/{PNG_DIR.name}")
    print("="*50)

if __name__ == "__main__":
    available = get_available_targets(OUTPUT_ROOT)
    
    if available:
        print("\n==========================================")
        print("      INFERENZA HYBRID HAT (XPixel)       ")
        print("==========================================\n")
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
            print("❌ Selezione non valida.")
    else:
        print("❌ Nessuna cartella trovata in 'outputs'. Assicurati di aver fatto il training.")
