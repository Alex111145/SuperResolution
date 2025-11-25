"""
TRAINING A6000 BUNKER EDITION (48GB VRAM) + TENSORBOARD IMAGES
Configurazione Blindata anti-OOM.
Batch Size: 2 | Accumulation: 16
NOVITÀ: Le immagini di preview vengono caricate su TensorBoard.
ADATTATO: Per compatibilità Windows/Linux e percorsi relativi.
"""

import argparse
import sys
import os
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from torch.cuda.amp import GradScaler 

# ============================================================================
# 1. CONFIGURAZIONE HARDWARE BLINDATA
# ============================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Abilita TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# CPU Workers
CPU_CORES = os.cpu_count() or 16
WORKERS = min(16, CPU_CORES) 

# CONFIGURAZIONE DI SICUREZZA (Batch 2 / Accum 16)
HARDWARE_CONFIG = {
    'batch_size': 2,           
    'accum_steps': 16,         
    'target_hr_size': 512,
}

# ============================================================================
# PATHS & IMPORTS PORTABILI
# ============================================================================
# Base del progetto, risolta sempre in modo relativo allo script
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))
ROOT_DATA_DIR = PROJECT_ROOT / "data"

try:
    # Gli import ora funzioneranno grazie a sys.path.insert(0, str(PROJECT_ROOT))
    from src.architecture import HybridSuperResolutionModel
    from src.dataset import AstronomicalDataset
    from src.losses import CombinedLoss
    from src.metrics import Metrics
except ImportError:
    sys.exit(f"❌ Errore import src. Assicurati che la cartella 'src' esista nella root del progetto: {PROJECT_ROOT}/src")

# ============================================================================
# UTILS
# ============================================================================
def create_json_files_if_missing(split_dir):
    """
    Crea i file JSON temporanei per il dataset, usando Path.
    """
    def scan_split(split_name):
        d = split_dir / split_name
        data = []
        if not d.exists(): return data
        
        for entry in os.scandir(d):
            if entry.is_dir() and entry.name.startswith("pair_"):
                # Crea percorsi assoluti per la portabilità
                lr = (d / entry.name / "observatory.fits").resolve()
                hr = (d / entry.name / "hubble.fits").resolve()
                
                if lr.exists() and hr.exists():
                    data.append({
                        "patch_id": entry.name, 
                        # I percorsi salvati nel JSON devono essere stringhe
                        "ground_path": str(lr), 
                        "hubble_path": str(hr)
                    })
        return sorted(data, key=lambda x: x['patch_id'])

    print("   📂 Scansione dataset (Train/Val)...")
    train_list = scan_split("train")
    val_list = scan_split("val")

    ft = split_dir / "train_temp.json"
    fv = split_dir / "val_temp.json"
    
    with open(ft, 'w') as f: json.dump(train_list, f, indent=4)
    with open(fv, 'w') as f: json.dump(val_list, f, indent=4)
    
    print(f"   JSON temporanei creati: {len(train_list)} train, {len(val_list)} val.")
    
    return ft, fv

# ============================================================================
# TRAIN LOOP
# ============================================================================
def train(args, target_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.empty_cache() 
    
    print(f"\n🚀 START TRAIN A6000 BUNKER + TB IMAGES: {target_dir.name} ({device})")
    eff_batch = HARDWARE_CONFIG['batch_size'] * HARDWARE_CONFIG['accum_steps']
    print(f"   🔥 Batch Efficace: {eff_batch}")
    
    # OUTPUTS basati su PROJECT_ROOT e nome del target
    out_dir = PROJECT_ROOT / "outputs" / target_dir.name
    save_dir = out_dir / "checkpoints"
    log_dir = out_dir / "tensorboard"
    img_dir = out_dir / "images"
    for d in [save_dir, log_dir, img_dir]: d.mkdir(parents=True, exist_ok=True)
    
    # Assumiamo che lo split sia nella cartella delle patch finali
    # Modificato da '6_patches_aligned' (nome vecchio) a '6_patches_final' (nome nuovo)
    # oppure crea il percorso in base a dove lo script di split salva (Modello_2_pre_da_usopatch_dataset_step3.py salva in '6_patches_final/splits')
    
    # Se Modello_2_pre_da_usopatch_dataset_step3.py è lo Step 4 che hai fornito:
    splits_dir = target_dir / "6_patches_final" / "splits" 
    
    if not splits_dir.exists():
        sys.exit(f"❌ Splits non trovati in {splits_dir}. Assicurati di aver eseguito lo Step 4.")

    json_train, json_val = create_json_files_if_missing(splits_dir)
    
    # Dataset (base_path è PROJECT_ROOT per risolvere i percorsi 'src' e data)
    train_ds = AstronomicalDataset(json_train, base_path=PROJECT_ROOT, augment=True)
    val_ds = AstronomicalDataset(json_val, base_path=PROJECT_ROOT, augment=False)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=HARDWARE_CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )
    
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    # Pulisce i JSON temporanei
    if json_train.exists(): json_train.unlink()
    if json_val.exists(): json_val.unlink()

    print("   🔧 Inizializzazione Modello...")
    model = HybridSuperResolutionModel(
        smoothing=args.smoothing, 
        device=device, 
        output_size=HARDWARE_CONFIG['target_hr_size']
    ).to(device)
    
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"   📥 Resuming: {resume_path.name}")
            model.load_state_dict(torch.load(resume_path, map_location=device))
        else:
             print(f"   ⚠️ File di resume non trovato: {resume_path}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = CombinedLoss().to(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    scaler = GradScaler() if device.type == 'cuda' else None
    writer = SummaryWriter(str(log_dir))
    
    best_loss = float('inf')
    global_step = 0
    accum_steps = HARDWARE_CONFIG['accum_steps']

    print("   🟢 Training Avviato!")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad(set_to_none=True)
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{args.epochs}", unit="batch")
        
        for i, batch in enumerate(pbar):
            lr = batch['lr'].to(device, non_blocking=True) 
            hr = batch['hr'].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda') if device.type == 'cuda' else torch.no_grad():
                pred = model(lr) 
                loss, _ = criterion(pred, hr)
                loss = loss / accum_steps 
            
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (i + 1) % accum_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            current_loss = loss.item() * accum_steps
            epoch_loss += current_loss
            global_step += 1
            
            writer.add_scalar('Train/Loss_Iter', current_loss, global_step)
            pbar.set_postfix({'loss': f"{current_loss:.4f}"})
            
        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Train/Loss_Epoch', avg_loss, epoch)
        
        # VALIDATION & IMAGE LOGGING
        model.eval()
        val_loss = 0
        metrics = Metrics()
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                lr = batch['lr'].to(device, non_blocking=True)
                hr = batch['hr'].to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda') if device.type == 'cuda' else torch.no_grad():
                    pred = model(lr)
                    v_loss, _ = criterion(pred, hr)
                
                val_loss += v_loss.item()
                metrics.update(pred, hr)
                
                # === SALVATAGGIO IMMAGINI (DISK + TENSORBOARD) ===
                if i == 0: 
                    # Upscale semplice dell'input per confronto visivo (Nearest Neighbor)
                    lr_up = F.interpolate(lr, size=(512,512), mode='nearest')
                    
                    # Unisci: Input(Sgranato) | Predizione(AI) | Target(Hubble)
                    comp = torch.cat((lr_up, pred, hr), dim=3)
                    
                    # 1. Salva su DISCO
                    vutils.save_image(comp, img_dir / f"val_ep_{epoch+1}.png", normalize=False)
                    
                    # 2. Manda a TENSORBOARD (Tab IMAGES)
                    # Comp.cpu() + .float() per evitare problemi di tipo con Tensorboard
                    writer.add_image('Validation/Preview (Input | AI | Target)', comp.squeeze(0).cpu().float().clamp(0, 1), epoch)
                # =================================================
            
        res = metrics.compute()
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"   📉 Val Loss: {avg_val_loss:.4f} | PSNR: {res['psnr']:.2f} | SSIM: {res['ssim']:.3f}")
        
        writer.add_scalar('Val/Loss', avg_val_loss, epoch)
        writer.add_scalar('Val/PSNR', res['psnr'], epoch)
        writer.add_scalar('Val/SSIM', res['ssim'], epoch)
        
        scheduler.step(avg_val_loss)
        
        # Salvataggio checkpoint usando Path
        torch.save(model.state_dict(), save_dir / "last.pth")
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), save_dir / "best.pth")
            print("   🏆 New Best Model!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--smoothing', type=str, default='balanced') 
    parser.add_argument('--resume', type=str)
    args = parser.parse_args()
    
    if args.target:
        train(args, ROOT_DATA_DIR / args.target)
    else:
        # Ricerca target basata sulla presenza della cartella splits
        valid = [d for d in ROOT_DATA_DIR.iterdir() if (d / "6_patches_final" / "splits").exists()]
        if len(valid) == 1:
            train(args, valid[0])
        elif len(valid) > 1:
            print("\nTarget disponibili per il Training:")
            for i, d in enumerate(valid): print(f"{i+1}: {d.name}")
            try:
                s = int(input(f"Seleziona (1-{len(valid)}): ")) - 1
                if 0 <= s < len(valid):
                    train(args, valid[s])
            except: pass