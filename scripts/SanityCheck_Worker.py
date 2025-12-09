#!/usr/bin/env python3
import os
import argparse
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm

# ================= CONFIGURAZIONE PATH =================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# IMPORTS SPECIFICI PER SANITY CHECK
try:
    from src.architecture_sanity import SanityHybridModel
    from src.dataset import AstronomicalDataset 
    from src.losses_sanity import SanityStarLoss
    from src.metrics_sanity import SanityMetrics
except ImportError as e:
    sys.exit(f"❌ Errore Import Sanity Modules: {e}")

# ================= HYPERPARAMETERS SANITY CHECK =================
BATCH_SIZE = 4        
ACCUM_STEPS = 1       
LR = 5e-4             
TOTAL_EPOCHS = 1000

LOG_INTERVAL = 5     
IMAGE_INTERVAL = 20  

torch.backends.cudnn.benchmark = True 

def train_worker(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    
    print(f"🚀 SANITY CHECK WORKER | GPU: {num_gpus}")
    
    target_name = f"{args.target}_SANITY_CHECK_RUN"
    out_dir = PROJECT_ROOT / "outputs" / target_name
    save_dir = out_dir / "checkpoints"
    img_dir = out_dir / "images"
    log_dir = out_dir / "tensorboard"
    for d in [save_dir, img_dir, log_dir]: d.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(log_dir))

    # --- MODIFICA: Legge dalla cartella dedicata '8_sanity_split' ---
    splits_dir = PROJECT_ROOT / "data" / args.target / "8_sanity_split" / "splits_json"
    
    if not splits_dir.exists():
        sys.exit(f"❌ Errore: Cartella split non trovata: {splits_dir}\n   Esegui prima 'SanityCheck_Prepare.py'!")

    with open(splits_dir / "train.json") as f: train_data = json.load(f)
    with open(splits_dir / "val.json") as f: val_data = json.load(f)
    
    # File temporanei
    ft_path = splits_dir / f"temp_train_sanity_{os.getpid()}.json"
    fv_path = splits_dir / f"temp_val_sanity_{os.getpid()}.json"
    with open(ft_path, 'w') as f: json.dump(train_data, f)
    with open(fv_path, 'w') as f: json.dump(val_data, f)

    train_ds = AstronomicalDataset(ft_path, base_path=PROJECT_ROOT, augment=False)
    val_ds = AstronomicalDataset(fv_path, base_path=PROJECT_ROOT, augment=False)
    
    curr_batch = min(BATCH_SIZE, len(train_ds))
    
    train_loader = DataLoader(train_ds, batch_size=curr_batch, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    # --- MODELLO SANITY ---
    print(f"   🔥 Init SanityHybridModel (Smoothing=None)...")
    model = SanityHybridModel(smoothing='none', device=device).to(device)
    
    if num_gpus > 1:
        model = nn.DataParallel(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-7)
    
    # --- LOSS SANITY ---
    criterion = SanityStarLoss().to(device)
    scaler = torch.cuda.amp.GradScaler() 
    best_psnr = 0.0

    # === TRAINING LOOP ===
    for epoch in range(1, TOTAL_EPOCHS + 1):
        model.train()
        acc_loss = 0.0
        
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Sanity Ep {epoch}", ncols=100, colour='yellow', leave=False)
        
        for i, batch in enumerate(pbar):
            lr = batch['lr'].to(device, non_blocking=True)
            hr = batch['hr'].to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                pred = model(lr)
                loss, _ = criterion(pred, hr)
            
            scaler.scale(loss).backward()
            acc_loss += loss.item()
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        
        # Validation
        if epoch % LOG_INTERVAL == 0:
            avg_loss = acc_loss / len(train_loader)
            writer.add_scalar('Train/Loss', avg_loss, epoch)
            
            model.eval()
            metrics = SanityMetrics()
            with torch.inference_mode(): 
                for v_batch in val_loader:
                    v_lr = v_batch['lr'].to(device)
                    v_hr = v_batch['hr'].to(device)
                    with torch.cuda.amp.autocast():
                        v_pred = model(v_lr)
                    metrics.update(v_pred.float(), v_hr.float())
            
            res = metrics.compute()
            writer.add_scalar('Val/PSNR', res['psnr'], epoch)
            
            print(f"   Ep {epoch:04d} | Loss: {avg_loss:.4f} | PSNR: {res['psnr']:.2f} dB")

            # Salvataggio
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            if res['psnr'] > best_psnr:
                best_psnr = res['psnr']
                torch.save(state_dict, save_dir / "best_sanity_model.pth")
            torch.save(state_dict, save_dir / "last_sanity_model.pth")

            if epoch % IMAGE_INTERVAL == 0:
                v_lr_up = torch.nn.functional.interpolate(v_lr, size=(512,512), mode='nearest')
                comp = torch.cat((v_lr_up, v_pred, v_hr), dim=3).clamp(0,1)
                vutils.save_image(comp, img_dir / f"sanity_epoch_{epoch}.png")

    try:
        if ft_path.exists(): ft_path.unlink()
        if fv_path.exists(): fv_path.unlink()
    except: pass
    print(f"\n✅ Sanity Check Terminato.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    args = parser.parse_args()
    train_worker(args)