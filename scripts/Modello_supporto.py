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

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# IMPORT MODULI TRAINING
try:
    from src.architecture import TrainHybridModel
    from src.dataset import AstronomicalDataset
    from src.losses import TrainStarLoss
    from src.metrics import Metrics as TrainMetrics 
except ImportError as e:
    sys.exit(f"❌ Errore Import Training Modules: {e}")

# ================= HYPERPARAMETERS TRAINING (FIX STABILITÀ GTX/RTX) =================

BATCH_SIZE = 5        # Basso per evitare OOM e instabilità
ACCUM_STEPS = 16      # 2 * 16 = 32 (Batch Size Effettivo simulato)
LR = 2e-4             
TOTAL_EPOCHS = 500    

LOG_INTERVAL = 1      
IMAGE_INTERVAL = 10   

# --- FIX CRITICO PER "ILLEGAL MEMORY ACCESS" ---
# benchmark=True può causare crash su alcune GPU durante la selezione dell'algoritmo.
# Lo disabilitiamo per stabilità.
torch.backends.cudnn.benchmark = False 
torch.backends.cudnn.deterministic = True

def train_worker(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    
    target_name = f"{args.target}_SwinIR_FULL_TRAIN"
    out_dir = PROJECT_ROOT / "outputs" / target_name
    save_dir = out_dir / "checkpoints"
    img_dir = out_dir / "images"
    log_dir = out_dir / "tensorboard"
    for d in [save_dir, img_dir, log_dir]: d.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(log_dir))

    splits_dir = PROJECT_ROOT / "data" / args.target / "8_dataset_split" / "splits_json"
    
    if not splits_dir.exists():
        sys.exit(f"❌ Splits non trovati in {splits_dir}. Esegui Dataset_1.py!")

    with open(splits_dir / "train.json") as f: train_data = json.load(f)
    with open(splits_dir / "val.json") as f: val_data = json.load(f)
    
    ft_path = splits_dir / f"temp_train_real_{os.getpid()}.json"
    fv_path = splits_dir / f"temp_val_real_{os.getpid()}.json"
    with open(ft_path, 'w') as f: json.dump(train_data, f)
    with open(fv_path, 'w') as f: json.dump(val_data, f)

    train_ds = AstronomicalDataset(ft_path, base_path=PROJECT_ROOT, augment=True)
    val_ds = AstronomicalDataset(fv_path, base_path=PROJECT_ROOT, augment=False)
    
    # Num workers basso e pin_memory=False per massima stabilità su Workstation
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,      # Ridotto per evitare overhead CPU
        pin_memory=False,   # DISABILITATO per evitare problemi di memoria condivisa
        drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    print(f"   🔥 Init TrainHybridModel (SwinIR) - Batch: {BATCH_SIZE} | Accum: {ACCUM_STEPS}")
    print(f"      CUDNN Benchmark: DISABLED (Stability Mode)")
    
    model = TrainHybridModel(smoothing=None, device=device).to(device)
    
    if num_gpus > 1: model = nn.DataParallel(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-7)
    
    criterion = TrainStarLoss().to(device)
    scaler = torch.cuda.amp.GradScaler() 
    best_ssim = 0.0

    print("   🚀 Avvio Training...")

    for epoch in range(1, TOTAL_EPOCHS + 1):
        model.train()
        acc_loss = 0.0
        
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Train Ep {epoch}", ncols=100, colour='green', leave=False)
        
        for i, batch in enumerate(pbar):
            lr = batch['lr'].to(device, non_blocking=True)
            hr = batch['hr'].to(device, non_blocking=True)
            
            # Autocast protetto
            with torch.amp.autocast('cuda'):
                pred = model(lr)
                loss, _ = criterion(pred, hr)
                loss = loss / ACCUM_STEPS
            
            # Backward pass con Scaler
            scaler.scale(loss).backward()
            acc_loss += loss.item() * ACCUM_STEPS
            
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            pbar.set_postfix(loss=f"{loss.item()*ACCUM_STEPS:.4f}")

        scheduler.step()
        
        if epoch % LOG_INTERVAL == 0:
            avg_loss = acc_loss / len(train_loader)
            writer.add_scalar('Train/Loss', avg_loss, epoch)
            
            model.eval()
            metrics = TrainMetrics()
            
            # Valutazione
            with torch.inference_mode(): 
                for v_batch in val_loader:
                    v_lr = v_batch['lr'].to(device)
                    v_hr = v_batch['hr'].to(device)
                    with torch.amp.autocast('cuda'):
                        v_pred = model(v_lr)
                    metrics.update(v_pred.float(), v_hr.float())
            
            res = metrics.compute()
            writer.add_scalar('Val/SSIM', res['ssim'], epoch)
            writer.add_scalar('Val/FSIM', res['fsim'], epoch)
            
            print(f"   Ep {epoch:04d} | Loss: {avg_loss:.4f} | SSIM: {res['ssim']:.4f} | FSIM: {res['fsim']:.4f}")

            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            if res['ssim'] > best_ssim:
                best_ssim = res['ssim']
                torch.save(state, save_dir / "best_train_model.pth")
            torch.save(state, save_dir / "last_train_model.pth")

            if epoch % IMAGE_INTERVAL == 0:
                v_lr_up = torch.nn.functional.interpolate(v_lr, size=(512,512), mode='bicubic')
                comp = torch.cat((v_lr_up, v_pred, v_hr), dim=3).clamp(0,1)
                vutils.save_image(comp, img_dir / f"train_epoch_{epoch}.png")

    try:
        if ft_path.exists(): ft_path.unlink()
        if fv_path.exists(): fv_path.unlink()
    except: pass
    print(f"\n✅ Training SwinIR Completato.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    args = parser.parse_args()
    train_worker(args)