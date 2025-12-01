"""
TRAINING WORKER (NVIDIA H200 - MAX SATURATION)
Ottimizzato per VRAM 60% -> 90% e CPU 30% -> 80%+
"""
import os
import cv2
cv2.setNumThreads(0) 

import argparse
import sys
import json
import torch
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter 
import traceback

# === NVIDIA OPTIMIZATIONS ===
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
# ============================

# Setup Path
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
ROOT_DATA_DIR = PROJECT_ROOT / "data"

print(f"🔧 [Worker] Loading modules from {PROJECT_ROOT}...")
try:
    from src.architecture import HybridSuperResolutionModel
    from src.dataset import AstronomicalDataset
    from src.losses import CombinedLoss
    from src.metrics import Metrics
    print("   ✅ Moduli src importati correttamente.")
except Exception as e:
    print(f"\n❌ ERRORE CRITICO NELL'IMPORTARE SRC:")
    traceback.print_exc()
    sys.exit(1)

def create_partitioned_json(split_dir, rank, total_gpus):
    def load_and_slice(json_name):
        f_path = split_dir / json_name
        if not f_path.exists(): return []
        with open(f_path, 'r') as f: full_list = json.load(f)
        full_list.sort(key=lambda x: x['patch_id'])
        return full_list[rank::total_gpus]

    train_slice = load_and_slice("train.json")
    val_slice = load_and_slice("val.json") 

    ft = split_dir / f"train_worker_{rank}.json"
    fv = split_dir / f"val_worker_{rank}.json"
    
    with open(ft, 'w') as f: json.dump(train_slice, f, indent=4)
    with open(fv, 'w') as f: json.dump(val_slice, f, indent=4)
    
    return ft, fv, len(train_slice)

def train_worker(args):
    device = torch.device('cuda:0')
    rank = args.rank
    
    target_name = f"{args.target}_GPU_{rank}" 
    out_dir = PROJECT_ROOT / "outputs" / target_name
    save_dir = out_dir / "checkpoints"
    img_dir = out_dir / "images"
    log_dir = out_dir / "tensorboard" 
    for d in [save_dir, img_dir, log_dir]: d.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(log_dir))

    splits_dir = ROOT_DATA_DIR / args.target / "8_dataset_split" / "splits_json"
    if not splits_dir.exists():
        sys.exit(f"❌ Splits non trovati in {splits_dir}. Esegui Modello_2.")

    json_train, json_val, num_samples = create_partitioned_json(splits_dir, rank, 1)
    
    # --- CONFIGURAZIONE PERFORMANCE ESTREMA (H200) ---
    # Target: Saturare 141GB VRAM e CPU 100%
    BATCH_SIZE = 3      # Aumentato da 2 a 3 (+50% carico VRAM)
    ACCUM_STEPS = 20     # 3 * 11 = 33 Batch Effettivo
    
    LOG_INTERVAL = 10    
    IMAGE_LOG_INTERVAL = 1 
    CHECKPOINT_INTERVAL = 1
    TOTAL_EPOCHS = 150
    LR = 4e-4           
    
    train_ds = AstronomicalDataset(json_train, base_path=PROJECT_ROOT, augment=True)
    val_ds = AstronomicalDataset(json_val, base_path=PROJECT_ROOT, augment=False)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        # CPU TUNING:
        num_workers=16,    # Raddoppiato (era 8) per alzare il carico CPU > 30%
        pin_memory=True,
        prefetch_factor=4, # Bufferizziamo più dati per non far attendere la GPU
        persistent_workers=True # Mantiene i worker vivi tra le epoche
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8)

    model = HybridSuperResolutionModel(smoothing='balanced', device=device, output_size=512).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4, betas=(0.9, 0.99))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-7)
    criterion = CombinedLoss().to(device)
    
    scaler = torch.amp.GradScaler('cuda')

    print(f"🔥 Configurazione H200 (Saturated): Batch={BATCH_SIZE}, Workers=16, Prefetch=4")
    print(f"📊 Training Samples: {len(train_ds)} | Validation Samples: {len(val_ds)}")

    for epoch in range(TOTAL_EPOCHS):
        current_epoch = epoch + 1
        
        # === FASE DI TRAINING ===
        model.train()
        acc_loss_total = 0.0
        optimizer.zero_grad()
        
        # Barra Verde per Training
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                          desc=f"Epoca {current_epoch}/{TOTAL_EPOCHS} [Train]", 
                          leave=True, ncols=120, colour='green')
        
        for i, batch in train_pbar:
            lr = batch['lr'].to(device, non_blocking=True)
            hr = batch['hr'].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                pred = model(lr)
                loss, loss_dict = criterion(pred, hr)
                loss_scaled = loss / ACCUM_STEPS
            
            scaler.scale(loss_scaled).backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            loss_val = loss.item()
            acc_loss_total += loss_val
            train_pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step()

        # === FASE DI ANALISI ===
        if epoch % LOG_INTERVAL == 0:
            writer.add_scalar('Train/Loss', acc_loss_total / len(train_loader), epoch)
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)
            
            model.eval()
            metrics = Metrics()
            
            # Barra Ciano per Analisi
            val_pbar = tqdm(val_loader, total=len(val_loader), 
                            desc=f"Epoca {current_epoch} [Analisi]", 
                            leave=True, ncols=120, colour='cyan')
            
            with torch.no_grad():
                for v_batch in val_pbar:
                    v_lr = v_batch['lr'].to(device)
                    v_hr = v_batch['hr'].to(device)
                    
                    with torch.amp.autocast('cuda'):
                        v_pred = model(v_lr)
                        
                    metrics.update(v_pred.float(), v_hr.float())
            
            res = metrics.compute()
            tqdm.write(f"📊 RISULTATI ANALISI: PSNR={res['psnr']:.2f} | SSIM={res['ssim']:.4f}")
            
            writer.add_scalar('Val/PSNR', res['psnr'], epoch)
            writer.add_scalar('Val/SSIM', res['ssim'], epoch)
            
            if epoch % IMAGE_LOG_INTERVAL == 0:
                v_lr_up = torch.nn.functional.interpolate(v_lr, size=(512,512))
                preview = torch.cat((v_lr_up, v_pred, v_hr), dim=3).clamp(0,1)
                vutils.save_image(preview, img_dir / f"epoch_{epoch}.png")

        if epoch % CHECKPOINT_INTERVAL == 0:
            torch.save(model.state_dict(), save_dir / "best_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--rank', type=int, required=True)
    args = parser.parse_args()
    train_worker(args)