"""
TRAINING WORKER (NVIDIA H200 OPTIMIZED)
Ampia VRAM + Tensor Cores = Max Performance.
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
# Abilita il benchmark: PyTorch testerà vari algoritmi convoluzionali
# all'inizio e sceglierà il più veloce per la tua H200.
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True # Usa TensorFloat-32 sulle schede Ampere/Hopper
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
    
    # --- CONFIGURAZIONE H200 (Extreme Performance) ---
    BATCH_SIZE = 24      # Con 141GB VRAM, possiamo spingere. Se OOM, scendi a 16.
    ACCUM_STEPS = 1      # Con batch così alti, spesso non serve accumulare.
    LOG_INTERVAL = 10    
    IMAGE_LOG_INTERVAL = 20
    CHECKPOINT_INTERVAL = 10
    TOTAL_EPOCHS = 150
    LR = 4e-4           
    
    # Dataset
    train_ds = AstronomicalDataset(json_train, base_path=PROJECT_ROOT, augment=True)
    val_ds = AstronomicalDataset(json_val, base_path=PROJECT_ROOT, augment=False)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=8, # H200 mangia dati veloce, servono workers
        pin_memory=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    # Modello
    model = HybridSuperResolutionModel(smoothing='balanced', device=device, output_size=512).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4, betas=(0.9, 0.99))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-7)
    criterion = CombinedLoss().to(device)
    
    # Mixed Precision Scaler (Fondamentale per H200)
    scaler = torch.cuda.amp.GradScaler()

    print(f"🔥 Configurazione H200: Batch={BATCH_SIZE}, AMP=ON, TF32=ON")

    pbar = tqdm(range(TOTAL_EPOCHS), desc=f"Epochs", position=0, leave=True)

    for epoch in pbar:
        model.train()
        acc_loss_total = 0.0
        optimizer.zero_grad()
        
        for i, batch in enumerate(train_loader):
            lr = batch['lr'].to(device, non_blocking=True)
            hr = batch['hr'].to(device, non_blocking=True)
            
            # Autocast: Usa FP16/BF16 dove possibile (Velocissimo su H200)
            with torch.cuda.amp.autocast():
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
            
            acc_loss_total += loss.item()
            
            if i % 5 == 0:
                pbar.set_postfix_str(f"Loss: {loss.item():.4f}")

        scheduler.step()

        if epoch % LOG_INTERVAL == 0:
            writer.add_scalar('Train/Loss', acc_loss_total / len(train_loader), epoch)
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)
            
            model.eval()
            metrics = Metrics()
            with torch.no_grad():
                for v_batch in val_loader:
                    v_lr = v_batch['lr'].to(device)
                    v_hr = v_batch['hr'].to(device)
                    
                    with torch.cuda.amp.autocast():
                        v_pred = model(v_lr)
                        
                    metrics.update(v_pred.float(), v_hr.float())
            
            res = metrics.compute()
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