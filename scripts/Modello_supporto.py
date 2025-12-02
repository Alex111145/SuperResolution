"""
TRAINING INDEPENDENT WORKER
File: scripts/Modello_supporto.py
Versione: 128->512 | Full Logging | TensorBoard Images
"""

import os
import warnings

# Pulizia Log
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="astropy.io.fits")

# Ottimizzazioni Sistema
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
import cv2
cv2.setNumThreads(0)

import argparse
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from torch.cuda.amp import GradScaler 
from torch.utils.tensorboard import SummaryWriter 

# Path Setup
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
ROOT_DATA_DIR = PROJECT_ROOT / "data"

try:
    from src.architecture import HybridSuperResolutionModel
    from src.dataset import AstronomicalDataset
    from src.losses import CombinedLoss
    from src.metrics import Metrics
except ImportError:
    sys.exit("❌ Errore import src. Lancia da Modello_3.py")

def partition_data_from_folders(splits_root, rank, total_gpus):
    def scan_folder(split_name):
        d = splits_root / split_name
        data = []
        if not d.exists(): return []
        
        all_dirs = []
        for entry in os.scandir(d):
            if entry.is_dir() and entry.name.startswith("pair_"):
                all_dirs.append(entry)
        
        all_dirs.sort(key=lambda x: x.name)
        my_dirs = all_dirs[rank::total_gpus]
        
        for entry in my_dirs:
            p_path = Path(entry.path)
            lr = p_path / "observatory.fits"
            hr = p_path / "hubble.fits"
            
            if lr.exists() and hr.exists():
                data.append({
                    "patch_id": entry.name,
                    "ground_path": str(lr.resolve()),
                    "hubble_path": str(hr.resolve())
                })
        return data

    train_list = scan_folder("train")
    val_list = scan_folder("val")

    out_train = splits_root / f"temp_train_worker_{rank}.json"
    out_val = splits_root / f"temp_val_worker_{rank}.json"

    with open(out_train, 'w') as f: json.dump(train_list, f, indent=4)
    with open(out_val, 'w') as f: json.dump(val_list, f, indent=4)

    return out_train, out_val, len(train_list)

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

    splits_dir = ROOT_DATA_DIR / args.target / "6_patches_final" / "splits"
    if not splits_dir.exists():
        sys.exit(f"❌ ERRORE: Cartella splits non trovata.")

    json_train, json_val, num_samples = partition_data_from_folders(splits_dir, rank, args.total_gpus)
    
    # --- CONFIGURAZIONE 128 -> 512 ---
    BATCH_SIZE = 2      
    ACCUM_STEPS = 16    
    LOG_INTERVAL = 2         # Grafici ogni 2 epoche
    IMAGE_LOG_INTERVAL = 5   # Foto ogni 5 epoche
    CHECKPOINT_INTERVAL = 10
    
    MODEL_OUTPUT_SIZE = 512  # Target Hubble

    print(f"📦 GPU {rank}: Training {num_samples} img. LR:128 -> HR:{MODEL_OUTPUT_SIZE}")

    train_ds = AstronomicalDataset(json_train, base_path=PROJECT_ROOT, augment=True)
    val_ds = AstronomicalDataset(json_val, base_path=PROJECT_ROOT, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = HybridSuperResolutionModel(
        smoothing='balanced', 
        device=device, 
        output_size=MODEL_OUTPUT_SIZE
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    criterion = CombinedLoss().to(device)
    scaler = torch.amp.GradScaler('cuda')

    TOTAL_EPOCHS = 150 
    
    for epoch in range(TOTAL_EPOCHS):
        model.train()
        
        acc_loss_total = 0.0
        acc_loss_l1 = 0.0
        
        pbar = tqdm(train_loader, 
                    desc=f"GPU {rank} [Ep {epoch}/{TOTAL_EPOCHS}]", 
                    position=rank, 
                    leave=True,
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt} loss={postfix}]")

        for i, batch in enumerate(pbar):
            lr = batch['lr'].to(device, non_blocking=True) # 128x128
            hr = batch['hr'].to(device, non_blocking=True) # 512x512
            
            with torch.amp.autocast('cuda'):
                pred = model(lr) # Output atteso: 512x512
                
                # Safety Net: Se il modello non esce a 512, interpoliamo
                if pred.shape[-1] != hr.shape[-1]:
                    pred = F.interpolate(pred, size=hr.shape[-2:], mode='bilinear', align_corners=False)
                
                loss, loss_dict = criterion(pred, hr)
                loss_scaled = loss / ACCUM_STEPS
            
            scaler.scale(loss_scaled).backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            acc_loss_total += loss.item()
            acc_loss_l1 += loss_dict.get('l1', torch.tensor(0)).item()
            pbar.set_postfix_str(f"{loss.item():.4f}")

        # --- VALIDATION ---
        if epoch % LOG_INTERVAL == 0:
            steps = len(train_loader)
            writer.add_scalar('Train/Total_Loss', acc_loss_total / steps if steps > 0 else 0, epoch)

            model.eval()
            
            # Accumulatori per i grafici validazione
            val_total = 0.0
            val_l1 = 0.0
            val_astro = 0.0
            val_perc = 0.0
            
            metrics = Metrics()
            preview_img = None

            with torch.no_grad():
                for j, v_batch in enumerate(val_loader):
                    v_lr = v_batch['lr'].to(device) # 128
                    v_hr = v_batch['hr'].to(device) # 512
                    
                    with torch.amp.autocast('cuda'):
                        v_pred = model(v_lr) # 512
                        
                        if v_pred.shape[-1] != v_hr.shape[-1]:
                             v_pred = F.interpolate(v_pred, size=v_hr.shape[-2:], mode='bilinear', align_corners=False)
                        
                        # Ottieni TUTTE le componenti della Loss
                        v_loss, v_loss_dict = criterion(v_pred, v_hr)
                    
                    # Somma componenti
                    val_total += v_loss.item()
                    val_l1 += v_loss_dict.get('l1', torch.tensor(0)).item()
                    val_astro += v_loss_dict.get('astro', torch.tensor(0)).item()
                    val_perc += v_loss_dict.get('perceptual', torch.tensor(0)).item()
                    
                    metrics.update(v_pred.float(), v_hr.float())

                    # Genera Preview (Solo prima immagine)
                    if j == 0 and epoch % IMAGE_LOG_INTERVAL == 0:
                        # Upscale LR (128->512) Nearest Neighbor per vedere i pixel originali
                        v_lr_up = F.interpolate(v_lr, size=(MODEL_OUTPUT_SIZE, MODEL_OUTPUT_SIZE), mode='nearest')
                        # Normalizza per visualizzazione (clamp 0-1)
                        preview_img = torch.cat((v_lr_up, v_pred, v_hr), dim=3).clamp(0, 1)

            # Calcolo Medie
            n_val = len(val_loader) if len(val_loader) > 0 else 1
            res = metrics.compute()

            # --- SCRITTURA GRAFICI TENSORBOARD ---
            writer.add_scalar('Val/Loss_Total', val_total / n_val, epoch)
            writer.add_scalar('Val/Loss_L1', val_l1 / n_val, epoch)
            writer.add_scalar('Val/Loss_Astro', val_astro / n_val, epoch)
            writer.add_scalar('Val/Loss_Perceptual', val_perc / n_val, epoch)
            
            writer.add_scalar('Val/SSIM', res['ssim'], epoch)
            writer.add_scalar('Val/PSNR', res['psnr'], epoch)

            # --- SCRITTURA IMMAGINI ---
            if preview_img is not None:
                 # 1. Su Disco (PNG)
                 vutils.save_image(preview_img, img_dir / f"val_ep{epoch:03d}.png", normalize=False)
                 # 2. Su TensorBoard
                 writer.add_image('Val/Preview_LR_Pred_HR', preview_img[0], epoch)
            
            writer.flush()
            pbar.set_postfix_str(f"L_Tr:{acc_loss_total/steps:.3f} | L_Val:{val_total/n_val:.3f} | SSIM:{res['ssim']:.3f}")

        # CHECKPOINT
        if epoch % CHECKPOINT_INTERVAL == 0:
            torch.save(model.state_dict(), save_dir / f"epoch_{epoch:04d}.pth")
            torch.save(model.state_dict(), save_dir / "last.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--total_gpus', type=int, default=10)
    args = parser.parse_args()
    train_worker(args)