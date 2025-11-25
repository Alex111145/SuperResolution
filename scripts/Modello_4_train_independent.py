"""
TRAINING INDEPENDENT WORKER (TENSORBOARD LOG EVERY 2 EPOCHS)
Batch Size: 2 | Workers: 1
Update: Logga Training e Validation su TensorBoard ogni 2 epoche.
"""

import os
# Ottimizzazioni di sistema
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import cv2
cv2.setNumThreads(0)

import argparse
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from torch.cuda.amp import GradScaler 
from torch.utils.tensorboard import SummaryWriter 

# Configurazione Path
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
    sys.exit("❌ Errore import src.")

def create_partitioned_json(split_dir, rank, total_gpus):
    """Crea un JSON contenente SOLO 1/7 dei dati totali."""
    def scan_and_split(split_name):
        d = split_dir / split_name
        data = []
        if not d.exists(): return []
        
        all_files = []
        for entry in os.scandir(d):
            if entry.is_dir() and entry.name.startswith("pair_"):
                all_files.append(entry)
        
        all_files.sort(key=lambda x: x.name)
        my_files = all_files[rank::total_gpus]
        
        for entry in my_files:
            lr = (d / entry.name / "observatory.fits").resolve()
            hr = (d / entry.name / "hubble.fits").resolve()
            if lr.exists() and hr.exists():
                data.append({
                    "patch_id": entry.name, 
                    "ground_path": str(lr), 
                    "hubble_path": str(hr)
                })
        return data

    train_list = scan_and_split("train")
    val_list = scan_and_split("val")

    ft = split_dir / f"train_worker_{rank}.json"
    fv = split_dir / f"val_worker_{rank}.json"
    
    with open(ft, 'w') as f: json.dump(train_list, f, indent=4)
    with open(fv, 'w') as f: json.dump(val_list, f, indent=4)
    
    return ft, fv, len(train_list)

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
    json_train, json_val, num_samples = create_partitioned_json(splits_dir, rank, 7)
    
    # CONFIGURAZIONE STABILE
    BATCH_SIZE = 2  
    ACCUM_STEPS = 8 
    LOG_INTERVAL = 2 # <--- AGGIORNA TENSORBOARD OGNI 2 EPOCHE
    
    train_ds = AstronomicalDataset(json_train, base_path=PROJECT_ROOT, augment=True)
    val_ds = AstronomicalDataset(json_val, base_path=PROJECT_ROOT, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = HybridSuperResolutionModel(smoothing='balanced', device=device, output_size=512).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    criterion = CombinedLoss().to(device)
    scaler = torch.amp.GradScaler('cuda')

    TOTAL_EPOCHS = 150 
    
    pbar = tqdm(range(TOTAL_EPOCHS), desc=f"GPU {rank}", position=rank, leave=True,
                bar_format="{desc}: Ep {n_fmt} | {postfix}")

    for epoch in pbar:
        model.train()
        
        acc_loss_total = 0.0
        acc_loss_l1 = 0.0
        acc_loss_astro = 0.0
        acc_loss_perc = 0.0
        
        optimizer.zero_grad()
        
        for i, batch in enumerate(train_loader):
            lr = batch['lr'].to(device, non_blocking=True)
            hr = batch['hr'].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                pred = model(lr)
                loss, loss_dict = criterion(pred, hr)
                loss_scaled = loss / ACCUM_STEPS
            
            scaler.scale(loss_scaled).backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            current_val = loss.item()
            acc_loss_total += current_val
            acc_loss_l1 += loss_dict['l1'].item()
            acc_loss_astro += loss_dict['astro'].item()
            acc_loss_perc += loss_dict['perceptual'].item()

            # Aggiorna la barra TQDM in tempo reale (solo testo, niente log su disco)
            if i % 10 == 0:
                img_processed = (i + 1) * BATCH_SIZE
                pbar.set_postfix_str(f"Img {img_processed}/{num_samples} | L: {current_val:.4f}")

        # --- LOGGING & VALIDATION (OGNI 2 EPOCHE) ---
        if epoch % LOG_INTERVAL == 0:
            
            # 1. Logga Training Loss
            steps = len(train_loader)
            avg_total = acc_loss_total / steps
            
            writer.add_scalar('Train_Main/Total_Loss', avg_total, epoch)
            writer.add_scalar('Train_Components/L1_Pixel', acc_loss_l1 / steps, epoch)
            writer.add_scalar('Train_Components/Astro_Star', acc_loss_astro / steps, epoch)
            writer.add_scalar('Train_Components/Perceptual_VGG', acc_loss_perc / steps, epoch)

            # 2. Esegui Validazione Completa
            model.eval()
            val_loss = 0
            metrics = Metrics()
            preview_img = None 

            with torch.no_grad():
                for j, v_batch in enumerate(val_loader):
                    v_lr = v_batch['lr'].to(device)
                    v_hr = v_batch['hr'].to(device)
                    
                    with torch.amp.autocast('cuda'):
                        v_pred = model(v_lr)
                        v_loss, _ = criterion(v_pred, v_hr)
                    
                    val_loss += v_loss.item()
                    metrics.update(v_pred.float(), v_hr.float())
                    
                    if j == 0:
                        v_lr_up = torch.nn.functional.interpolate(v_lr, size=(512,512), mode='nearest')
                        preview_img = torch.cat((v_lr_up, v_pred, v_hr), dim=3)

            res = metrics.compute()
            avg_val_loss = val_loss / len(val_loader)

            # 3. Logga Validazione su TensorBoard
            writer.add_scalar('Validation/Loss', avg_val_loss, epoch)
            writer.add_scalar('Validation/PSNR', res['psnr'], epoch)
            writer.add_scalar('Validation/SSIM_SRM', res['ssim'], epoch)
            
            # 4. Logga Immagini e Preview (ogni 10 epoche per non intasare disco, ma grafici ogni 2)
            if preview_img is not None and epoch % 10 == 0:
                preview_img = preview_img.clamp(0, 1)
                writer.add_image('Preview/Confronto', preview_img[0], epoch)
                vutils.save_image(preview_img, img_dir / "live_preview.png", normalize=False)

            # Forza scrittura su disco
            writer.flush()
            
            pbar.set_postfix_str(f"L:{avg_total:.3f}|V:{avg_val_loss:.3f}|SSIM:{res['ssim']:.3f}")

        if epoch % 50 == 0:
            torch.save(model.state_dict(), save_dir / "last.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--rank', type=int, required=True)
    args = parser.parse_args()
    train_worker(args)