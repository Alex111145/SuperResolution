"""
TRAINING WORKER (TIFF EDITION - DEBUG VERSION)
Rimosso il try-except sugli import per vedere il vero errore.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
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
import traceback # Importante per il debug

# Setup Path
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# --- MODIFICA DEBUG: Import diretti senza try-except silenzioso ---
print(f"🔧 [Worker] Loading modules from {PROJECT_ROOT}...")
try:
    from src.architecture import HybridSuperResolutionModel
    from src.dataset import AstronomicalDataset
    from src.losses import CombinedLoss
    from src.metrics import Metrics
    print("   ✅ Moduli src importati correttamente.")
except Exception as e:
    print(f"\n❌ ERRORE CRITICO NELL'IMPORTARE SRC:")
    print("-" * 60)
    traceback.print_exc() # Stampa l'errore esatto (Sintassi, Indentazione, ecc.)
    print("-" * 60)
    sys.exit(1)
# ---------------------------------------------------------------

def create_partitioned_json(split_dir, rank, total_gpus):
    """Gestione partizionamento dataset per multi-GPU simulato."""
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

    json_train, json_val, num_samples = create_partitioned_json(splits_dir, rank, 10)
    
    # --- IPERPARAMETRI ---
    BATCH_SIZE = 2      
    ACCUM_STEPS = 16    
    LOG_INTERVAL = 3    
    IMAGE_LOG_INTERVAL = 10
    CHECKPOINT_INTERVAL = 10
    TOTAL_EPOCHS = 150
    LR = 4e-4           # Aggressive LR
    
    # Dataset
    train_ds = AstronomicalDataset(json_train, base_path=PROJECT_ROOT, augment=True)
    val_ds = AstronomicalDataset(json_val, base_path=PROJECT_ROOT, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    # Modello & Ottimizzatore
    model = HybridSuperResolutionModel(smoothing='balanced', device=device, output_size=512).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4, betas=(0.9, 0.99))
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-7)
    
    criterion = CombinedLoss().to(device)
    scaler = torch.amp.GradScaler('cuda')

    pbar = tqdm(range(TOTAL_EPOCHS), desc=f"GPU {rank}", position=rank, leave=True)

    for epoch in pbar:
        model.train()
        
        acc_loss_total = 0.0
        acc_loss_char = 0.0
        acc_loss_l1_raw = 0.0
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
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Statistiche
            acc_loss_total += loss.item()
            acc_loss_char += loss_dict.get('char', torch.tensor(0)).item()
            acc_loss_l1_raw += loss_dict.get('l1_raw', torch.tensor(0)).item()
            acc_loss_astro += loss_dict.get('astro', torch.tensor(0)).item()
            acc_loss_perc += loss_dict.get('perceptual', torch.tensor(0)).item()

            if i % 10 == 0:
                pbar.set_postfix_str(f"L:{loss.item():.4f} | LR:{optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step()

        if epoch % LOG_INTERVAL == 0:
            steps = len(train_loader)
            writer.add_scalar('Train_Main/Total_Loss', acc_loss_total / steps, epoch)
            writer.add_scalar('Train_Components/Charbonnier_Optim', acc_loss_char / steps, epoch)
            writer.add_scalar('Train_Components/L1_Pixel_Metric', acc_loss_l1_raw / steps, epoch)
            writer.add_scalar('Train_Components/Astro_Star', acc_loss_astro / steps, epoch)
            writer.add_scalar('Train_Components/Perceptual', acc_loss_perc / steps, epoch)
            writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # Validation
            model.eval()
            val_loss = 0.0
            metrics = Metrics()
            preview_img = None
            
            with torch.no_grad():
                for j, v_batch in enumerate(val_loader):
                    v_lr = v_batch['lr'].to(device)
                    v_hr = v_batch['hr'].to(device)
                    
                    with torch.amp.autocast('cuda'):
                        v_pred = model(v_lr)
                        v_l, _ = criterion(v_pred, v_hr)
                    
                    val_loss += v_l.item()
                    metrics.update(v_pred.float(), v_hr.float())
                    
                    if j == 0 and epoch % IMAGE_LOG_INTERVAL == 0:
                        v_lr_up = torch.nn.functional.interpolate(v_lr, size=(512,512), mode='nearest')
                        preview_img = torch.cat((v_lr_up, v_pred, v_hr), dim=3)

            res = metrics.compute()
            writer.add_scalar('Validation/Loss', val_loss / len(val_loader), epoch)
            writer.add_scalar('Validation/PSNR', res['psnr'], epoch)
            writer.add_scalar('Validation/SSIM', res['ssim'], epoch)
            
            if preview_img is not None:
                preview_img = preview_img.clamp(0, 1)
                writer.add_image('Preview', preview_img[0], epoch)
                vutils.save_image(preview_img, img_dir / f"ep_{epoch}.png")
            
            writer.flush()

        if epoch % CHECKPOINT_INTERVAL == 0:
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            torch.save(model.state_dict(), save_dir / "last.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--rank', type=int, required=True)
    args = parser.parse_args()
    train_worker(args)