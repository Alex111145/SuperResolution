"""
TRAINING WORKER (TIFF EDITION)
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
    """Legge il JSON completo (train.json) e ne prende una fetta per questa GPU."""
    
    def load_and_slice(json_name):
        f_path = split_dir / json_name
        if not f_path.exists(): return []
        with open(f_path, 'r') as f: full_list = json.load(f)
        
        # Sort per determinismo
        full_list.sort(key=lambda x: x['patch_id'])
        # Prendi 1 elemento ogni N (stride)
        my_slice = full_list[rank::total_gpus]
        return my_slice

    train_slice = load_and_slice("train.json")
    val_slice = load_and_slice("val.json") # Ogni GPU valida sul suo pezzetto

    # Salva JSON temporanei per il Dataset
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

    # Path ai JSON creati da Modello_2
    splits_dir = ROOT_DATA_DIR / args.target / "8_dataset_split" / "splits_json"
    
    if not splits_dir.exists():
        sys.exit(f"❌ Splits non trovati in {splits_dir}. Esegui Modello_2.")

    json_train, json_val, num_samples = create_partitioned_json(splits_dir, rank, 10)
    
    BATCH_SIZE = 2      
    ACCUM_STEPS = 16    
    LOG_INTERVAL = 3    
    IMAGE_LOG_INTERVAL = 10
    CHECKPOINT_INTERVAL = 10
    
    train_ds = AstronomicalDataset(json_train, base_path=PROJECT_ROOT, augment=True)
    val_ds = AstronomicalDataset(json_val, base_path=PROJECT_ROOT, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    model = HybridSuperResolutionModel(smoothing='balanced', device=device, output_size=512).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    criterion = CombinedLoss().to(device)
    scaler = torch.amp.GradScaler('cuda')

    TOTAL_EPOCHS = 150 
    
    pbar = tqdm(range(TOTAL_EPOCHS), desc=f"GPU {rank}", position=rank, leave=True)

    for epoch in pbar:
        model.train()
        acc_loss = 0.0
        
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
            
            acc_loss += loss.item()
            if i % 10 == 0:
                pbar.set_postfix_str(f"Loss: {loss.item():.4f}")

        # LOGGING
        if epoch % LOG_INTERVAL == 0:
            writer.add_scalar('Train/Loss', acc_loss / len(train_loader), epoch)
            
            # Validazione
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
                        l, _ = criterion(v_pred, v_hr)
                    val_loss += l.item()
                    metrics.update(v_pred.float(), v_hr.float())
                    
                    if j == 0 and epoch % IMAGE_LOG_INTERVAL == 0:
                        v_lr_up = torch.nn.functional.interpolate(v_lr, size=(512,512), mode='nearest')
                        preview_img = torch.cat((v_lr_up, v_pred, v_hr), dim=3)

            res = metrics.compute()
            writer.add_scalar('Val/PSNR', res['psnr'], epoch)
            writer.add_scalar('Val/SSIM', res['ssim'], epoch)
            
            if preview_img is not None:
                preview_img = preview_img.clamp(0, 1)
                writer.add_image('Preview', preview_img[0], epoch)
                vutils.save_image(preview_img, img_dir / f"ep_{epoch}.png")

        if epoch % CHECKPOINT_INTERVAL == 0:
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            torch.save(model.state_dict(), save_dir / "last.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--rank', type=int, required=True)
    args = parser.parse_args()
    train_worker(args)