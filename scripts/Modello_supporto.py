import os
import argparse
import sys
import json
import torch
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.architecture import HybridSuperResolutionModel
    from src.dataset import AstronomicalDataset
    from src.losses import CombinedLoss
    from src.metrics import Metrics
except ImportError:
    print("❌ Errore Import src.")
    sys.exit(1)

BATCH_SIZE = 3          
ACCUM_STEPS = 20        
LR = 4e-4
TOTAL_EPOCHS = 150
LOG_INTERVAL = 1        
IMAGE_INTERVAL = 1      

def train_worker(args):
    device = torch.device('cuda:0')
    target_name = f"{args.target}_GPU_0"
    
    out_dir = PROJECT_ROOT / "outputs" / target_name
    save_dir = out_dir / "checkpoints"
    img_dir = out_dir / "images"
    log_dir = out_dir / "tensorboard"
    for d in [save_dir, img_dir, log_dir]: d.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(log_dir))

    splits_dir = PROJECT_ROOT / "data" / args.target / "8_dataset_split" / "splits_json"
    if not splits_dir.exists():
        sys.exit("❌ Splits non trovati. Esegui Modello_2.py")
    
    with open(splits_dir / "train.json") as f: train_data = json.load(f)
    with open(splits_dir / "val.json") as f: val_data = json.load(f)
    
    ft_path = splits_dir / "temp_train.json"
    fv_path = splits_dir / "temp_val.json"
    with open(ft_path, 'w') as f: json.dump(train_data, f)
    with open(fv_path, 'w') as f: json.dump(val_data, f)

    train_ds = AstronomicalDataset(ft_path, base_path=PROJECT_ROOT, augment=True)
    val_ds = AstronomicalDataset(fv_path, base_path=PROJECT_ROOT, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=8, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    print(f"🚀 Training: {len(train_ds)} campioni.")
    model = HybridSuperResolutionModel(smoothing='balanced', device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-7)
    criterion = CombinedLoss().to(device)
    scaler = torch.amp.GradScaler('cuda')

    best_psnr = 0.0

    for epoch in range(1, TOTAL_EPOCHS + 1):
        model.train()
        
        acc_loss_total = 0.0
        acc_char = 0.0
        acc_astro = 0.0
        acc_perc = 0.0
        
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{TOTAL_EPOCHS}", ncols=120, colour='green')
        
        for i, batch in enumerate(pbar):
            lr = batch['lr'].to(device, non_blocking=True)
            hr = batch['hr'].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                pred = model(lr)
                loss, loss_dict = criterion(pred, hr) 
                loss_scaled = loss / ACCUM_STEPS
            
            scaler.scale(loss_scaled).backward()
            
            acc_loss_total += loss.item()
            acc_char += loss_dict.get('char', 0).item() if isinstance(loss_dict.get('char'), torch.Tensor) else 0
            acc_astro += loss_dict.get('astro', 0).item() if isinstance(loss_dict.get('astro'), torch.Tensor) else 0
            acc_perc += loss_dict.get('perceptual', 0).item() if isinstance(loss_dict.get('perceptual'), torch.Tensor) else 0

            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        
        if epoch % LOG_INTERVAL == 0:
            steps = len(train_loader)
            current_lr = optimizer.param_groups[0]['lr']
            
            writer.add_scalar('Train/Loss_Total', acc_loss_total / steps, epoch)
            writer.add_scalar('Train/Loss_Charbonnier', acc_char / steps, epoch)
            writer.add_scalar('Train/Loss_Astro', acc_astro / steps, epoch)
            writer.add_scalar('Train/Loss_Perceptual', acc_perc / steps, epoch)
            writer.add_scalar('Train/Learning_Rate', current_lr, epoch)
            
            model.eval()
            metrics = Metrics()
            
            with torch.no_grad():
                for v_batch in val_loader:
                    v_lr = v_batch['lr'].to(device)
                    v_hr = v_batch['hr'].to(device)
                    with torch.amp.autocast('cuda'):
                        v_pred = model(v_lr)
                    
                    metrics.update(v_pred.float(), v_hr.float())
            
            res = metrics.compute()
            
            writer.add_scalar('Val/PSNR', res['psnr'], epoch)
            writer.add_scalar('Val/SSIM', res['ssim'], epoch)
            
            tqdm.write(f"📊 EP {epoch} | PSNR: {res['psnr']:.2f} | Astro Loss: {(acc_astro/steps):.4f}")

            if res['psnr'] > best_psnr:
                best_psnr = res['psnr']
                torch.save(model.state_dict(), save_dir / "best_model.pth")
            torch.save(model.state_dict(), save_dir / "last.pth")

            if epoch % IMAGE_INTERVAL == 0:
                v_lr_up = torch.nn.functional.interpolate(v_lr, size=(512,512))
                comp = torch.cat((v_lr_up, v_pred, v_hr), dim=3).clamp(0,1)
                writer.add_image('Preview', comp[0], epoch)
                vutils.save_image(comp, img_dir / f"epoch_{epoch}.png")
            
            writer.flush()

    ft_path.unlink(missing_ok=True)
    fv_path.unlink(missing_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    args = parser.parse_args()
    train_worker(args)