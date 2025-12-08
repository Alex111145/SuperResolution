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

try:
    from src.architecture import HybridSuperResolutionModel
    from src.dataset import AstronomicalDataset
    from src.losses import CombinedLoss
    from src.metrics import Metrics
except ImportError as e:
    print(f"\n❌ ERRORE IMPORT: {e}")
    sys.exit(1)

# ================= HYPERPARAMETERS =================
# BATCH SIZE: 4 Totali. 
# Su 2 GPU = 2 immagini per GPU (Molto sicuro).
BATCH_SIZE = 4        

ACCUM_STEPS = 32      
LR = 2e-4
TOTAL_EPOCHS = 300

LOG_INTERVAL = 5      
IMAGE_INTERVAL = 5    

# --- FIX STABILITÀ CUDA ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Disabilitiamo benchmark per evitare algoritmi conv che usano memoria out-of-bounds
torch.backends.cudnn.benchmark = False 
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True 

def train_worker(args):
    # 1. Pulizia Memoria
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    target_name = f"{args.target}_Worker_HAT_Medium"
    
    # 2. Setup Cartelle
    out_dir = PROJECT_ROOT / "outputs" / target_name
    save_dir = out_dir / "checkpoints"
    img_dir = out_dir / "images"
    log_dir = out_dir / "tensorboard"
    
    for d in [save_dir, img_dir, log_dir]: 
        d.mkdir(parents=True, exist_ok=True)

    # Inizializza Tensorboard
    writer = SummaryWriter(str(log_dir))

    # 3. Caricamento Dati
    splits_dir = PROJECT_ROOT / "data" / args.target / "8_dataset_split" / "splits_json"
    if not splits_dir.exists():
        sys.exit(f"❌ Splits non trovati in: {splits_dir}")
    
    with open(splits_dir / "train.json") as f: train_data = json.load(f)
    with open(splits_dir / "val.json") as f: val_data = json.load(f)
    
    ft_path = splits_dir / f"temp_train_{os.getpid()}.json"
    fv_path = splits_dir / f"temp_val_{os.getpid()}.json"
    
    with open(ft_path, 'w') as f: json.dump(train_data, f)
    with open(fv_path, 'w') as f: json.dump(val_data, f)

    train_ds = AstronomicalDataset(ft_path, base_path=PROJECT_ROOT, augment=True)
    val_ds = AstronomicalDataset(fv_path, base_path=PROJECT_ROOT, augment=False)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,          
        pin_memory=True, 
        prefetch_factor=2, 
        persistent_workers=True,
        drop_last=True          
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    print(f"🚀 Training Avviato su {device} (HAT-Medium)")
    print(f"   Config: Batch={BATCH_SIZE} | Accum={ACCUM_STEPS}")
    
    # Inizializzazione Modello
    model = HybridSuperResolutionModel(smoothing='balanced', device=device).to(device)
    
    # --- MULTI-GPU SETUP ---
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"   🔥 MULTI-GPU ATTIVO: {num_gpus} GPU in uso.")
        model = nn.DataParallel(model)
    else:
        print(f"   ⚠️ Single GPU Mode")
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-7)
    criterion = CombinedLoss().to(device)

    # 4. Configurazione AMP
    try:
        from torch.amp import GradScaler, autocast
        scaler = GradScaler('cuda')
        use_new_amp = True
        amp_device = 'cuda'
    except ImportError:
        scaler = torch.cuda.amp.GradScaler()
        use_new_amp = False
        amp_device = None

    best_psnr = 0.0

    # === TRAINING LOOP ===
    for epoch in range(1, TOTAL_EPOCHS + 1):
        model.train()
        
        acc_loss_total = 0.0
        acc_char = 0.0
        acc_astro = 0.0
        acc_perc = 0.0
        
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{TOTAL_EPOCHS}", ncols=120, colour='cyan') 
        
        for i, batch in enumerate(pbar):
            # FIX: .contiguous() previene errori di memoria con DataParallel scatter
            lr = batch['lr'].to(device, non_blocking=True).contiguous()
            hr = batch['hr'].to(device, non_blocking=True).contiguous()
            
            # --- Forward Pass ---
            if use_new_amp:
                with autocast(amp_device):
                    pred = model(lr)
                    loss, loss_dict = criterion(pred, hr) 
                    loss_scaled = loss / ACCUM_STEPS
            else:
                with torch.cuda.amp.autocast():
                    pred = model(lr)
                    loss, loss_dict = criterion(pred, hr) 
                    loss_scaled = loss / ACCUM_STEPS
            
            # --- Backward Pass ---
            scaler.scale(loss_scaled).backward()
            
            acc_loss_total += loss.item()
            
            def get_val(d, k):
                val = d.get(k, 0.0)
                return val.item() if hasattr(val, 'item') else val

            acc_char += get_val(loss_dict, 'char')
            acc_astro += get_val(loss_dict, 'astro')
            acc_perc += get_val(loss_dict, 'perceptual')

            # --- Optimizer Step ---
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) 
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Gestione ultimo batch
        if (i + 1) % ACCUM_STEPS != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        scheduler.step()
        
        # === LOGGING ===
        if epoch % LOG_INTERVAL == 0:
            steps = len(train_loader)
            current_lr = optimizer.param_groups[0]['lr']
            
            writer.add_scalar('Train/Loss_Total', acc_loss_total / steps, epoch)
            writer.add_scalar('Train/Loss_Components/Charbonnier', acc_char / steps, epoch)
            writer.add_scalar('Train/Loss_Components/Astro', acc_astro / steps, epoch)
            writer.add_scalar('Train/Loss_Components/Perceptual', acc_perc / steps, epoch)
            writer.add_scalar('Train/Learning_Rate', current_lr, epoch)
            
            model.eval()
            metrics = Metrics()
            
            with torch.inference_mode():
                for v_batch in val_loader:
                    v_lr = v_batch['lr'].to(device).contiguous()
                    v_hr = v_batch['hr'].to(device).contiguous()
                    
                    if use_new_amp:
                        with autocast(amp_device): v_pred = model(v_lr)
                    else:
                        with torch.cuda.amp.autocast(): v_pred = model(v_lr)
                        
                    metrics.update(v_pred.float(), v_hr.float())
            
            res = metrics.compute()
            writer.add_scalar('Val/PSNR', res['psnr'], epoch)
            writer.add_scalar('Val/SSIM', res['ssim'], epoch)
            writer.flush()
            
            tqdm.write(f"📊 EP {epoch} | PSNR: {res['psnr']:.2f} | SSIM: {res['ssim']:.4f}")

            if res['psnr'] > best_psnr:
                best_psnr = res['psnr']
                if isinstance(model, nn.DataParallel):
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                torch.save(state_dict, save_dir / "best_model.pth")
                tqdm.write("   🏆 Best Model Saved")
            
            if isinstance(model, nn.DataParallel):
                state_last = model.module.state_dict()
            else:
                state_last = model.state_dict()
            torch.save(state_last, save_dir / "last.pth")

            # Preview
            if epoch % IMAGE_INTERVAL == 0:
                v_lr_up = torch.nn.functional.interpolate(v_lr, size=(512,512), mode='nearest').cpu()
                v_pred_cpu = v_pred.detach().cpu()
                v_hr_cpu = v_hr.cpu()
                comp = torch.cat((v_lr_up, v_pred_cpu, v_hr_cpu), dim=3).clamp(0,1)
                writer.add_image('Preview', comp[0], epoch)
                vutils.save_image(comp, img_dir / f"epoch_{epoch}.png")
            
            writer.flush()

    try:
        if ft_path.exists(): ft_path.unlink()
        if fv_path.exists(): fv_path.unlink()
    except: pass
    
    writer.close()
    print(f"\n✅ Training Completato. Output: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    args = parser.parse_args()
    
    try:
        train_worker(args)
    except Exception as e:
        print(f"\n❌ ERRORE WORKER: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)