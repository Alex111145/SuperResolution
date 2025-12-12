#!/usr/bin/env python3
import os
import argparse
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm

# Setup percorsi
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.architecture_train import TrainHybridModel
    from src.dataset import AstronomicalDataset
    from src.losses_train import TrainStarLoss
    from src.metrics_train import TrainMetrics
except ImportError as e:
    sys.exit(f"❌ Errore Import: {e}")

# === HYPERPARAMETERS ===
BATCH_SIZE = 1      
ACCUM_STEPS = 4       # Gradiente accumulato per stabilità
LR = 1e-4             # LR ridotto per sicurezza
TOTAL_EPOCHS = 500    
LOG_INTERVAL = 5      
IMAGE_INTERVAL = 10   

def setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

def train_worker():
    setup()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    is_master = (rank == 0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    args = parser.parse_args()

    # Paths Output
    out_dir = PROJECT_ROOT / "outputs" / f"{args.target}_DDP"
    save_dir = out_dir / "checkpoints"
    img_dir = out_dir / "images"
    log_dir = out_dir / "tensorboard"
    
    # Path del checkpoint per il RESUME
    latest_ckpt_path = save_dir / "latest_checkpoint.pth"

    if is_master:
        for d in [save_dir, img_dir, log_dir]: d.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))
        print(f"🚀 [Master] Avvio Training su Target: {args.target}")

    # Dataset Setup
    splits_dir = PROJECT_ROOT / "data" / args.target / "8_dataset_split" / "splits_json"
    
    with open(splits_dir / "train.json") as f: train_data = json.load(f)
    with open(splits_dir / "val.json") as f: val_data = json.load(f)
    
    ft_path = splits_dir / f"temp_train_r{rank}.json"
    fv_path = splits_dir / f"temp_val_r{rank}.json"
    with open(ft_path, 'w') as f: json.dump(train_data, f)
    with open(fv_path, 'w') as f: json.dump(val_data, f)

    train_ds = AstronomicalDataset(ft_path, base_path=PROJECT_ROOT, augment=True)
    val_ds = AstronomicalDataset(fv_path, base_path=PROJECT_ROOT, augment=False)
    
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, 
                              num_workers=8, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, sampler=val_sampler)

    # --- MODEL & OPTIMIZER SETUP ---
    model = TrainHybridModel(smoothing='none', device=device).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-7)
    scaler = torch.amp.GradScaler('cuda')
    criterion = TrainStarLoss(vgg_weight=0.05).to(device)

    # --- RESUME LOGIC (CRUCIALE) ---
    start_epoch = 1
    best_psnr = 0.0

    # Controlliamo se esiste un checkpoint precedente
    # Usiamo map_location per assicurarci che carichi sulla GPU giusta
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    
    if latest_ckpt_path.exists():
        if is_master: print(f"🔄 Trovato checkpoint: {latest_ckpt_path}. Caricamento in corso...")
        # Carica il checkpoint
        checkpoint = torch.load(latest_ckpt_path, map_location=map_location)
        
        # Ripristina stati
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Ripristina variabili loop
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint['best_psnr']
        
        if is_master: print(f"✅ Ripresa training dall'epoca {start_epoch} (Best PSNR: {best_psnr:.2f})")
    else:
        if is_master: print("✨ Nessun checkpoint trovato. Inizio training da zero.")

    # --- TRAINING LOOP ---
    # Nota: range parte da start_epoch
    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        acc_loss = 0.0
        optimizer.zero_grad()
        
        loader_iter = tqdm(train_loader, desc=f"Ep {epoch}", ncols=100, leave=False) if is_master else train_loader

        for i, batch in enumerate(loader_iter):
            lr_img = batch['lr'].to(device, non_blocking=True)
            hr_img = batch['hr'].to(device, non_blocking=True)
            
            # --- TRAINING STEP (CON PROTEZIONI) ---
            with torch.amp.autocast('cuda'):
                pred = model(lr_img)
                loss, _ = criterion(pred, hr_img)
                loss = loss / ACCUM_STEPS 
            
            # Check NaN
            if torch.isnan(loss) or torch.isinf(loss):
                if is_master: print(f"⚠️ Warning: NaN Loss detected at step {i}. Skipping.")
                optimizer.zero_grad() 
                continue

            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            if not torch.isnan(loss):
                acc_loss += loss.item() * ACCUM_STEPS

        scheduler.step()
        
        # --- VALIDATION, LOGGING & SAVING ---
        if epoch % LOG_INTERVAL == 0 or epoch == TOTAL_EPOCHS:
            dist.barrier()
            loss_tensor = torch.tensor(acc_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            denom = (len(train_loader) * dist.get_world_size())
            avg_loss = loss_tensor.item() / denom if denom > 0 else 0

            if is_master:
                writer.add_scalar('Train/Loss', avg_loss, epoch)
                
                model.eval()
                metrics = TrainMetrics()
                with torch.inference_mode(): 
                    for v_batch in val_loader:
                        v_lr = v_batch['lr'].to(device)
                        v_hr = v_batch['hr'].to(device)
                        with torch.amp.autocast('cuda'):
                            v_pred = model(v_lr)
                        v_pred = torch.nan_to_num(v_pred)
                        metrics.update(v_pred.float(), v_hr.float())
                
                res = metrics.compute()
                psnr_val = res['psnr'] if not torch.isnan(torch.tensor(res['psnr'])) else 0.0
                
                writer.add_scalar('Val/PSNR', psnr_val, epoch)
                print(f" Ep {epoch:04d} | Loss: {avg_loss:.4f} | PSNR: {psnr_val:.2f} dB")

                # Salva Best Model (solo pesi)
                if psnr_val > best_psnr:
                    best_psnr = psnr_val
                    torch.save(model.module.state_dict(), save_dir / "best_train_model.pth")
                
                # Visualizzazione
                if epoch % IMAGE_INTERVAL == 0:
                    v_lr_up = torch.nn.functional.interpolate(v_lr, size=(512,512), mode='nearest')
                    comp = torch.cat((v_lr_up, v_pred, v_hr), dim=3).clamp(0,1)
                    vutils.save_image(comp, img_dir / f"train_epoch_{epoch}.png")

        # --- SALVATAGGIO CHECKPOINT (OGNI EPOCA) ---
        # Salviamo sempre "latest" per poter riprendere
        if is_master:
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_psnr': best_psnr
            }
            torch.save(checkpoint_dict, latest_ckpt_path)

    try:
        if ft_path.exists(): ft_path.unlink()
        if fv_path.exists(): fv_path.unlink()
    except: pass
    cleanup()

if __name__ == "__main__":
    train_worker()