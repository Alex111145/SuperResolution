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
import warnings

# --- SETUP BASE ---
warnings.filterwarnings("ignore")
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"
sys.path.insert(0, str(PROJECT_ROOT))

# --- IMPORT MODULI ---
try:
    from src.architecture_train import TrainHybridModel
    from src.dataset import AstronomicalDataset
    from src.losses_train import TrainStarLoss
    from src.metrics_train import TrainMetrics
except ImportError as e:
    sys.exit(f"❌ Errore Import: {e}")

# === HYPERPARAMETERS ===
BATCH_SIZE = 1      
ACCUM_STEPS = 4       # Accumulo gradienti
LR = 1e-4             # Learning Rate
TOTAL_EPOCHS = 500  
LOG_INTERVAL = 5    
IMAGE_INTERVAL = 10   

def setup():
    """Inizializza il gruppo di processi per DDP"""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

def train_worker():
    setup()
    
    # Info Processo
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    is_master = (rank == 0)

    # Parsing Argomenti
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True, help="Nome target")
    args = parser.parse_args()

    # --- GESTIONE MULTI-TARGET ---
    target_names = [t.strip() for t in args.target.split(',') if t.strip()]
    target_output_name = "_".join(target_names)

    # Setup Percorsi Output
    out_dir = PROJECT_ROOT / "outputs" / f"{target_output_name}_DDP"
    save_dir = out_dir / "checkpoints"
    img_dir = out_dir / "images"
    log_dir = out_dir / "tensorboard"
    splits_dir_temp = out_dir / "temp_splits"

    # File Checkpoint
    latest_ckpt_path = save_dir / "latest_checkpoint.pth"
    best_weights_path = save_dir / "best_train_model.pth"

    if is_master:
        for d in [save_dir, img_dir, log_dir, splits_dir_temp]: d.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))
        print(f"🚀 [Master] Training su: {target_output_name} | GPUs: {world_size}")

    dist.barrier() 

    # --- AGGREGAZIONE DATASET ---
    all_train_data = []
    all_val_data = []
    
    for t_name in target_names:
        s_dir = ROOT_DATA_DIR / t_name / "8_dataset_split" / "splits_json"
        try:
            with open(s_dir / "train.json") as f: all_train_data.extend(json.load(f))
            with open(s_dir / "val.json") as f: all_val_data.extend(json.load(f))
        except FileNotFoundError:
            if is_master: print(f"⚠️ Dati non trovati per {t_name}, salto.")

    # Scrittura JSON temporanei
    ft_path = splits_dir_temp / f"temp_train_r{rank}.json"
    fv_path = splits_dir_temp / f"temp_val_r{rank}.json"
    with open(ft_path, 'w') as f: json.dump(all_train_data, f)
    with open(fv_path, 'w') as f: json.dump(all_val_data, f)

    # --- DATALOADERS ---
    train_ds = AstronomicalDataset(ft_path, base_path=PROJECT_ROOT, augment=True)
    val_ds = AstronomicalDataset(fv_path, base_path=PROJECT_ROOT, augment=False)
    
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, 
                              num_workers=8, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, sampler=val_sampler)

    # --- MODELLO & OPTIMIZER ---
    model = TrainHybridModel(smoothing='none', device=device).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-7)
    scaler = torch.amp.GradScaler('cuda')
    criterion = TrainStarLoss(vgg_weight=0.05).to(device) 

    # ==========================================================================
    # --- LOGICA RESUME AUTOMATICA ---
    # ==========================================================================
    start_epoch = 1
    best_psnr = 0.0
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}

    # Controllo presenza checkpoint completo (Resume Standard)
    if latest_ckpt_path.exists():
        if is_master: print(f"🔄 Trovato checkpoint completo: {latest_ckpt_path.name}")
        try:
            checkpoint = torch.load(latest_ckpt_path, map_location=map_location)
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1
            best_psnr = checkpoint.get('best_psnr', 0.0)
            
            if is_master: print(f"✅ Resume automatico da Epoca {start_epoch} (Best PSNR: {best_psnr:.2f})")
        except Exception as e:
            if is_master: print(f"❌ Errore critico nel checkpoint: {e}. Si riparte da zero.")
    
    # Fallback: Se non c'è il resume completo, controlliamo se esistono pesi 'best' 
    # per fare un warm-start (ma ripartendo da epoca 1)
    elif best_weights_path.exists():
        if is_master: print("⚠️ Nessun resume completo, ma trovati pesi 'best'. Carico pesi e riparto da Epoca 1.")
        try:
            state_dict = torch.load(best_weights_path, map_location=map_location)
            model.module.load_state_dict(state_dict)
        except Exception as e:
            if is_master: print(f"❌ Errore caricamento pesi best: {e}")
            
    elif is_master:
        print("✨ Nessun checkpoint trovato. Inizio training da zero.")
    # ==========================================================================


    # === TRAINING LOOP ===
    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        acc_loss = 0.0
        optimizer.zero_grad()
        
        loader_iter = tqdm(train_loader, desc=f"Ep {epoch}", ncols=100, leave=False) if is_master else train_loader

        for i, batch in enumerate(loader_iter):
            lr_img = batch['lr'].to(device, non_blocking=True)
            hr_img = batch['hr'].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                pred = model(lr_img)
                loss, _ = criterion(pred, hr_img)
                loss = loss / ACCUM_STEPS 
            
            if torch.isnan(loss) or torch.isinf(loss):
                if is_master: print(f"⚠️ NaN Loss step {i}")
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
        
        # --- VALIDAZIONE ---
        if epoch % LOG_INTERVAL == 0 or epoch == TOTAL_EPOCHS:
            dist.barrier()
            
            loss_tensor = torch.tensor(acc_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / (len(train_loader) * world_size)

            model.eval()
            local_metrics = TrainMetrics()
            
            val_iter = tqdm(val_loader, desc="Val", ncols=50, leave=False) if is_master else val_loader

            with torch.inference_mode():
                for v_batch in val_iter:
                    v_lr = v_batch['lr'].to(device)
                    v_hr = v_batch['hr'].to(device)
                    with torch.amp.autocast('cuda'):
                        v_pred = model(v_lr)
                    v_pred = torch.nan_to_num(v_pred)
                    local_metrics.update(v_pred.float(), v_hr.float())
            
            total_psnr = torch.tensor(local_metrics.psnr, device=device)
            total_count = torch.tensor(local_metrics.count, device=device)
            dist.all_reduce(total_psnr, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
            global_psnr = total_psnr.item() / total_count.item()

            if is_master:
                writer.add_scalar('Train/Loss', avg_loss, epoch)
                writer.add_scalar('Val/PSNR', global_psnr, epoch)
                print(f" Ep {epoch:04d} | Loss: {avg_loss:.4f} | PSNR: {global_psnr:.2f} dB")

                if global_psnr > best_psnr:
                    best_psnr = global_psnr
                    print("   🏆 Nuovo Best PSNR!")
                    torch.save(model.module.state_dict(), save_dir / "best_train_model.pth")
                
                if epoch % IMAGE_INTERVAL == 0:
                    v_lr_up = torch.nn.functional.interpolate(v_lr, size=(512,512), mode='nearest')
                    comp = torch.cat((v_lr_up, v_pred, v_hr), dim=3).clamp(0,1)
                    vutils.save_image(comp, img_dir / f"train_epoch_{epoch}.png")

        # --- SALVATAGGIO CHECKPOINT AUTOMATICO (Ogni Epoca) ---
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

        dist.barrier()
        model.train()

    if is_master:
        try:
            if ft_path.exists(): ft_path.unlink()
            if fv_path.exists(): fv_path.unlink()
        except: pass
        
    cleanup()

if __name__ == "__main__":
    train_worker()
