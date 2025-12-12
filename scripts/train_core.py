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
import subprocess
import signal
import warnings

# --- 1. SOPPRESSIONE WARNING SPECIFICI ---
warnings.filterwarnings("ignore") # Ignora warning generici
# Filtra specificamente il warning sui gradient strides di DDP
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides") 

# --- CONFIGURAZIONE PERCORSI ---
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"
sys.path.insert(0, str(PROJECT_ROOT))

# --- IMPORTAZIONI MODULI ---
try:
    from src.architecture_train import TrainHybridModel
    from src.dataset import AstronomicalDataset
    from src.losses_train import TrainStarLoss
    from src.metrics_train import TrainMetrics
except ImportError as e:
    sys.exit(f"❌ Errore Import: Assicurati che 'src/' contenga i moduli necessari. Errore: {e}")

# === HYPERPARAMETERS ===
BATCH_SIZE = 1      
ACCUM_STEPS = 4     
LR = 2e-4           
TOTAL_EPOCHS = 500  
LOG_INTERVAL = 5    
IMAGE_INTERVAL = 10     

# --- UTILITY DDP ---

def setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.backends.cudnn.benchmark = True

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def run_ddp_worker(args):
    setup()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    is_master = (rank == 0)

    target_names = [t.strip() for t in args.target.split(',') if t.strip()]
    target_output_name = "_".join(target_names)

    out_dir = PROJECT_ROOT / "outputs" / f"{target_output_name}_DDP"
    splits_dir_temp = out_dir / "temp_splits"

    if is_master:
        save_dir = out_dir / "checkpoints"
        img_dir = out_dir / "images"
        log_dir = out_dir / "tensorboard"
        for d in [save_dir, img_dir, log_dir, splits_dir_temp]: d.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))
        print(f"🚀 [Master] Training DDP: {target_output_name} | GPUs: {world_size}")

    dist.barrier()

    # --- 1. Aggregazione Dataset ---
    all_train_data = []
    all_val_data = []
    
    for target_name in target_names:
        splits_dir = ROOT_DATA_DIR / target_name / "8_dataset_split" / "splits_json"
        with open(splits_dir / "train.json") as f: all_train_data.extend(json.load(f))
        with open(splits_dir / "val.json") as f: all_val_data.extend(json.load(f))

    # JSON temporanei
    ft_path = splits_dir_temp / f"temp_train_{target_output_name}_r{rank}.json"
    fv_path = splits_dir_temp / f"temp_val_{target_output_name}_r{rank}.json"
    with open(ft_path, 'w') as f: json.dump(all_train_data, f)
    with open(fv_path, 'w') as f: json.dump(all_val_data, f)

    # Dataset & Loader
    train_ds = AstronomicalDataset(ft_path, base_path=PROJECT_ROOT, augment=True)
    val_ds = AstronomicalDataset(fv_path, base_path=PROJECT_ROOT, augment=False)
    
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, 
                              num_workers=4, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, sampler=val_sampler)

    # Modello
    model = TrainHybridModel(smoothing='none', device=device).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-7)
    criterion = TrainStarLoss(vgg_weight=0.1).to(device)
    
    scaler = torch.amp.GradScaler('cuda')

    best_psnr = 0.0

    # === LOOP DI TRAINING ===
    for epoch in range(1, TOTAL_EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        acc_loss = 0.0
        
        # 'set_to_none=True' aiuta a evitare problemi di strides
        optimizer.zero_grad(set_to_none=True) 
        
        loader_iter = tqdm(train_loader, desc=f"Ep {epoch}", ncols=100, leave=False) if is_master else train_loader

        for i, batch in enumerate(loader_iter):
            lr_img = batch['lr'].to(device, non_blocking=True)
            hr_img = batch['hr'].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                pred = model(lr_img)
                loss, _ = criterion(pred, hr_img)
                loss = loss / ACCUM_STEPS 
            
            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            acc_loss += loss.item() * ACCUM_STEPS

        scheduler.step()
        
        # --- VALIDAZIONE E SINCRONIZZAZIONE ---
        if epoch % LOG_INTERVAL == 0:
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
                    
                    local_metrics.update(v_pred.float(), v_hr.float())
            
            total_psnr_tensor = torch.tensor(local_metrics.psnr, device=device)
            total_count_tensor = torch.tensor(local_metrics.count, device=device)
            
            dist.all_reduce(total_psnr_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_count_tensor, op=dist.ReduceOp.SUM)
            
            global_avg_psnr = total_psnr_tensor.item() / total_count_tensor.item()

            if is_master:
                writer.add_scalar('Train/Loss', avg_loss, epoch)
                writer.add_scalar('Val/PSNR', global_avg_psnr, epoch)
                print(f" Ep {epoch:04d} | Loss: {avg_loss:.4f} | PSNR: {global_avg_psnr:.2f} dB")

                state = model.module.state_dict()
                if global_avg_psnr > best_psnr:
                    best_psnr = global_avg_psnr
                    print("   🏆 Nuovo Best PSNR!")
                    torch.save(state, save_dir / "best_train_model.pth")
                
                if epoch % IMAGE_INTERVAL == 0:
                    v_lr_up = torch.nn.functional.interpolate(v_lr, size=(512,512), mode='nearest')
                    comp = torch.cat((v_lr_up, v_pred, v_hr), dim=3).clamp(0,1)
                    vutils.save_image(comp, img_dir / f"train_epoch_{epoch}.png")

            dist.barrier()
            model.train()

    cleanup()

# --- MAIN CONTROLLER ---
def select_target_directories(required_subdir='8_dataset_split'):
    all_subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs']]
    valid_subdirs = [d for d in all_subdirs if (d / required_subdir / "splits_json" / "train.json").exists()]
    if not valid_subdirs: return []
    valid_subdirs.sort(key=lambda x: x.name)
    for i, d in enumerate(valid_subdirs): print(f" [{i+1}] {d.name}")
    print(f" [A] TUTTE ({len(valid_subdirs)})")
    try:
        raw = input("Scelta: ").strip().upper()
        if raw == 'A': return valid_subdirs
        idxs = [int(x)-1 for x in raw.split(',') if x.isdigit()]
        return [valid_subdirs[i] for i in idxs if 0 <= i < len(valid_subdirs)]
    except: return []

def main_controller():
    if "RANK" in os.environ:
        parser = argparse.ArgumentParser()
        parser.add_argument('--target', type=str, required=True)
        args = parser.parse_args()
        run_ddp_worker(args)
        return

    target_dirs = select_target_directories()
    if not target_dirs: return
    target_names = ",".join([d.name for d in target_dirs])
    
    nproc = torch.cuda.device_count()
    cmd = [sys.executable, '-m', 'torch.distributed.run', f'--nproc_per_node={nproc}', '--master_port=29500', str(Path(__file__).resolve()), '--target', target_names]
    subprocess.run(cmd)

if __name__ == "__main__":
    main_controller()