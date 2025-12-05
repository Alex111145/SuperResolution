#!/usr/bin/env python3
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

# ================= HYPERPARAMETERS OTTIMIZZATI =================
BATCH_SIZE = 6         
ACCUM_STEPS = 12       # Batch Size Effettiva = 72
LR = 2e-4              # Learning Rate più aggressivo per sbloccare il training
TOTAL_EPOCHS = 300 

# >>> SCHEDULE DELLE LOSS (2 FASI) <<<
# Fase 1 (0-30): Solo L1 Loss per imparare la geometria e intensità base
# Fase 2 (30+): Attivazione Astro e Perceptual per i dettagli
EPOCH_ENABLE_ADVANCED_LOSS = 30

LOG_INTERVAL = 1      
IMAGE_INTERVAL = 5     

# Ottimizzazioni Memoria
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True 

def train_worker(args):
    # Check ambiente
    print(f"🔧 Python Executable: {sys.executable}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device('cuda:0')
        print(f"✅ Training su GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ ERRORE: Nessuna GPU rilevata.")
        sys.exit(1)

    target_name = f"{args.target}_SwinIR_GlobalNorm" 
    out_dir = PROJECT_ROOT / "outputs" / target_name
    save_dir = out_dir / "checkpoints"
    img_dir = out_dir / "images"
    log_dir = out_dir / "tensorboard"
    
    for d in [save_dir, img_dir, log_dir]: 
        d.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(log_dir))

    # Caricamento Dataset JSON
    splits_dir = PROJECT_ROOT / "data" / args.target / "8_dataset_split" / "splits_json"
    if not splits_dir.exists():
        sys.exit(f"❌ Splits non trovati in: {splits_dir}\n   Esegui Modello_2.py!")
    
    try:
        with open(splits_dir / "train.json") as f: train_data = json.load(f)
        with open(splits_dir / "val.json") as f: val_data = json.load(f)
    except:
        sys.exit("❌ Errore lettura JSON.")

    # Creazione file temp per il DataLoader
    ft_path = splits_dir / f"temp_train_{os.getpid()}.json"
    fv_path = splits_dir / f"temp_val_{os.getpid()}.json"
    with open(ft_path, 'w') as f: json.dump(train_data, f)
    with open(fv_path, 'w') as f: json.dump(val_data, f)

    train_ds = AstronomicalDataset(ft_path, base_path=PROJECT_ROOT, augment=True)
    val_ds = AstronomicalDataset(fv_path, base_path=PROJECT_ROOT, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    print(f"🚀 Avvio Training: {len(train_ds)} coppie")
    print(f"   Config: LR={LR} | Switch Loss @ Epoca {EPOCH_ENABLE_ADVANCED_LOSS}")

    # Modello
    model = HybridSuperResolutionModel(device=device).to(device)
    
    # Loss Iniziale: SOLO Charbonnier (L1)
    # Pesi: (Charbonnier, Perceptual, Astro)
    criterion = CombinedLoss(l1_w=1.0, perceptual_w=0.0, astro_w=0.0).to(device)
    # Forza i pesi iniziali per sicurezza
    criterion.weights = (1.0, 0.0, 0.0) 
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-7)

    scaler = torch.amp.GradScaler('cuda')
    best_psnr = 0.0

    for epoch in range(1, TOTAL_EPOCHS + 1):
        
        # --- ATTIVAZIONE LOSS AVANZATE ---
        if epoch == EPOCH_ENABLE_ADVANCED_LOSS:
            # Attiva Perceptual (0.05) e Astro (0.1)
            criterion.weights = (1.0, 0.05, 0.1)
            tqdm.write(f"\n💡 EPOCH {epoch}: ATTIVAZIONE Perceptual & Astro Loss!")

        model.train()
        acc_loss = 0.0
        
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{TOTAL_EPOCHS}", ncols=110, colour='green') 
        
        for i, batch in enumerate(pbar):
            lr_img = batch['lr'].to(device, non_blocking=True)
            hr_img = batch['hr'].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                pred = model(lr_img)
                loss, _ = criterion(pred, hr_img) 
                loss_scaled = loss / ACCUM_STEPS
            
            scaler.scale(loss_scaled).backward()
            acc_loss += loss.item()
            
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Gestione ultimo batch residuo
        if (i + 1) % ACCUM_STEPS != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        scheduler.step()
        
        # Validazione
        if epoch % LOG_INTERVAL == 0:
            writer.add_scalar('Train/Loss', acc_loss/len(train_loader), epoch)
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)
            
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
            
            tqdm.write(f"📊 Val | PSNR: {res['psnr']:.2f} | SSIM: {res['ssim']:.4f}")

            if res['psnr'] > best_psnr:
                best_psnr = res['psnr']
                torch.save(model.state_dict(), save_dir / "best_model.pth")
            
            torch.save(model.state_dict(), save_dir / "last.pth")

            if epoch % IMAGE_INTERVAL == 0:
                # Upscale bicubico per confronto visivo
                # FIX: Aggiunto .cpu() per spostare v_lr_up dalla GPU alla RAM
                v_lr_up = torch.nn.functional.interpolate(v_lr, size=(512,512), mode='bicubic').cpu()
                
                # Ora tutti e tre i tensori sono su CPU
                comp = torch.cat((v_lr_up, v_pred.cpu(), v_hr.cpu()), dim=3).clamp(0,1)
                vutils.save_image(comp, img_dir / f"epoch_{epoch}.png")
            
            writer.flush()

    try:
        os.remove(ft_path)
        os.remove(fv_path)
    except: pass
    
    writer.close()
    print(f"\n✅ Training Completato: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    args = parser.parse_args()
    try:
        train_worker(args)
    except Exception as e:
        print(f"\n❌ ERRORE WORKER: {e}")
        sys.exit(1)