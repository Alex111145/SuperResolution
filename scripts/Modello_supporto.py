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

# ================= HYPERPARAMETERS SWINIR =================
# Valori aggiornati in base alle tue specifiche (Batch=6, Accum=12)
BATCH_SIZE = 6         
ACCUM_STEPS = 12  

LR = 2e-4  
TOTAL_EPOCHS = 300 

LOG_INTERVAL = 2      
IMAGE_INTERVAL = 10     

# Ottimizzazione Memoria
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True 

def train_worker(args):
    # 1. Setup Device
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device('cuda:0')
        print(f"✅ Training su GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ ERRORE: Nessuna GPU rilevata.")
        sys.exit(1)

    # NOME CARTELLA SPECIFICO PER SWINIR
    target_name = f"{args.target}_Worker_SwinIR_Medium"
    
    # 2. Setup Cartelle
    out_dir = PROJECT_ROOT / "outputs" / target_name
    save_dir = out_dir / "checkpoints"
    img_dir = out_dir / "images"
    log_dir = out_dir / "tensorboard"
    
    for d in [save_dir, img_dir, log_dir]: 
        d.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(log_dir))

    # 3. Caricamento Dataset
    splits_dir = PROJECT_ROOT / "data" / args.target / "8_dataset_split" / "splits_json"
    if not splits_dir.exists():
        sys.exit(f"❌ Splits non trovati in: {splits_dir}")
    
    try:
        with open(splits_dir / "train.json") as f: train_data = json.load(f)
        with open(splits_dir / "val.json") as f: val_data = json.load(f)
    except FileNotFoundError:
        sys.exit("❌ File JSON non trovati. Esegui Modello_2.py")

    # File temp per compatibilità loader
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
        persistent_workers=True,
        drop_last=True          
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    print(f"🚀 Training SwinIR Avviato")
    print(f"   Config: Batch={BATCH_SIZE} | Accum={ACCUM_STEPS} (Effective={BATCH_SIZE*ACCUM_STEPS}) | Epochs={TOTAL_EPOCHS}")
    print(f"   Dataset: {len(train_ds)} train | {len(val_ds)} val")

    # 4. Inizializzazione Modello
    model = HybridSuperResolutionModel(device=device).to(device)
    
    # Optimizer AdamW (Raccomandato per SwinIR)
    optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-7)
    criterion = CombinedLoss().to(device)

    scaler = torch.amp.GradScaler('cuda')
    best_psnr = 0.0

    # === TRAINING LOOP ===
    for epoch in range(1, TOTAL_EPOCHS + 1):
        model.train()
        
        # >>> INIZIALIZZAZIONE NUOVE VARIABILI PER LOG COMPLETO <<<
        acc_loss_total = 0.0
        acc_char = 0.0
        acc_astro = 0.0
        acc_perc = 0.0
        
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{TOTAL_EPOCHS} [SwinIR]", ncols=120, colour='cyan') 
        
        for i, batch in enumerate(pbar):
            lr_img = batch['lr'].to(device, non_blocking=True)
            hr_img = batch['hr'].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                pred = model(lr_img)
                loss, loss_dict = criterion(pred, hr_img) # Otteniamo il dizionario delle loss
                loss_scaled = loss / ACCUM_STEPS
            
            scaler.scale(loss_scaled).backward()
            
            # >>> ACCUMULO DELLE LOSS COMPONENTI <<<
            acc_loss_total += loss.item()
            acc_char += loss_dict.get('char', torch.tensor(0.0)).item()
            acc_astro += loss_dict.get('astro', torch.tensor(0.0)).item()
            acc_perc += loss_dict.get('perceptual', torch.tensor(0.0)).item()
            
            # Step accumulato
            if (i + 1) % ACCUM_STEPS == 0:
                # Gradient Clipping: FONDAMENTALE per SwinIR
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Gestione ultimo batch
        if (i + 1) % ACCUM_STEPS != 0:
            if (i + 1) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
            optimizer.zero_grad()

        scheduler.step()
        
        # === LOGGING COMPLETO ===
        if epoch % LOG_INTERVAL == 0:
            # Calcoliamo la media delle loss accumulate
            avg_loss = acc_loss_total / len(train_loader)
            
            writer.add_scalar('Train/Loss_Total', avg_loss, epoch)
            writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # >>> NUOVI GRAFICI AGGIUNTI <<<
            writer.add_scalar('Train/Loss_Components/Charbonnier', acc_char / len(train_loader), epoch)
            writer.add_scalar('Train/Loss_Components/Astro', acc_astro / len(train_loader), epoch)
            writer.add_scalar('Train/Loss_Components/Perceptual', acc_perc / len(train_loader), epoch)
            
            # Validazione (PSNR/SSIM)
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
            
            tqdm.write(f"📊 EP {epoch} | PSNR: {res['psnr']:.2f} dB | SSIM: {res['ssim']:.4f}")

            if res['psnr'] > best_psnr:
                best_psnr = res['psnr']
                torch.save(model.state_dict(), save_dir / "best_model.pth")
            
            torch.save(model.state_dict(), save_dir / "last.pth")

            # Preview Immagini
            if epoch % IMAGE_INTERVAL == 0:
                v_lr_up = torch.nn.functional.interpolate(v_lr, size=(512,512), mode='nearest').cpu()
                comp = torch.cat((v_lr_up, v_pred.cpu(), v_hr.cpu()), dim=3).clamp(0,1)
                writer.add_image('Visual_Check', comp[0], epoch)
                vutils.save_image(comp, img_dir / f"epoch_{epoch}.png")
            
            writer.flush()

    try:
        if ft_path.exists(): ft_path.unlink()
        if fv_path.exists(): fv_path.unlink()
    except: pass
    
    writer.close()
    print(f"\n✅ Training SwinIR Completato. Modello in: {out_dir}")

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
