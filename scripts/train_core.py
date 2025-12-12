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

# --- CONFIGURAZIONE PERCORSI ---
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"
sys.path.insert(0, str(PROJECT_ROOT))

# --- IMPORTAZIONI MODULI (Presuppone che siano in src/) ---
try:
    # Questi moduli devono esistere nel percorso src/
    from src.architecture_train import TrainHybridModel
    from src.dataset import AstronomicalDataset
    from src.losses_train import TrainStarLoss
    from src.metrics_train import TrainMetrics
except ImportError as e:
    sys.exit(f"❌ Errore Import: Assicurati che 'src/' contenga i moduli necessari. Errore: {e}")

# === HYPERPARAMETERS ===
BATCH_SIZE = 3 	 	
ACCUM_STEPS = 1 	 	
LR = 2e-4 	 	 	
TOTAL_EPOCHS = 500 	
LOG_INTERVAL = 5 	
IMAGE_INTERVAL = 5 	

# --- UTILITY DDP ---

def setup():
    """Inizializzazione del processo distribuito DDP."""
    # Utilizza l'ambiente fornito da torchrun
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    """Pulizia del processo distribuito DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()

# --- FUNZIONE PRINCIPALE DDP WORKER (Modificata per Aggregazione) ---

def run_ddp_worker(args):
    """Logica di addestramento eseguita da ciascun rank DDP su un dataset aggregato."""
    setup()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    is_master = (rank == 0)

    # Gestione dei target multipli (es. "M101,M82")
    target_names = [t.strip() for t in args.target.split(',') if t.strip()]
    target_output_name = "_".join(target_names) # Nome aggregato per l'output

    # Percorsi output
    out_dir = PROJECT_ROOT / "outputs" / f"{target_output_name}_DDP"
    splits_dir_temp = out_dir / "temp_splits" # Directory temporanea per i JSON aggregati

    if is_master:
        save_dir = out_dir / "checkpoints"
        img_dir = out_dir / "images"
        log_dir = out_dir / "tensorboard"
        for d in [save_dir, img_dir, log_dir, splits_dir_temp]: d.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))
        print(f"🚀 [Master, Rank {rank}] Avvio Training su Dataset Aggregato: {target_output_name}")
        print(f"   Target inclusi: {target_names}")
    
    dist.barrier() # Sincronizza i processi dopo la creazione delle directory

    # --- 1. Aggregazione Dataset ---
    all_train_data = []
    all_val_data = []
    
    for target_name in target_names:
        splits_dir = ROOT_DATA_DIR / target_name / "8_dataset_split" / "splits_json"
        
        train_path = splits_dir / "train.json"
        val_path = splits_dir / "val.json"
        
        if not train_path.exists():
            if is_master: print(f"❌ File split non trovato per {target_name}. Salto.")
            continue
            
        with open(train_path) as f: all_train_data.extend(json.load(f))
        with open(val_path) as f: all_val_data.extend(json.load(f))

    if not all_train_data:
        if is_master: print("❌ Nessun dato di training valido dopo l'aggregazione.")
        cleanup()
        sys.exit(1)

    # 2. Creazione file JSON temporanei aggregati per DDP (specifici per il rank)
    ft_path = splits_dir_temp / f"temp_train_{target_output_name}_r{rank}.json"
    fv_path = splits_dir_temp / f"temp_val_{target_output_name}_r{rank}.json"
    
    # Salviamo i dati aggregati nel JSON temporaneo (necessario per DistributedSampler)
    with open(ft_path, 'w') as f: json.dump(all_train_data, f)
    with open(fv_path, 'w') as f: json.dump(all_val_data, f)

    # 3. Setup DataLoader
    train_ds = AstronomicalDataset(ft_path, base_path=PROJECT_ROOT, augment=True)
    val_ds = AstronomicalDataset(fv_path, base_path=PROJECT_ROOT, augment=False)
    
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, 
                              num_workers=8, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, sampler=val_sampler)

    # 4. Modello DDP e Ottimizzazione
    model = TrainHybridModel(smoothing='none', device=device).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-7)
    criterion = TrainStarLoss(vgg_weight=0.05).to(device)
    
    try: scaler = torch.amp.GradScaler('cuda')
    except: scaler = torch.cuda.amp.GradScaler()

    best_psnr = 0.0

    # === LOOP DI TRAINING ===
    for epoch in range(1, TOTAL_EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        acc_loss = 0.0
        optimizer.zero_grad()
        
        loader_iter = tqdm(train_loader, desc=f"Ep {epoch}", ncols=100, leave=False) if is_master else train_loader

        for i, batch in enumerate(loader_iter):
            lr_img = batch['lr'].to(device, non_blocking=True)
            hr_img = batch['hr'].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda') if hasattr(torch.amp, 'autocast') else torch.cuda.amp.autocast():
                pred = model(lr_img)
                loss, _ = criterion(pred, hr_img)
                loss = loss / ACCUM_STEPS 
            
            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            acc_loss += loss.item() * ACCUM_STEPS

        scheduler.step()
        
        if epoch % LOG_INTERVAL == 0:
            # Sincronizzazione per calcolare la media della loss globale
            dist.barrier()
            loss_tensor = torch.tensor(acc_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / (len(train_loader) * world_size)

            if is_master:
                writer.add_scalar('Train/Loss', avg_loss, epoch)
                
                # Valutazione
                model.eval()
                metrics = TrainMetrics()
                val_tqdm = tqdm(val_loader, desc="Val", ncols=50, leave=False)
                # ... (Logica di valutazione e salvataggio checkpoint/immagini)
                with torch.inference_mode(): 
                    for v_batch in val_tqdm:
                        v_lr = v_batch['lr'].to(device)
                        v_hr = v_batch['hr'].to(device)
                        with torch.amp.autocast('cuda') if hasattr(torch.amp, 'autocast') else torch.cuda.amp.autocast():
                            v_pred = model(v_lr)
                        metrics.update(v_pred.float(), v_hr.float())
                        
                res = metrics.compute()
                writer.add_scalar('Val/PSNR', res['psnr'], epoch)
                print(f" Ep {epoch:04d} | Loss: {avg_loss:.4f} | PSNR: {res['psnr']:.2f} dB")

                # Salvataggio checkpoint
                state = model.module.state_dict()
                if res['psnr'] > best_psnr:
                    best_psnr = res['psnr']
                    torch.save(state, save_dir / "best_train_model.pth")
                
                # Salvataggio immagine di debug
                if epoch % IMAGE_INTERVAL == 0:
                    v_lr_up = torch.nn.functional.interpolate(v_lr, size=(512,512), mode='nearest')
                    comp = torch.cat((v_lr_up, v_pred, v_hr), dim=3).clamp(0,1)
                    vutils.save_image(comp, img_dir / f"train_epoch_{epoch}.png")
                
                model.train() # Torna in modalità train

    # --- 5. Pulizia file temporanei ---
    try:
        if is_master:
            # Pulisce i file temporanei creati da tutti i rank
            for r in range(world_size):
                (splits_dir_temp / f"temp_train_{target_output_name}_r{r}.json").unlink(missing_ok=True)
                (splits_dir_temp / f"temp_val_{target_output_name}_r{r}.json").unlink(missing_ok=True)
            # Rimuove la cartella temporanea (se vuota)
            splits_dir_temp.rmdir() 
    except Exception as e:
        if is_master: print(f"⚠️ Errore pulizia file temporanei: {e}")
        
    cleanup()

# --- LOGICA DI CONTROLLO MULTI-TARGET (CONTROLLER) ---
# Questa sezione viene eseguita solo se lo script NON è lanciato da torchrun.

def select_target_directories(required_subdir='8_dataset_split'):
    """
    Menu interattivo per selezionare uno, più o tutti i target con gli split pronti.
    (Stessa logica del launcher, riproposta qui per la modalità CLI o come fallback)
    """
    all_subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs']]
    
    valid_subdirs = [
        d for d in all_subdirs 
        if (d / required_subdir / "splits_json" / "train.json").exists()
    ]

    if not valid_subdirs: 
        print(f"\n❌ Nessuna directory valida con split JSON pronti trovata in {ROOT_DATA_DIR}")
        return []

    print("\n--- 🧠 SELEZIONE DATASET PER TRAINING DDP ---")
    valid_subdirs.sort(key=lambda x: x.name)
    for i, d in enumerate(valid_subdirs): 
        print(f" 	[{i+1}] {d.name}")
        
    print(f" 	[A] TUTTE ({len(valid_subdirs)} in totale)")
    print(" (Es. '1,3,4' per selezione multipla, o 'A' per tutte)")
        
    try:
        raw_val = input("Scelta: ").strip().upper()
        if not raw_val: return []

        if raw_val == 'A':
            return valid_subdirs
        
        selected_indices = []
        for part in raw_val.split(','):
            try:
                idx = int(part) - 1
                if 0 <= idx < len(valid_subdirs):
                    selected_indices.append(idx)
            except ValueError:
                continue

        unique_indices = []
        for idx in selected_indices:
            if idx not in unique_indices:
                unique_indices.append(idx)
                
        return [valid_subdirs[i] for i in unique_indices]
        
    except Exception: 
        return []

def start_ddp_process_controller(target_names_str):
    """Avvia il training DDP per i target aggregati."""
    
    if not torch.cuda.is_available():
        print("❌ Nessuna GPU CUDA trovata. Impossibile avviare DDP.")
        return 1
    
    nproc_per_node = torch.cuda.device_count()
    print(f"🖥️ 	Trovate {nproc_per_node} GPU. Avvio DDP per target(s): {target_names_str}...")
    
    # Comando per avviare torchrun (torch.distributed.run)
    cmd = [
        sys.executable,
        '-m', 'torch.distributed.run',
        f'--nproc_per_node={nproc_per_node}',
        '--master_port=29500', 
        str(Path(__file__).resolve()), 	
        '--target', target_names_str 
    ]
    
    # Esegui il comando
    try:
        process = subprocess.Popen(cmd, cwd=PROJECT_ROOT)
        process.wait()
        return process.returncode
    except FileNotFoundError:
        print("❌ Errore: 'torchrun' non trovato. Assicurati che PyTorch e CUDA siano installati correttamente.")
        return 1
    except Exception as e:
        print(f"⚠️ Errore durante l'esecuzione DDP: {e}")
        return 1


def main_controller():
    """Funzione di controllo principale: decide se agire come Controller o Worker DDP."""
    
    # Se il processo è chiamato da torchrun (Worker DDP)
    if "RANK" in os.environ or "LOCAL_RANK" in os.environ:
        parser = argparse.ArgumentParser()
        parser.add_argument('--target', type=str, required=True, help="Nomi delle cartelle target in data/ separate da virgola (es. M101,M82)")
        args = parser.parse_args()
        run_ddp_worker(args)
        return

    # --- Se il processo è chiamato direttamente (Controller) ---
    
    # Gestione argomenti CLI o Menu interattivo
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'): 
        target_names = [t.strip() for t in sys.argv[1].split(',')]
        target_dirs = [ROOT_DATA_DIR / name for name in target_names if (ROOT_DATA_DIR / name).is_dir()]
    else:
        target_dirs = select_target_directories()
        
    if not target_dirs:
        print("❌ Nessun target selezionato. Esco.")
        return

    target_names_str = ",".join([d.name for d in target_dirs]) 
    
    print(f"\n✅ Target(s) selezionato(i): {target_names_str}")
    
    print(f"\n=======================================================")
    print(f"🧠 INIZIO TRAINING DDP PER DATASET AGGREGATO: {target_names_str}")
    print(f"=======================================================")
    
    # Lancio UNICO DDP con la lista di target aggregati
    exit_code = start_ddp_process_controller(target_names_str) 
    
    if exit_code != 0:
        print(f"🔥 ATTENZIONE: Training DDP fallito per l'aggregato. Codice di uscita: {exit_code}")
    else:
        print(f"🎉 Training DDP completato con successo per il modello generalizzato.")
            
    print("\n\n🏁 PROCESSO DI TRAINING COMPLETATO.")


if __name__ == "__main__":
    main_controller()