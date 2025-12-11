#!/usr/bin/env python3
import os
import sys
import subprocess
import torch
from pathlib import Path

# Configurazione Base
PROJECT_ROOT = Path("/root/SuperResolution")
DATA_DIR = PROJECT_ROOT / "data"
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "train_core.py"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_available_targets():
    if not DATA_DIR.exists():
        return []
    return [d.name for d in DATA_DIR.iterdir() if d.is_dir()]

def get_available_gpus():
    count = torch.cuda.device_count()
    gpus = []
    for i in range(count):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**2
        gpus.append(f"[{i}] {name} ({int(mem)} MB)")
    return gpus

def main():
    clear_screen()
    print("==========================================")
    print("      🚀 SUPER RESOLUTION LAUNCHER        ")
    print("==========================================\n")

    # 1. SELEZIONE TARGET
    targets = get_available_targets()
    if not targets:
        print("❌ Nessun target trovato in /data!")
        sys.exit(1)
    
    print("📂 Target disponibili:")
    for idx, t in enumerate(targets):
        print(f"   [{idx}] {t}")
    
    while True:
        try:
            sel = input("\n👉 Seleziona numero target: ")
            t_idx = int(sel)
            if 0 <= t_idx < len(targets):
                selected_target = targets[t_idx]
                break
            print("❌ Numero non valido.")
        except ValueError:
            print("❌ Inserisci un numero.")

    print(f"\n✅ Target selezionato: {selected_target}\n")

    # 2. SELEZIONE GPU
    gpus = get_available_gpus()
    print("🎮 GPU Disponibili:")
    for g in gpus:
        print(f"   {g}")
    
    print("\nOpzioni:")
    print("   [a] Usa TUTTE le GPU")
    print("   [0,1] Inserisci ID separati da virgola (es. 0,2)")
    
    gpu_env_string = ""
    nproc = 0
    
    while True:
        sel = input("\n👉 Scelta GPU: ").strip().lower()
        if sel == 'a':
            gpu_env_string = ",".join([str(i) for i in range(len(gpus))])
            nproc = len(gpus)
            break
        else:
            try:
                # Validazione input manuale
                ids = [x.strip() for x in sel.split(',')]
                # Verifica che siano numeri validi
                valid = all(x.isdigit() and int(x) < len(gpus) for x in ids)
                if valid and len(ids) > 0:
                    gpu_env_string = ",".join(ids)
                    nproc = len(ids)
                    break
                else:
                    print("❌ ID GPU non validi.")
            except:
                print("❌ Formato non valido.")

    print(f"\n✅ GPU Selezionate: {gpu_env_string} (Totale processi: {nproc})")
    
    # 3. LANCIO TORCHRUN
    print("\n==========================================")
    print("⚡ Avvio Training DDP...")
    print("==========================================\n")

    # Costruzione comando
    # NCCL_P2P_DISABLE=1 è CRUCIALE per le A40 su RunPod per evitare freeze
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_env_string
    env["NCCL_P2P_DISABLE"] = "1"
    env["NCCL_IB_DISABLE"] = "1"
    env["OMP_NUM_THREADS"] = "4"

    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "--master_port=29500",
        str(SCRIPT_PATH),
        "--target", selected_target
    ]

    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Errore durante il training: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Training interrotto dall'utente.")

if __name__ == "__main__":
    main()