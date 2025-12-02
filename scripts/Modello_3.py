"""
LAUNCHER (Il Capo)
File: scripts/Modello_3.py
"""
import subprocess
import sys
import time
import os
from pathlib import Path

# --- CONFIGURAZIONE ---
TARGET_NAME = "M42" 
NUM_GPUS = 10 
WORKER_SCRIPT_NAME = "Modello_supporto.py"

# Path Assoluti
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = str(HERE.parent)

print("="*60)
print(f"🚀 LANCIO DI {NUM_GPUS} TRAINING PARALLELI SU {TARGET_NAME}")
print(f"   (Root progetto: {PROJECT_ROOT})")
print("="*60)

processes = []

# Configura PYTHONPATH per includere la root del progetto
env_base = os.environ.copy()
if 'PYTHONPATH' in env_base:
    env_base["PYTHONPATH"] = f"{PROJECT_ROOT}:{env_base['PYTHONPATH']}"
else:
    env_base["PYTHONPATH"] = PROJECT_ROOT

for i in range(NUM_GPUS):
    print(f"   ⚡ Avvio Worker {i} su GPU {i}...")
    
    env = env_base.copy()
    # Assegna una specifica GPU al processo
    env["CUDA_VISIBLE_DEVICES"] = str(i)
    # Ottimizzazione memoria per evitare frammentazione
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    cmd = [
        sys.executable, 
        os.path.join(HERE, WORKER_SCRIPT_NAME), 
        "--target", TARGET_NAME,
        "--rank", str(i),
        "--total_gpus", str(NUM_GPUS)
    ]
    
    # Avvia senza bloccare
    p = subprocess.Popen(cmd, env=env)
    processes.append(p)
    time.sleep(2) # Delay per evitare picchi CPU all'avvio

print(f"\n✅ Tutti i {NUM_GPUS} worker sono partiti!")
print("   Premi CTRL+C per fermare tutto.")

try:
    for p in processes:
        p.wait()
except KeyboardInterrupt:
    print("\n🛑 STOP GENERALE. Arresto processi...")
    for p in processes:
        p.terminate()