"""
LAUNCHER (Il Capo)
Questo file DEVE chiamarsi: scripts/Modello_3.py
"""
import subprocess
import sys
import time
import os
from pathlib import Path

# FIX: Calcola la root del progetto
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = str(HERE.parent)

# CONFIGURA QUI IL TARGET
TARGET_NAME = "M33" 

print("="*60)
print(f"🚀 LANCIO DI 10 TRAINING PARALLELI SU {TARGET_NAME}")
print(f"   (Root aggiunta a PYTHONPATH: {PROJECT_ROOT})")
print("="*60)

processes = []
NUM_GPUS = 10 

# FIX IMPORT: Crea l'ambiente base con PYTHONPATH configurato
env_base = os.environ.copy()
if 'PYTHONPATH' in env_base:
    env_base["PYTHONPATH"] = f"{PROJECT_ROOT}:{env_base['PYTHONPATH']}"
else:
    env_base["PYTHONPATH"] = PROJECT_ROOT

for i in range(NUM_GPUS):
    print(f"   ⚡ Avvio Worker {i} su GPU {i}...")
    
    env = env_base.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(i)
    
    # FIX CUDA OOM: Aggiunge la configurazione di allocazione
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Chiama lo script Worker
    cmd = [
        sys.executable, 
        "Modello_supporto.py", 
        "--target", TARGET_NAME,
        "--rank", str(i)
    ]
    
    # Avvia in background
    p = subprocess.Popen(cmd, env=env)
    processes.append(p)
    time.sleep(1) 

print(f"\n✅ Tutti i {NUM_GPUS} worker sono partiti!")
print("   Per fermarli tutti, premi CTRL+C qui.")

try:
    for p in processes:
        p.wait()
except KeyboardInterrupt:
    print("\n🛑 STOP GENERALE. Arresto di tutti i processi...")
    for p in processes:
        p.terminate()