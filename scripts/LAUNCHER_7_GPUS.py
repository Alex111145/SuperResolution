"""
LAUNCHER (Il Capo)
Questo file DEVE chiamarsi: scripts/LAUNCHER_10_GPUS.py
"""
import subprocess
import sys
import time
import os

# CONFIGURA QUI IL TARGET
TARGET_NAME = "M33" 

print("="*60)
print(f"🚀 LANCIO DI 10 TRAINING PARALLELI SU {TARGET_NAME}")
print("="*60)

processes = []
NUM_GPUS = 10 # Nuovo valore

for i in range(NUM_GPUS):
    print(f"   ⚡ Avvio Worker {i} su GPU {i}...")
    
    # Assegna una specifica GPU al processo
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(i)
    
    # Chiama lo script Worker
    cmd = [
        sys.executable, 
        "scripts/Modello_4_train_independent.py", 
        "--target", TARGET_NAME,
        "--rank", str(i)
    ]
    
    # Avvia in background
    p = subprocess.Popen(cmd, env=env)
    processes.append(p)
    time.sleep(1) 

print(f"\n✅ Tutti i {NUM_GPUS} worker sono partiti!")
print("   Per fermarli tutti, premi CTRL+C qui.")

try:
    for p in processes:
        p.wait()
except KeyboardInterrupt:
    print("\n🛑 STOP GENERALE. Arresto di tutti i processi...")
    for p in processes:
        p.terminate()