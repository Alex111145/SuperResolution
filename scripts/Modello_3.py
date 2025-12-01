"""
LAUNCHER H200 SXM EDITION
Clean & Fast startup for NVIDIA GPUs.
"""
import subprocess
import sys
import time
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = str(HERE.parent)

TARGET_NAME = "M42" # <--- CAMBIA SE NECESSARIO
NUM_GPUS = 1        # H200 singola (o cambia se ne usi di più)

print(f"🚀 LANCIO TRAINING SU {TARGET_NAME} (NVIDIA H200 SXM)")

# Setup environment pulito
env = os.environ.copy()
env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"

# Rimuoviamo eventuali rimasugli di config AMD se presenti nel sistema
for k in list(env.keys()):
    if "MIOPEN" in k or "HIP" in k:
        del env[k]

# NVIDIA Optimizations
# Consente a CUDA di allocare memoria in modo più aggressivo
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

processes = []
for i in range(NUM_GPUS):
    print(f" ⚡ Worker {i} avviato...")
    env_worker = env.copy()
    env_worker["CUDA_VISIBLE_DEVICES"] = str(i)
    
    cmd = [sys.executable, "Modello_supporto.py", "--target", TARGET_NAME, "--rank", str(i)]
    p = subprocess.Popen(cmd, env=env_worker)
    processes.append(p)

try:
    for p in processes: p.wait()
except KeyboardInterrupt:
    for p in processes: p.terminate()