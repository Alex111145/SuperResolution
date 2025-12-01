"""
LAUNCHER PARALLELO
"""
import subprocess
import sys
import time
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = str(HERE.parent)

TARGET_NAME = "M42" # <--- CAMBIARE QUI
NUM_GPUS = 10 

print(f"🚀 LANCIO TRAINING SU {TARGET_NAME} (TIFF DATA)")

env_base = os.environ.copy()
env_base["PYTHONPATH"] = f"{PROJECT_ROOT}:{env_base.get('PYTHONPATH', '')}"

processes = []
for i in range(NUM_GPUS):
    print(f" ⚡ Worker {i}...")
    env = env_base.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(i)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    cmd = [sys.executable, "Modello_supporto.py", "--target", TARGET_NAME, "--rank", str(i)]
    p = subprocess.Popen(cmd, env=env)
    processes.append(p)
    time.sleep(1)

try:
    for p in processes: p.wait()
except KeyboardInterrupt:
    for p in processes: p.terminate()