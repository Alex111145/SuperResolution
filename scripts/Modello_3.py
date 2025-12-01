import subprocess
import sys
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = str(HERE.parent)

TARGET_NAME = "M42"

print(f"🚀 LANCIO TRAINING: {TARGET_NAME}")

env = os.environ.copy()
env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print(f" ⚡ Avvio worker...")
env["CUDA_VISIBLE_DEVICES"] = "0"

cmd = [sys.executable, "Modello_supporto.py", "--target", TARGET_NAME]
p = subprocess.Popen(cmd, env=env)

try:
    p.wait()
except KeyboardInterrupt:
    p.terminate()