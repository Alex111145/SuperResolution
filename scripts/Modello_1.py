import os
import sys
import subprocess
from pathlib import Path

def run_cmd(cmd):
    print(f"🚀 Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def setup_project():
    HERE = Path(__file__).resolve().parent
    PROJECT_ROOT = HERE.parent 
    MODELS_DIR = PROJECT_ROOT / "models"
    MODELS_DIR.mkdir(exist_ok=True)
    
    print("="*60)
    print("🛠️  ASTRO SUPER-RES SETUP (TIFF EDITION)")
    print(f"📂  Project Root: {PROJECT_ROOT}")
    print("="*60)

    # --- FASE 1: Configurazione GPU ---
    print("\n📦 [1/2] Configurazione ambiente GPU...")
    try:
        run_cmd(f"{sys.executable} -m pip install tensorboard")
        # Check veloce torch
        import torch
        print(f"   ✅ PyTorch rilevato: {torch.__version__}")
    except:
        print("   ⚠️  PyTorch non trovato o errore. Installazione in corso...")
        cuda_cmd = (f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        run_cmd(cuda_cmd)

    # --- FASE 2: Librerie Progetto ---
    print("\n📦 [2/2] Installazione dipendenze Progetto...")
    libs = [
        "einops", "timm", "lmdb", "addict", "future", "yapf",
        "scipy", "\"numpy<2.0\"", "tqdm", "pyyaml",
        "matplotlib", "scikit-image", "opencv-python",
        "Pillow",  # CRUCIALE PER TIFF
        "astropy", "astroalign", "reproject"
    ]
    
    try:
        libs_str = " ".join(libs)
        run_cmd(f"{sys.executable} -m pip install {libs_str}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore dipendenze: {e}")

    print("\n✅ SETUP COMPLETE!")

if __name__ == "__main__":
    setup_project()