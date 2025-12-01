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
    print("🛠️  ASTRO SUPER-RES SETUP")
    print(f"📂  Root: {PROJECT_ROOT}")
    print("="*60)

    print("\n📦 [1/2] GPU Setup...")
    try:
        run_cmd(f"{sys.executable} -m pip install tensorboard")
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
    except:
        print("   ⚠️  Installazione PyTorch...")
        cuda_cmd = (f"{sys.executable} -m pip install torch torchvision torchaudio "
                   "--index-url https://download.pytorch.org/whl/cu118")
        run_cmd(cuda_cmd)

    print("\n📦 [2/2] Dipendenze...")
    libs = [
        "einops", "timm", "lmdb", "addict", "future", "yapf",
        "scipy", "\"numpy<2.0\"", "tqdm", "pyyaml",
        "matplotlib", "scikit-image", "opencv-python",
        "Pillow", "astropy", "astroalign", "reproject"
    ]
    
    try:
        libs_str = " ".join(libs)
        run_cmd(f"{sys.executable} -m pip install {libs_str}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore: {e}")

    print("\n✅ SETUP COMPLETE!")

if __name__ == "__main__":
    setup_project()