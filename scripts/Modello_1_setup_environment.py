import os
import sys
import subprocess
from pathlib import Path

def run_cmd(cmd):
    print(f"🚀 Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def setup_project():
    # Ottiene il percorso di QUESTO script (dentro 'scripts')
    HERE = Path(__file__).resolve().parent
    # La root del progetto è il genitore di 'scripts'
    PROJECT_ROOT = HERE.parent 
    
    # Modelli e Requirements sono nella ROOT
    MODELS_DIR = PROJECT_ROOT / "models"
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Il requirements.txt potrebbe essere in scripts o in root.
    REQUIREMENTS_FILE = HERE / "requirements.txt" 

    print("="*60)
    print("🛠️  ASTRO SUPER-RES SETUP COMPLETO")
    print(f"📂  Project Root: {PROJECT_ROOT}")
    print("="*60)

    # --- AVVISO PER DIPENDENZE DI SISTEMA ---
    print("\n🚨 ATTENZIONE: Se sei su Linux, assicurati di aver installato le librerie di sistema:")
    print("   apt-get update && apt-get install -y libgl1 libglib2.0-0")
    print("   Queste non sono installabili tramite pip.")
    print("─"*60)

    # --- FASE 1: Configurazione GPU (CUDA) ---
    print("\n📦 [1/2] Configurazione ambiente GPU (CUDA)...")
    try:
        # 1. Installiamo TensorBoard
        run_cmd(f"{sys.executable} -m pip install tensorboard")

        # 2. Disinstalliamo versioni vecchie per evitare conflitti
        print("   🧹 Pulizia vecchie versioni torch...")
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"],
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )

        # 3. Installiamo PyTorch + Torchvision con supporto CUDA 11.8
        print("   📥 Scaricamento e installazione PyTorch CUDA (attendere)...")
        cuda_install_cmd = (
            f"{sys.executable} -m pip install "
            "torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cu118"
        )
        run_cmd(cuda_install_cmd)

    except subprocess.CalledProcessError as e:
        print(f"⚠️  Errore installazione GPU: {e}")
        print("   Continuerò con il resto delle librerie...")

    # --- FASE 2: Installazione Librerie Progetto (Tutte incluse) ---
    print("\n📦 [2/2] Installazione dipendenze Progetto (Astro, CV, AI Utils)...")
    
    # Lista consolidata di TUTTE le librerie necessarie per la pipeline
    libs = [
        # Deep Learning Utils / Architecture helper
        "einops",          # INCLUSO SU TUA RICHIESTA
        "timm",            # INCLUSO SU TUA RICHIESTA
        "lmdb", 
        "addict", 
        "future", 
        "yapf",
        
        # Scientific & Math
        "scipy", 
        "\"numpy<2.0\"",  # Quotato per evitare conflitti con versioni 2.x
        "tqdm",
        "pyyaml",
        
        # Image Processing & Visualization
        "matplotlib",       
        "scikit-image", 
        "opencv-python",           
        "opencv-contrib-python",   
        
        # Astronomy
        "astropy", 
        "astroalign", 
        "reproject",        # NECESSARIO per Dataset_step1 (registrazione)
    ]
    
    try:
        # Unisce la lista in un'unica stringa
        libs_str = " ".join(libs)
        run_cmd(f"{sys.executable} -m pip install {libs_str}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore durante l'installazione delle dipendenze extra: {e}")

    # --- FASE 3: Requirements.txt (Opzionale/Fallback) ---
    if REQUIREMENTS_FILE.exists():
        print(f"\n📦 Controllo dipendenze aggiuntive da {REQUIREMENTS_FILE.name}...")
        run_cmd(f"{sys.executable} -m pip install -r {str(REQUIREMENTS_FILE)}")
    
    print("\n✅ SETUP COMPLETE!")

def ask_continue_to_next_step():
    """Chiede se proseguire con il prossimo script."""
    next_script_name = 'Modello_2_prepare_data.py' 
    HERE = Path(__file__).resolve().parent
    next_script_path = HERE / next_script_name

    print("\n" + "="*70)
    print("🎯 STEP 1 COMPLETATO!")
    print("="*70)
    print(f"\n📋 PROSSIMO STEP SUGGERITO: {next_script_name}")
    
    while True:
        print("\n" + "─"*70)
        choice = input(f"👉 Vuoi avviare '{next_script_name}' ora? [S/n]: ").strip().lower()
        if choice in ('', 's', 'si', 'y', 'yes'):
            if next_script_path.exists():
                print(f"\n🚀 Avvio {next_script_name}...")
                try:
                    subprocess.run([sys.executable, str(next_script_path)])
                except Exception as e:
                    print(f"❌ Errore durante l'avvio dello script: {e}")
            else:
                print(f"❌ Errore: File non trovato: {next_script_path}")
            return
        elif choice in ('n', 'no'):
            print(f"\n👋 Ok. Ricorda i comandi dal Readme:")
            print("   HEAVY: python scripts/Modello_2_prepare_data.py")
            print("   LIGHT: python scripts/Modello_2_prepare_data_copy.py")
            return
        else:
            print("❌ Scelta non valida.")

if __name__ == "__main__":
    setup_project()
    ask_continue_to_next_step()