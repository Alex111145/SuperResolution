import os
import sys
import shutil
import torch
from pathlib import Path
import argparse

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent

def finalize_model(target_base_name):
    print(f"🔄 FINALIZZAZIONE: {target_base_name}")
    
    outputs_dir = PROJECT_ROOT / "outputs"
    
    # --- Cerca la cartella del training SwinIR o precedenti ---
    candidates = [
        outputs_dir / f"{target_base_name}_Worker_SwinIR_Light", # SwinIR (Nuovo)
        outputs_dir / f"{target_base_name}_Worker_RRDBNet_Pure", # RRDBNet
        outputs_dir / f"{target_base_name}_Worker_HAT_Nano",      
        outputs_dir / f"{target_base_name}_GPU_0"                  
    ]
    
    worker_dir = None
    for c in candidates:
        if c.exists():
            worker_dir = c
            break
            
    if worker_dir is None:
        print(f"❌ Nessuna cartella di training trovata in {outputs_dir}")
        print("   Esegui prima il training (Modello_3.py)")
        return

    print(f"   📂 Trovata cartella training: {worker_dir.name}")

    ckpt_best = worker_dir / "checkpoints" / "best_model.pth"
    ckpt_last = worker_dir / "checkpoints" / "last.pth"
    
    final_dir = outputs_dir / target_base_name / "final_weights"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = final_dir / "best.pth"

    source_path = ckpt_best if ckpt_best.exists() else ckpt_last
    if not source_path or not source_path.exists():
        print("❌ Nessun file .pth trovato.")
        return

    try:
        shutil.copy2(source_path, final_model_path)
        print(f"   💾 Modello salvato in: {final_model_path}")
        print("   ✅ Pronto per l'inferenza.")
            
    except Exception as e:
        print(f"❌ Errore durante la copia: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    args = parser.parse_args()
    finalize_model(args.target)
