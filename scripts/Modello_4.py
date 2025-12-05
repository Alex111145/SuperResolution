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
    
    # --- MODIFICA: Cerca la cartella del nuovo training RRDBNet ---
    # Priorità alla nuova versione Pure, poi fallback sulle vecchie
    candidates = [
        outputs_dir / f"{target_base_name}_Worker_RRDBNet_Pure", # Nuova versione
        outputs_dir / f"{target_base_name}_Worker_HAT_Nano",      # Versione precedente
        outputs_dir / f"{target_base_name}_GPU_0"                  # Versione originale
    ]
    
    worker_dir = None
    for c in candidates:
        if c.exists():
            worker_dir = c
            break
            
    if worker_dir is None:
        print(f"❌ Nessuna cartella di training trovata in {outputs_dir}")
        print(f"   Cercavo: {[c.name for c in candidates]}")
        print("   Esegui prima il training (Modello_3.py)")
        return

    print(f"   📂 Trovata cartella training: {worker_dir.name}")

    ckpt_best = worker_dir / "checkpoints" / "best_model.pth"
    ckpt_last = worker_dir / "checkpoints" / "last.pth"
    
    final_dir = outputs_dir / target_base_name / "final_weights"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = final_dir / "best.pth"

    source_path = None
    if ckpt_best.exists():
        print("   ✅ Best Model trovato")
        source_path = ckpt_best
    elif ckpt_last.exists():
        print("   ⚠️ Uso ultimo checkpoint (Best non trovato)")
        source_path = ckpt_last
    else:
        print("❌ Nessun file .pth trovato nei checkpoints.")
        return

    try:
        shutil.copy2(source_path, final_model_path)
        print(f"   💾 Modello salvato in: {final_model_path}")
        
        print("   🔍 Verifica compatibilità pesi...")
        state_dict = torch.load(final_model_path, map_location='cpu')
        
        # Verifica generica per RRDBNet (con o senza wrapper DataParallel)
        keys = list(state_dict.keys())
        if any('conv_first' in k for k in keys):
            print("   ✅ Struttura Pesi OK (RRDBNet identificato)")
        else:
            print("   ⚠️ Attenzione: Chiavi del modello insolite, ma procedo.")
            
    except Exception as e:
        print(f"❌ Errore durante la copia: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    args = parser.parse_args()
    
    finalize_model(args.target)
