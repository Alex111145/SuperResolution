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
    worker_dir = outputs_dir / f"{target_base_name}_GPU_0"
    
    if not worker_dir.exists():
        print(f"❌ Output non trovato: {worker_dir}")
        print("   Esegui prima Modello_3.py")
        return

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
        print("   ⚠️ Uso ultimo checkpoint")
        source_path = ckpt_last
    else:
        print("❌ Nessun checkpoint.")
        return

    try:
        shutil.copy2(source_path, final_model_path)
        print(f"   💾 Salvato: {final_model_path}")
        
        print("   🔍 Verifica...")
        state_dict = torch.load(final_model_path, map_location='cpu')
        if 'stage1.conv_first.weight' in state_dict:
            print("   ✅ Pesi validi (Hybrid Model)")
        else:
            print("   ⚠️ Struttura incerta")
            
    except Exception as e:
        print(f"❌ Errore: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    args = parser.parse_args()
    
    finalize_model(args.target)