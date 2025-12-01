"""
MODELLO 4: FINALIZZAZIONE MODELLO (SINGLE GPU)
Prepara il modello finale per l'uso (copia e pulizia).
Non serve più il merge poiché usiamo una singola H200.
"""
import os
import sys
import shutil
import torch
from pathlib import Path
import argparse

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent

def finalize_model(target_base_name):
    print(f"🔄 FINALIZZAZIONE MODELLO PER: {target_base_name}")
    
    outputs_dir = PROJECT_ROOT / "outputs"
    
    # Cerchiamo l'output del Worker 0 (l'unico attivo)
    worker_dir = outputs_dir / f"{target_base_name}_GPU_0"
    
    if not worker_dir.exists():
        print(f"❌ Output non trovato: {worker_dir}")
        print("   Assicurati che il training (Modello_3) sia terminato.")
        return

    # Path sorgente e destinazione
    ckpt_best = worker_dir / "checkpoints" / "best_model.pth"
    ckpt_last = worker_dir / "checkpoints" / "last.pth" # Fallback
    
    final_dir = outputs_dir / target_base_name / "final_weights"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = final_dir / "best.pth"

    # Selezione file
    source_path = None
    if ckpt_best.exists():
        print("   ✅ Trovato checkpoint 'Best Model'")
        source_path = ckpt_best
    elif ckpt_last.exists():
        print("   ⚠️ 'Best Model' non trovato, uso l'ultimo checkpoint.")
        source_path = ckpt_last
    else:
        print("❌ Nessun checkpoint trovato.")
        return

    # Copia e verifica
    try:
        shutil.copy2(source_path, final_model_path)
        print(f"   💾 Modello salvato in: {final_model_path}")
        
        # Check integrità pesi
        print("   🔍 Verifica integrità pesi...")
        state_dict = torch.load(final_model_path, map_location='cpu')
        if 'stage1.conv_first.weight' in state_dict:
            print("   ✅ Struttura pesi valida (Hybrid Model rilevato).")
        else:
            print("   ⚠️  Struttura pesi incerta (chiavi non standard).")
            
    except Exception as e:
        print(f"❌ Errore durante la copia: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True, help="Nome del target (es. M42)")
    args = parser.parse_args()
    
    finalize_model(args.target)