"""
MERGE WEIGHTS
"""
import os
import sys
import torch
import copy
from pathlib import Path
import argparse

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.architecture import HybridSuperResolutionModel 
except ImportError:
    sys.exit("❌ Errore import src.")

def merge_models(target_base_name, exclude_ranks=None):
    print(f"🔄 MERGING PESI PER: {target_base_name}")
    outputs_dir = PROJECT_ROOT / "outputs"
    all_worker_dirs = sorted(list(outputs_dir.glob(f"{target_base_name}_GPU_*")))
    
    exclude_ranks = set(map(str, exclude_ranks or []))
    worker_dirs = [w for w in all_worker_dirs if w.name.split('_')[-1] not in exclude_ranks]

    if not worker_dirs:
        print("❌ Nessun worker trovato.")
        return

    final_path = outputs_dir / target_base_name / "checkpoints" / "best.pth"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    
    models_state_dicts = []
    for w_dir in worker_dirs:
        ckpt_path = w_dir / "checkpoints" / "best_model.pth"
        if not ckpt_path.exists(): ckpt_path = w_dir / "checkpoints" / "last.pth"
        
        if ckpt_path.exists():
            print(f"   Load: {w_dir.name}")
            models_state_dicts.append(torch.load(ckpt_path, map_location='cpu'))

    if not models_state_dicts: return

    print("   ⚗️  Averaging...")
    avg_state = copy.deepcopy(models_state_dicts[0])
    for key in avg_state:
        if 'num_batches_tracked' in key: continue
        for i in range(1, len(models_state_dicts)):
            avg_state[key] += models_state_dicts[i][key]
        if torch.is_floating_point(avg_state[key]):
            avg_state[key] /= len(models_state_dicts)

    torch.save(avg_state, final_path)
    print(f"✅ Modello fuso salvato in: {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--exclude', nargs='*', type=int, default=[])
    args = parser.parse_args()
    merge_models(args.target, args.exclude)