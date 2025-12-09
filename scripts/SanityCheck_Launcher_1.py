import subprocess
import sys
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def clear_screen(): os.system('clear')

def select_target():
    targets = []
    if ROOT_DATA_DIR.exists():
        for d in sorted(ROOT_DATA_DIR.iterdir()):
            if d.is_dir() and (d / "7_dataset_ready_LOG").exists() or (d / "7_dataset_ready").exists():
                targets.append(d.name)
    
    if not targets:
        print(f"{Colors.FAIL}❌ Nessun dataset trovato.{Colors.ENDC}")
        sys.exit(1)
    
    print(f"\n{Colors.HEADER}📂 SELEZIONE TARGET (Sanity Check):{Colors.ENDC}")
    for i, t in enumerate(targets):
        print(f"   [{i+1}] {t}")
    
    while True:
        try:
            idx = int(input(f"\nScegli numero: ").strip()) - 1
            if 0 <= idx < len(targets): return targets[idx]
        except: pass

def main():
    clear_screen()
    print(f"{Colors.HEADER}{Colors.BOLD}🚀 SANITY CHECK LAUNCHER (Overfitting Mode){Colors.ENDC}")
    print(f"{Colors.HEADER}==========================================={Colors.ENDC}")

    target_name = select_target()
    gpu_str = input(f"\n{Colors.BOLD}GPU IDs [es. 0,1]: {Colors.ENDC}").strip() or "0,1"
    
    # --- PUNTA AL WORKER SPECIFICO PER SANITY ---
    worker_script = HERE / "SanityCheck_Worker.py"
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = gpu_str
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cmd = [sys.executable, str(worker_script), "--target", target_name]

    print(f"\n{Colors.CYAN}⚡ Avvio Sanity Worker...{Colors.ENDC}\n")
    try:
        subprocess.Popen(cmd, env=env).wait()
    except KeyboardInterrupt:
        print(f"\nSTOP.")

if __name__ == "__main__":
    main()