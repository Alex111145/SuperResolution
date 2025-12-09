import subprocess
import sys
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

class Colors:
    HEADER = '\033[95m'; CYAN = '\033[96m'; GREEN = '\033[92m'; ENDC = '\033[0m'; BOLD = '\033[1m'

def select_target():
    targets = []
    if ROOT_DATA_DIR.exists():
        for d in sorted(ROOT_DATA_DIR.iterdir()):
            if d.is_dir() and ((d/"7_dataset_ready_LOG").exists() or (d/"7_dataset_ready").exists()):
                targets.append(d.name)
    
    if not targets: sys.exit("❌ Nessun dataset.")
    
    print(f"\n{Colors.HEADER}📂 SELEZIONE TARGET (Full Training):{Colors.ENDC}")
    for i, t in enumerate(targets): print(f"   [{i+1}] {t}")
    
    try: return targets[int(input(f"\nScegli numero: ")) - 1]
    except: sys.exit()

def main():
    os.system('clear')
    print(f"{Colors.HEADER}{Colors.BOLD}🚀 FULL TRAINING LAUNCHER{Colors.ENDC}")
    
    target_name = select_target()
    gpu_str = input(f"\n{Colors.BOLD}GPU IDs [es. 0,1]: {Colors.ENDC}").strip() or "0,1"
    
    # Punta al worker di training
    worker_script = HERE / "Train_Worker.py"
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = gpu_str
    
    cmd = [sys.executable, str(worker_script), "--target", target_name]
    
    print(f"\n{Colors.GREEN}⚡ Avvio Training su {target_name}...{Colors.ENDC}")
    try: subprocess.Popen(cmd, env=env).wait()
    except: pass

if __name__ == "__main__":
    main()