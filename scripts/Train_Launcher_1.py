import subprocess
import sys
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

class Colors:
    HEADER = '\033[95m'; CYAN = '\033[96m'; GREEN = '\033[92m'; ENDC = '\033[0m'; BOLD = '\033[1m'; YELLOW = '\033[93m'

def get_connected_gpus():
    """Rileva le GPU NVIDIA connesse tramite nvidia-smi."""
    try:
        # Query per ottenere index, nome e memoria
        cmd = ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"]
        output = subprocess.check_output(cmd, encoding='utf-8')
        gpus = []
        for line in output.strip().split('\n'):
            idx, name, mem = line.split(',')
            gpus.append({
                'id': idx.strip(),
                'name': name.strip(),
                'mem': f"{mem.strip()} MB"
            })
        return gpus
    except FileNotFoundError:
        return [] # nvidia-smi non trovato
    except Exception as e:
        print(f"{Colors.YELLOW}⚠️ Errore rilevamento GPU: {e}{Colors.ENDC}")
        return []

def select_gpu_ids():
    """Menu per selezionare le GPU."""
    gpus = get_connected_gpus()
    
    # Se non trovo GPU (es. Mac o CPU only), chiedo manuale come fallback
    if not gpus:
        print(f"\n{Colors.YELLOW}⚠️ Nessuna GPU NVIDIA rilevata automaticamente.{Colors.ENDC}")
        return input(f"{Colors.BOLD}Inserisci GPU IDs manuali [es. 0,1]: {Colors.ENDC}").strip() or "0"

    print(f"\n{Colors.HEADER}🖥️  SELEZIONE GPU:{Colors.ENDC}")
    
    # Stampa lista GPU trovate
    for gpu in gpus:
        print(f"   [{gpu['id']}] {gpu['name']} ({gpu['mem']})")
    
    print(f"\nOpzioni rapide:")
    print(f"   [a] Seleziona TUTTE le {len(gpus)} GPU (ids: {','.join(g['id'] for g in gpus)})")

    selection = input(f"\n{Colors.BOLD}Scegli id (es. 0,1) o 'a' per tutte: {Colors.ENDC}").strip().lower()

    if selection == 'a':
        return ",".join([g['id'] for g in gpus])
    else:
        # Se l'utente scrive i numeri direttamente (es "0" o "0,1"), li ritorniamo
        return selection

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
    
    # 1. Selezione Dataset
    target_name = select_target()
    
    # 2. Selezione GPU
    gpu_str = select_gpu_ids()
    if not gpu_str: gpu_str = "0" # Fallback safe
    
    # Punta al worker di training
    worker_script = HERE / "Train_Worker.py"
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = gpu_str
    
    cmd = [sys.executable, str(worker_script), "--target", target_name]
    
    print(f"\n{Colors.GREEN}⚡ Avvio Training su {target_name}{Colors.ENDC}")
    print(f"{Colors.CYAN}   GPU IDs: {gpu_str}{Colors.ENDC}")
    
    try: subprocess.Popen(cmd, env=env).wait()
    except: pass

if __name__ == "__main__":
    main()