import subprocess
import sys
import os
import torch
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

def select_single_gpu():
    """
    Rileva le GPU ma obbliga l'utente a selezionarne UNA sola.
    """
    print(f"\n{Colors.HEADER}🖥️  SELEZIONE GPU (Max 1 per Sanity Check){Colors.ENDC}")
    
    if not torch.cuda.is_available():
        print(f"{Colors.FAIL}❌ Nessuna GPU rilevata. Uscita.{Colors.ENDC}")
        sys.exit(1)
        
    count = torch.cuda.device_count()
    print(f"   Rilevate {count} GPU disponibili:")
    
    for i in range(count):
        name = torch.cuda.get_device_name(i)
        print(f"   {Colors.CYAN}[{i}]{Colors.ENDC} {name}")
        
    print(f"\n   ⚠️  Il Sanity Check richiede l'uso di una singola scheda.")

    while True:
        selection = input(f"\n{Colors.BOLD}Quale GPU usare? (Inserisci un solo numero, es. 0) > {Colors.ENDC}").strip()
        
        # Controllo che sia un numero singolo
        if selection.isdigit():
            idx = int(selection)
            if 0 <= idx < count:
                print(f"   ✅ Selezionata GPU {idx}: {torch.cuda.get_device_name(idx)}")
                return str(idx)
            else:
                print(f"{Colors.FAIL}⚠️ Indice {idx} non esistente.{Colors.ENDC}")
        else:
            print(f"{Colors.FAIL}⚠️ Input non valido. Inserisci un solo numero intero.{Colors.ENDC}")

def main():
    clear_screen()
    print(f"{Colors.HEADER}{Colors.BOLD}🚀 SANITY CHECK LAUNCHER (Single GPU){Colors.ENDC}")
    print(f"{Colors.HEADER}====================================={Colors.ENDC}")

    target_name = select_target()
    
    # Selezione singola GPU
    gpu_str = select_single_gpu()
    
    worker_script = HERE / "SanityCheck_Worker.py"
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = gpu_str
    # Ottimizzazione memoria anche per singola GPU
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cmd = [sys.executable, str(worker_script), "--target", target_name]

    print(f"\n{Colors.CYAN}⚡ Avvio Sanity Worker su GPU {gpu_str}...{Colors.ENDC}\n")
    try:
        subprocess.Popen(cmd, env=env).wait()
    except KeyboardInterrupt:
        print(f"\nSTOP.")

if __name__ == "__main__":
    main()