import subprocess
import sys
import os
from pathlib import Path

# ================= CONFIGURAZIONE PATH =================
# Posizione corrente: /.../scripts/
HERE = Path(__file__).resolve().parent
# Root del progetto: /.../
PROJECT_ROOT = HERE.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# Codici Colore per estetica terminale
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_available_targets():
    """Cerca cartelle in data/ che abbiano completato lo step 7."""
    targets = []
    if ROOT_DATA_DIR.exists():
        for d in ROOT_DATA_DIR.iterdir():
            if d.is_dir() and d.name not in ['splits', 'logs', '.ipynb_checkpoints']:
                # Verifica esistenza marker di completamento step precedente
                if (d / "7_dataset_ready_LOG").exists():
                    targets.append(d.name)
    return sorted(targets)

def select_target():
    """Menu interattivo selezione dataset."""
    targets = get_available_targets()
    if not targets:
        print(f"{Colors.FAIL}❌ Nessun target pronto in {ROOT_DATA_DIR}")
        print(f"   (Assicurati di aver completato la preparazione del dataset){Colors.ENDC}")
        sys.exit(1)
    
    print(f"\n{Colors.HEADER}📂 SELEZIONE TARGET (Dataset Pronti):{Colors.ENDC}")
    for i, t in enumerate(targets):
        print(f"   {Colors.CYAN}[{i+1}]{Colors.ENDC} {t}")
    
    while True:
        try:
            choice = input(f"\n{Colors.BOLD}Scegli numero target: {Colors.ENDC}").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(targets):
                return targets[idx]
            print(f"{Colors.WARNING}Numero non valido.{Colors.ENDC}")
        except ValueError:
            print(f"{Colors.WARNING}Inserisci un numero valido.{Colors.ENDC}")

def select_gpus():
    """Menu selezione GPU con validazione."""
    # Nota: Modifica questa lista se hai più o meno GPU fisiche
    available_gpus = [0, 1, 2, 3]
    
    print(f"\n{Colors.HEADER}🖥️  CONFIGURAZIONE GPU{Colors.ENDC}")
    print(f"   ID Rilevati/Disponibili: {available_gpus}")
    print(f"{Colors.BLUE}   Esempio: '0,1' per usare le prime due. '0' per la prima.{Colors.ENDC}")
    print(f"{Colors.BLUE}   Lascia vuoto per default (GPU 0).{Colors.ENDC}")

    while True:
        selection = input(f"\n{Colors.BOLD}Quali GPU usare? > {Colors.ENDC}").strip()
        
        if selection == "":
            return "0", 1 
            
        try:
            # Pulisce l'input e crea lista di int
            selected_ids = [int(x.strip()) for x in selection.split(',') if x.strip().isdigit()]
            
            if not selected_ids:
                print(f"{Colors.WARNING}Nessun ID valido inserito.{Colors.ENDC}")
                continue
                
            # Verifica esistenza fisica (simulata dalla lista available_gpus)
            invalid = [x for x in selected_ids if x not in available_gpus]
            if invalid:
                print(f"{Colors.FAIL}❌ ID GPU inesistenti: {invalid}{Colors.ENDC}")
                continue
            
            # Rimuove duplicati e ordina
            selected_ids = sorted(list(set(selected_ids)))
            cuda_str = ",".join(map(str, selected_ids))
            return cuda_str, len(selected_ids)
            
        except Exception as e:
            print(f"{Colors.FAIL}Errore input: {e}{Colors.ENDC}")

def main():
    clear_screen()
    print(f"{Colors.HEADER}{Colors.BOLD}🚀 LANCIATORE TRAINING HYBRID SR{Colors.ENDC}")
    print(f"{Colors.HEADER}===================================={Colors.ENDC}")

    target_name = select_target()
    gpu_str, num_gpus = select_gpus()
    
    print(f"\n{Colors.GREEN}✅ RIEPILOGO AVVIO:{Colors.ENDC}")
    print(f"   🎯 Target:     {Colors.BOLD}{target_name}{Colors.ENDC}")
    print(f"   🖥️  GPU IDs:    {Colors.BOLD}{gpu_str}{Colors.ENDC} (Totale: {num_gpus})")
    print(f"   📂 Root Proj:  {PROJECT_ROOT}")
    
    input(f"\nPremi {Colors.BOLD}[INVIO]{Colors.ENDC} per avviare il Worker...")

    # --- CONFIGURAZIONE AMBIENTE ---
    env = os.environ.copy()
    
    # Aggiunge la root al PYTHONPATH per permettere 'import src...'
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    
    # Ottimizzazione allocazione memoria CUDNN/PyTorch
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Visibilità GPU
    env["CUDA_VISIBLE_DEVICES"] = gpu_str
    
    # Script da lanciare
    worker_script = HERE / "Modello_supporto.py"
    
    if not worker_script.exists():
        print(f"\n{Colors.FAIL}❌ ERRORE CRITICO: File non trovato: {worker_script}{Colors.ENDC}")
        sys.exit(1)

    # Usa l'eseguibile Python del venv attivo (se disponibile)
    python_exe = os.environ.get('VIRTUAL_ENV')
    if python_exe:
        python_exe = os.path.join(python_exe, 'bin', 'python')
    else:
        python_exe = sys.executable
    
    cmd = [python_exe, str(worker_script), "--target", target_name]

    print(f"\n{Colors.CYAN}⚡ Avvio sottoprocesso Python...{Colors.ENDC}\n")
    print("-" * 50)
    
    try:
        # Esegue il worker e attende la fine
        p = subprocess.Popen(cmd, env=env)
        p.wait()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}🛑 Interrotto dall'utente (Launcher).{Colors.ENDC}")
        p.terminate()
    except Exception as e:
        print(f"\n{Colors.FAIL}❌ Errore critico nel launcher: {e}{Colors.ENDC}")

if __name__ == "__main__":
    main()