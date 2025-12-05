import sys
import os
from pathlib import Path

def setup_paths():
    # Calcola la root: .../SuperResolution/
    SRC_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SRC_DIR.parent
    MODELS_DIR = PROJECT_ROOT / "models"
    
    # Percorsi critici da aggiungere a sys.path
    paths_to_add = [
        MODELS_DIR / "BasicSR",
        MODELS_DIR / "HAT"
    ]
    
    print(f"🔧 Configurazione percorsi Python (Root: {PROJECT_ROOT})...")
    
    for p in paths_to_add:
        if p.exists():
            str_p = str(p)
            if str_p not in sys.path:
                sys.path.insert(0, str_p)
                print(f"   ✅ Aggiunto al path: {p.name}")
        else:
            print(f"   ⚠️ ATTENZIONE: Percorso non trovato: {p}")

# Esegui subito il setup quando importato
setup_paths()

def import_external_archs():
    """Tenta di importare le architetture e stampa errori specifici se fallisce."""
    print("🔧 Importazione Moduli Esterni...")
    
    RRDBNet = None
    HAT = None
    SwinIR = None
    
    # 1. Import BasicSR (RRDBNet)
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        print("   ✅ RRDBNet importato.")
    except ImportError as e:
        print(f"   ❌ Errore RRDBNet: {e}")

    # 2. Import HAT
    try:
        from hat.archs.hat_arch import HAT
        print("   ✅ HAT importato.")
    except ImportError:
        try:
            from archs.hat_arch import HAT
            print("   ✅ HAT importato (path alt).")
        except ImportError:
            pass # Ignoriamo se manca HAT, stiamo usando SwinIR

    # 3. Import SwinIR (NUOVO)
    try:
        from basicsr.archs.swinir_arch import SwinIR
        print("   ✅ SwinIR importato.")
    except ImportError as e:
        print(f"   ❌ Errore SwinIR: {e}")
        print("      Assicurati di aver installato: pip install basicsr")

    return RRDBNet, HAT, SwinIR
