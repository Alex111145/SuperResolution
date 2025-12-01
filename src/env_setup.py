import sys
from pathlib import Path

def setup_paths():
    SRC_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SRC_DIR.parent
    MODELS_DIR = PROJECT_ROOT / "models"
    
    paths_to_add = [
        MODELS_DIR / "BasicSR",
        MODELS_DIR / "HAT"
    ]
    
    print(f"🔧 Setup Path (Root: {PROJECT_ROOT})...")
    
    for p in paths_to_add:
        if p.exists():
            str_p = str(p)
            if str_p not in sys.path:
                sys.path.insert(0, str_p)
                print(f"   ✅ {p.name}")
        else:
            print(f"   ⚠️ Non trovato: {p}")

setup_paths()

def import_external_archs():
    print("🔧 Import Moduli...")
    
    RRDBNet = None
    HAT = None
    
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        print("   ✅ BasicSR (RRDBNet)")
    except ImportError as e:
        print(f"   ❌ BasicSR: {e}")

    try:
        from hat.archs.hat_arch import HAT
        print("   ✅ HAT")
    except ImportError as e:
        try:
            from archs.hat_arch import HAT
            print("   ✅ HAT (alt)")
        except ImportError as e2:
            print(f"   ❌ HAT: {e}")

    return RRDBNet, HAT