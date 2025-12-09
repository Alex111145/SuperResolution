import os
import sys
import time
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import warnings
from pathlib import Path
import subprocess
import shutil
import stat  # Import necessario per gestire i permessi

# Gestione importazioni opzionali
try:
    from reproject import reproject_interp
    REPROJECT_AVAILABLE = True
except ImportError:
    REPROJECT_AVAILABLE = False
    print("Libreria 'reproject' non trovata. La registrazione fallirà.")

warnings.filterwarnings('ignore')

# ================= CONFIGURAZIONE =================
# PROJECT_ROOT: La cartella del progetto (SuperResolution)
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR 
ROOT_DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR_ROOT = PROJECT_ROOT / "logs"

# === IMPOSTAZIONI TELESCOPIO ===
FORCE_FOV = 0.46  
USE_MANUAL_FOV = True 

NUM_THREADS = 2 
log_lock = threading.Lock()

# ================= UTILITY & SETUP =================

def setup_logging():
    os.makedirs(LOG_DIR_ROOT, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR_ROOT / f'pipeline_smart_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def make_executable(path):
    """Rende il file eseguibile (chmod +x) per evitare Permission Denied"""
    try:
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IEXEC)
        return True
    except Exception as e:
        print(f"Non riesco a rendere eseguibile {path}: {e}")
        return False

def find_astap_path():
    """
    Cerca ASTAP dando priorità alla cartella 'astap_local' che hai appena creato.
    """
    
    # Lista di possibili posizioni dove hai estratto astap_local
    # Cerca sia nella cartella corrente che in quella superiore
    search_dirs = [
        Path.cwd() / "astap_local",           # Cartella corrente
        PROJECT_ROOT / "astap_local",         # Dentro il progetto
        PROJECT_ROOT.parent / "astap_local"   # Accanto al progetto (dove c'era il .deb)
    ]

    # Percorsi tipici dentro il .deb estratto
    # Solitamente è usr/bin/astap oppure opt/astap/astap
    sub_paths = [
        Path("usr/bin/astap"),
        Path("opt/astap/astap"),
        Path("astap") # Se estratto flat
    ]

    print(f"Cerco ASTAP in 'astap_local'...")

    # 1. Cerca nella cartella estratta manualmente
    for base_dir in search_dirs:
        if base_dir.exists():
            for sub in sub_paths:
                candidate = base_dir / sub
                if candidate.exists():
                    make_executable(candidate) # FONDAMENTALE dopo l'estrazione manuale
                    return str(candidate)

    # 2. Se non lo trova lì, cerca nel sistema (installazione standard)
    sys_path = shutil.which("astap")
    if sys_path: return sys_path
    
    # 3. Percorsi Linux standard hardcoded
    linux_paths = ["/usr/bin/astap", "/opt/astap/astap"]
    for path in linux_paths:
        if os.path.isfile(path): return path
            
    return None

def select_target_directory():
    print("\n" + "="*35)
    print("PIPELINE REGISTRAZIONE (LINUX)".center(70))
    print("="*35)
    try:
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs']]
    except: return []
    if not subdirs: return []
    
    print("\nTarget disponibili:")
    for i, d in enumerate(subdirs): print(f" {i+1}: {d.name}")
    try:
        idx = int(input("Scelta: ")) - 1
        return [subdirs[idx]] if 0 <= idx < len(subdirs) else []
    except: return []

# ================= STEP 1: ASTAP SOLVING =================

def run_astap_cmd(cmd, logger):
    """Esegue il comando ASTAP su Linux."""
    cmd_str = [str(c) for c in cmd]
    # Check di sicurezza per vedere se il file esiste
    if not os.path.exists(cmd_str[0]):
        return None
    return subprocess.run(cmd_str, capture_output=True, text=True)

def solve_with_astap(inp_file, out_file, astap_exe, logger):
    try:
        shutil.copy2(inp_file, out_file)
        
        # Check preventivo
        try:
            with fits.open(out_file) as hdul:
                for hdu in hdul:
                    if WCS(hdu.header).has_celestial: return True
        except: pass 
        
        # --- TENTATIVO 1: Solving Veloce ---
        cmd_fast = [astap_exe, "-f", str(out_file), "-update", "-r", "30", "-z", "0"]
        res = run_astap_cmd(cmd_fast, logger)
        
        solved = False
        with fits.open(out_file) as hdul:
            for hdu in hdul:
                try:
                    if WCS(hdu.header).has_celestial: 
                        solved = True; break
                except: pass

        # --- TENTATIVO 2: Blind Solve ---
        if not solved:
            cmd_blind = [astap_exe, "-f", str(out_file), "-update", "-r", "180", "-z", "0"]
            if USE_MANUAL_FOV:
                cmd_blind.extend(["-fov", str(FORCE_FOV)])
                
            res = run_astap_cmd(cmd_blind, logger)
            
            with fits.open(out_file) as hdul:
                for hdu in hdul:
                    try:
                        if WCS(hdu.header).has_celestial: 
                            solved = True; break
                    except: pass

        if solved:
            for ext in ['.wcs', '.ini']:
                try: os.remove(out_file.with_suffix(ext))
                except: pass
            return True
        else:
            with log_lock: 
                logger.warning(f"FALLITO {inp_file.name}")
            return False

    except Exception as e:
        logger.error(f"Errore su {inp_file.name}: {e}")
        if out_file.exists(): os.remove(out_file)
        return False

def process_step1_folder(inp_dir, out_dir, astap_exe, logger):
    files = list(inp_dir.glob('*.fit')) + list(inp_dir.glob('*.fits')) + \
            list(inp_dir.glob('*.FIT')) + list(inp_dir.glob('*.FITS'))
    files = sorted(list(set(files)))
    if not files: return 0

    print(f"Solving {inp_dir.name} ({len(files)} img)...")
    success = 0
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = []
        for f in files:
            out_f = out_dir / f"{f.stem}_solved.fits"
            futures.append(executor.submit(solve_with_astap, f, out_f, astap_exe, logger))
        for f in tqdm(as_completed(futures), total=len(files), desc="   ASTAP"):
            if f.result(): success += 1
    return success

# ================= STEP 2: REGISTRAZIONE =================

def get_best_hdu(hdul):
    for hdu in hdul:
        if hdu.data is not None and hdu.data.ndim >= 2: return hdu
    return hdul[0]

def extract_wcs_info(f, logger=None):
    try:
        with fits.open(f) as hdul:
            hdu = get_best_hdu(hdul)
            w = WCS(hdu.header)
            if not w.has_celestial: return None
            
            scales = proj_plane_pixel_scales(w)
            if scales is not None and len(scales) >= 1:
                scale_arcsec = scales[0] * 3600
            else:
                scale_arcsec = abs(w.wcs.cdelt[0]) * 3600 

            return {
                'file': f, 'wcs': w, 'shape': hdu.data.shape, 
                'ra': w.wcs.crval[0], 'dec': w.wcs.crval[1], 'scale': scale_arcsec
            }
    except Exception: return None

def register_single_image_smart(info, ref_wcs, out_dir, logger):
    try:
        fname = info['file'].name
        with fits.open(info['file']) as hdul:
            hdu = get_best_hdu(hdul)
            data = np.nan_to_num(hdu.data)
            header = hdu.header
            wcs_orig = WCS(header)
            
            if data.ndim == 3: data = data[0]
            data = np.where(data < -10000, np.nan, data)

        native_scale_deg = info['scale'] / 3600.0
        
        target_wcs = WCS(naxis=2)
        target_wcs.wcs.crval = ref_wcs.wcs.crval 
        target_wcs.wcs.crpix = [data.shape[1]/2, data.shape[0]/2]
        target_wcs.wcs.cdelt = [-native_scale_deg, native_scale_deg]
        target_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"] 
        
        output_data, _ = reproject_interp((data, wcs_orig), target_wcs, shape_out=data.shape)

        out_name = f"reg_{fname}"
        hdr_new = target_wcs.to_header()
        hdr_new['REG_METH'] = "ASTAP_SOLVE+REPROJECT"
        
        fits.PrimaryHDU(data=output_data.astype(np.float32), header=hdr_new).writeto(out_dir/out_name, overwrite=True)
        return {'status': 'ok', 'file': out_name}

    except Exception as e:
        return {'status': 'err', 'file': info['file'].name, 'err': str(e)}

def main_registration(h_in, o_in, h_out, o_out, logger):
    h_files = list(h_in.glob('*_solved.fits'))
    o_files = list(o_in.glob('*_solved.fits'))
    
    print("   Lettura WCS headers...")
    h_infos = [x for x in [extract_wcs_info(f, logger) for f in h_files] if x]
    o_infos = [x for x in [extract_wcs_info(f, logger) for f in o_files] if x]
    
    if not h_infos: 
        logger.error("Nessun file Hubble risolto.")
        return False
    
    common_wcs = h_infos[0]['wcs']
    print(f"   Master Ref (Hubble): RA {common_wcs.wcs.crval[0]:.4f}, DEC {common_wcs.wcs.crval[1]:.4f}")
    
    success = 0
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as ex:
        futures = []
        for info in h_infos: futures.append(ex.submit(register_single_image_smart, info, common_wcs, h_out, logger))
        for info in o_infos: futures.append(ex.submit(register_single_image_smart, info, common_wcs, o_out, logger))
            
        for f in tqdm(as_completed(futures), total=len(futures), desc="   Registrazione"):
            if f.result()['status'] == 'ok': success += 1
            
    return success > 0

# ================= MAIN =================

def main():
    logger = setup_logging()
    
    # 1. Trova ASTAP nella cartella locale
    ASTAP_PATH = find_astap_path()
    
    if not ASTAP_PATH:
        print("\n" + "!"*50)
        print("ERRORE: ASTAP non trovato neanche in 'astap_local'!")
        print("Verifica di essere nella cartella dove hai lanciato 'dpkg -x'")
        print("!"*50 + "\n")
        return
    else:
        print(f"ASTAP trovato in: {ASTAP_PATH}")
    
    targets = select_target_directory()
    if not targets: return

    for BASE_DIR in targets:
        print(f"\n ELABORAZIONE: {BASE_DIR.name}")
        
        in_o = BASE_DIR / '1_originarie/local_raw'
        in_h = BASE_DIR / '1_originarie/img_lights'
        out_solved_o = BASE_DIR / '2_solved_astap/observatory'
        out_solved_h = BASE_DIR / '2_solved_astap/hubble'
        out_reg_o = BASE_DIR / '3_registered_native/observatory'
        out_reg_h = BASE_DIR / '3_registered_native/hubble'
        
        for p in [out_solved_o, out_solved_h, out_reg_o, out_reg_h]: p.mkdir(parents=True, exist_ok=True)

        print("   [1/2] Astrometric Solving...")
        s1 = process_step1_folder(in_o, out_solved_o, ASTAP_PATH, logger)
        s2 = process_step1_folder(in_h, out_solved_h, ASTAP_PATH, logger)
        
        if s1+s2 == 0:
            print("Nessun file risolto. Controlla il FOV in ASTAP o i cataloghi stellari.")
            continue

        print("   [2/2] Registrazione e Riproiezione...")
        if main_registration(out_solved_h, out_solved_o, out_reg_h, out_reg_o, logger):
            print("Pipeline Completata.")
        else:
            print("Errore Registrazione.")

if __name__ == "__main__":
    main()