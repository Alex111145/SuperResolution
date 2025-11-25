"""
STEP 3+4: RITAGLIO IMMAGINI REGISTRATE E CREAZIONE MOSAICO (SOLO HUBBLE)
"""

import os
import sys
import glob
import time
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
from astropy.io import fits
from tqdm import tqdm
import warnings
import subprocess

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAZIONE PATH DINAMICI
# ============================================================================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR_ROOT = ROOT_DATA_DIR / "logs"
SCRIPTS_DIR = CURRENT_SCRIPT_DIR

# ============================================================================
# SETUP LOGGING
# ============================================================================

def setup_logging():
    os.makedirs(LOG_DIR_ROOT, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = LOG_DIR_ROOT / f'crop_mosaic_hubble_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# ============================================================================
# FUNZIONI MENU E SELEZIONE
# ============================================================================

def select_target_directory(logger):
    """Mostra un menu per selezionare una o TUTTE le cartelle target."""
    logger.info("Avvio selezione cartella target.")
    print("\n" + "📂"*35)
    print("SELEZIONE CARTELLA TARGET".center(70))
    print("📂"*35)
    
    try:
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs', '__pycache__']]
    except Exception as e:
        logger.error(f"Impossibile leggere la cartella {ROOT_DATA_DIR}: {e}")
        return []

    if not subdirs:
        logger.warning(f"Nessuna sottocartella target trovata in {ROOT_DATA_DIR}")
        return []

    print("\nCartelle target disponibili:")
    print(f"   0: ✨ Processa TUTTI i {len(subdirs)} target")
    print("   " + "─"*30)
    for i, dir_path in enumerate(subdirs):
        print(f"   {i+1}: {dir_path.name}")

    while True:
        print("\n" + "─"*70)
        try:
            choice_str = input(f"👉 Seleziona un numero (0-{len(subdirs)}) o 'q' per uscire: ").strip()
            if choice_str.lower() == 'q': return [] 
            choice = int(choice_str)
            if choice == 0: return subdirs 
            choice_idx = choice - 1
            if 0 <= choice_idx < len(subdirs): return [subdirs[choice_idx]]
            else: print("❌ Scelta non valida.")
        except ValueError:
            print("❌ Input non valido.")

def find_smallest_dimensions(all_files, logger):
    logger.info("Ricerca dimensioni minime tra le immagini registrate.")
    print("\n🔍 Ricerca dimensioni minime...")
    min_height, min_width = float('inf'), float('inf')
    for filepath in tqdm(all_files, desc="Scansione", unit="file"):
        try:
            with fits.open(filepath) as hdul:
                data_shape = hdul[0].data.shape
                if len(data_shape) == 3: height, width = data_shape[1], data_shape[2]
                elif len(data_shape) == 2: height, width = data_shape
                else: continue
                if height < min_height: min_height = height
                if width < min_width: min_width = width
        except Exception as e: continue
    if min_height == float('inf'): return 0, 0
    return min_height, min_width

def crop_image(input_path, output_path, target_height, target_width, logger):
    try:
        with fits.open(input_path) as hdul:
            data = hdul[0].data
            header = hdul[0].header.copy()
            if len(data.shape) == 3: data = data[0]
            current_height, current_width = data.shape
            y_offset = (current_height - target_height) // 2
            x_offset = (current_width - target_width) // 2
            cropped_data = data[y_offset : y_offset + target_height, x_offset : x_offset + target_width]
            if 'CRPIX1' in header: header['CRPIX1'] -= x_offset
            if 'CRPIX2' in header: header['CRPIX2'] -= y_offset
            header['NAXIS1'] = target_width
            header['NAXIS2'] = target_height
            fits.PrimaryHDU(data=cropped_data, header=header).writeto(output_path, overwrite=True)
            return True
    except Exception as e:
        logger.error(f"ERRORE {input_path.name}: {e}")
        return False

def crop_all_images_for_target(base_dir, logger):
    logger.info(f"Avvio ritaglio per target: {base_dir.name} (SOLO HUBBLE)")
    input_dir_hubble = base_dir / '3_registered_native' / 'hubble'
    output_dir_hubble = base_dir / '4_cropped' / 'hubble'
    output_dir_hubble.mkdir(parents=True, exist_ok=True)
    all_files = list(input_dir_hubble.glob('*.fits')) + list(input_dir_hubble.glob('*.fit'))
    if not all_files: return False
    min_height, min_width = find_smallest_dimensions(all_files, logger)
    if min_height == 0: return False
    success_count = 0
    for filepath in tqdm(all_files, desc="Ritaglio Hubble", unit="file"):
        if crop_image(filepath, output_dir_hubble / filepath.name, min_height, min_width, logger):
            success_count += 1
    return success_count > 0

def create_mosaic_for_target(base_dir, logger):
    logger.info(f"Avvio creazione mosaico per target: {base_dir.name} (SOLO HUBBLE)")
    all_files = list((base_dir / '4_cropped' / 'hubble').glob('*.fits'))
    if not all_files: return False
    try:
        with fits.open(all_files[0]) as hdul:
            template_header = hdul[0].header.copy()
            shape = hdul[0].data.shape if len(hdul[0].data.shape) == 2 else hdul[0].data.shape[1:]
    except: return False
    total_flux = np.zeros(shape, dtype=np.float64)
    n_pixels = np.zeros(shape, dtype=np.int32)
    for filepath in tqdm(all_files, desc="Stacking Hubble", unit="file"):
        try:
            with fits.open(filepath) as hdul:
                d = hdul[0].data
                if len(d.shape) == 3: d = d[0]
                if d.shape != shape: continue
                valid = ~np.isnan(d) & (d != 0) 
                total_flux += np.nan_to_num(d, nan=0.0)
                n_pixels[valid] += 1
        except: continue
    mosaic_data = np.full(shape, np.nan, dtype=np.float32)
    valid_stack = n_pixels > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        mosaic_data[valid_stack] = (total_flux[valid_stack] / n_pixels[valid_stack]).astype(np.float32)
    mosaic_out = base_dir / '5_mosaics'
    mosaic_out.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=mosaic_data, header=template_header).writeto(mosaic_out / 'final_mosaic_hubble.fits', overwrite=True)
    return True

def main():
    logger = setup_logging()
    
    # Rilevamento Modalità Automatica
    auto_mode = len(sys.argv) > 1
    
    if auto_mode:
        input_path = Path(sys.argv[1]).resolve()
        print(f"\n🤖 Modalità Automatica: Target ricevuto {input_path.name}")
        target_dirs = [input_path]
    else:
        target_dirs = select_target_directory(logger)
        if not target_dirs: return

    for base_dir in target_dirs:
        logger.info(f"Processing target: {base_dir.name}")
        if not crop_all_images_for_target(base_dir, logger): continue
        create_mosaic_for_target(base_dir, logger)

    # SE AUTO MODE È ATTIVA, USCIAMO SENZA CHIEDERE INPUT
    if not auto_mode:
        input("Premi Enter per chiudere...")

if __name__ == "__main__":
    main()