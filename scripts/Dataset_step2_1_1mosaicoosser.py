"""
STEP 3+4: RITAGLIO IMMAGINI REGISTRATE E CREAZIONE MOSAICO (SOLO OSSERVATORIO)
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

def setup_logging():
    os.makedirs(LOG_DIR_ROOT, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = LOG_DIR_ROOT / f'crop_mosaic_observatory_{timestamp}.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_filename, encoding='utf-8'), logging.StreamHandler()])
    return logging.getLogger(__name__)

def select_target_directory():
    print("\n" + "📂"*35)
    print("SELEZIONE CARTELLA TARGET".center(70))
    print("📂"*35)
    try: subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs', '__pycache__']]
    except: return []
    if not subdirs: return []
    print("\nCartelle target disponibili:")
    print(f"   0: ✨ Processa TUTTI i {len(subdirs)} target")
    for i, dir_path in enumerate(subdirs): print(f"   {i+1}: {dir_path.name}")
    while True:
        try:
            choice_str = input(f"👉 Seleziona un numero (0-{len(subdirs)}) o 'q': ").strip()
            if choice_str == 'q': return []
            choice = int(choice_str)
            if choice == 0: return subdirs
            if 0 <= choice-1 < len(subdirs): return [subdirs[choice-1]]
        except: pass

def find_smallest_dimensions(all_files):
    print("\n🔍 Ricerca dimensioni minime...")
    min_height, min_width = float('inf'), float('inf')
    for filepath in tqdm(all_files, desc="Scansione", unit="file"):
        try:
            with fits.open(filepath) as hdul:
                data_shape = hdul[0].data.shape
                if len(data_shape) == 3: height, width = data_shape[1], data_shape[2]
                else: height, width = data_shape
                if height < min_height: min_height = height
                if width < min_width: min_width = width
        except: continue
    return min_height, min_width

def crop_image(input_path, output_path, target_height, target_width):
    try:
        with fits.open(input_path) as hdul:
            data = hdul[0].data
            header = hdul[0].header.copy()
            if len(data.shape) == 3: data = data[0]
            current_height, current_width = data.shape
            y_offset = (current_height - target_height) // 2
            x_offset = (current_width - target_width) // 2
            cropped_data = data[y_offset:y_offset + target_height, x_offset:x_offset + target_width]
            if 'CRPIX1' in header: header['CRPIX1'] -= x_offset
            if 'CRPIX2' in header: header['CRPIX2'] -= y_offset
            header['NAXIS1'] = target_width
            header['NAXIS2'] = target_height
            fits.PrimaryHDU(data=cropped_data, header=header).writeto(output_path, overwrite=True)
            return True
    except: return False

def crop_all_images_for_target(base_dir):
    print(f"RITAGLIO: {base_dir.name} (SOLO OSSERVATORIO)")
    input_dir = base_dir / '3_registered_native' / 'observatory'
    output_dir = base_dir / '4_cropped' / 'observatory'
    output_dir.mkdir(parents=True, exist_ok=True)
    all_files = list(input_dir.glob('*.fits')) + list(input_dir.glob('*.fit'))
    if not all_files: return False
    min_h, min_w = find_smallest_dimensions(all_files)
    success = 0
    for f in tqdm(all_files, desc="Ritaglio Osservatorio"):
        if crop_image(f, output_dir / f.name, min_h, min_w): success += 1
    return success > 0

def create_mosaic_for_target(base_dir):
    print(f"MOSAICO: {base_dir.name} (SOLO OSSERVATORIO)")
    all_files = list((base_dir / '4_cropped' / 'observatory').glob('*.fits'))
    if not all_files: return False
    try:
        with fits.open(all_files[0]) as hdul:
            hdr = hdul[0].header.copy()
            shape = hdul[0].data.shape if len(hdul[0].data.shape) == 2 else hdul[0].data.shape[1:]
    except: return False
    flux = np.zeros(shape); px = np.zeros(shape)
    for f in tqdm(all_files, desc="Stacking"):
        try:
            with fits.open(f) as h:
                d = h[0].data
                if len(d.shape) == 3: d = d[0]
                if d.shape != shape: continue
                valid = ~np.isnan(d) & (d!=0)
                flux += np.nan_to_num(d)
                px[valid] += 1
        except: continue
    mos = np.full(shape, np.nan, dtype=np.float32)
    valid = px > 0
    with np.errstate(divide='ignore'): mos[valid] = flux[valid]/px[valid]
    out = base_dir / '5_mosaics'; out.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=mos, header=hdr).writeto(out / 'final_mosaic_observatory.fits', overwrite=True)
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
        target_dirs = select_target_directory()
        if not target_dirs: return

    for base_dir in target_dirs:
        logger.info(f"Processing: {base_dir}")
        if crop_all_images_for_target(base_dir):
            create_mosaic_for_target(base_dir)

    # SE AUTO MODE È ATTIVA, USCIAMO SENZA CHIEDERE INPUT
    if not auto_mode:
        input("Premi Enter per chiudere...")

if __name__ == "__main__":
    main()