"""
STEP 3 (FINAL): ESTRAZIONE PATCH TURBO + CLEAN DEBUG + ZIP
------------------------------------------------------------------------
VERSIONE DEFINITIVA PER TRAINING 4x:
- Dimensioni: Input 128px -> Output 512px.
- Logica: File-Centric (Veloce) + Filtro Sovrapposizione (Preciso).
- Output: Dataset pronto e ZIP per debug visivo.
------------------------------------------------------------------------
"""

import os
import sys
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from skimage.transform import resize
from tqdm import tqdm
import warnings
from concurrent.futures import ProcessPoolExecutor
from reproject import reproject_interp 
import multiprocessing

warnings.filterwarnings('ignore')

# ================= CONFIGURAZIONE =================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# --- PARAMETRI (Rapporto 4x) ---
HR_SIZE = 512          # Hubble Patch (Target)
AI_LR_SIZE = 128       # Obs Patch (Input) -> 512 / 4 = 128
STRIDE = 64            # Sovrapposizione (64px è un buon compromesso velocità/quantità)
MIN_COVERAGE = 0.95    # Qualità alta (scarta se >5% è nero)
MIN_PIXEL_VALUE = 0.0001 
DEBUG_SAMPLES = 999999 # Salva tutti i debug per lo zip

# --- MEMORIA CONDIVISA WORKER ---
shared_hubble = {}
global_counter = None 
global_lock = None

# ================= UTILITY =================

def get_pixel_scale_deg(wcs):
    scales = proj_plane_pixel_scales(wcs)
    return np.mean(scales)

def normalize_local_stretch(data):
    try:
        d = np.nan_to_num(data)
        vmin, vmax = np.percentile(d, [1, 99.5])
        if vmax <= vmin: return np.zeros_like(d)
        return np.sqrt(np.clip((d - vmin) / (vmax - vmin), 0, 1))
    except: return np.zeros_like(data)

def save_clean_debug_png(patch_h, patch_o_lr, idx, save_dir):
    """Salva PNG pulito senza cerchi gialli"""
    try:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        h_v = normalize_local_stretch(patch_h)
        o_v = normalize_local_stretch(patch_o_lr)
        
        # 1. Hubble
        ax[0].imshow(h_v, origin='lower', cmap='inferno')
        ax[0].set_title(f"Hubble HR (512px)")
        ax[0].axis('off')

        # 2. Osservatorio
        ax[1].imshow(o_v, origin='lower', cmap='viridis')
        ax[1].set_title(f"Obs LR (128px)")
        ax[1].axis('off')

        # 3. Overlay
        o_rez = resize(o_v, (HR_SIZE, HR_SIZE), order=0)
        rgb = np.zeros((HR_SIZE, HR_SIZE, 3))
        rgb[..., 0] = h_v       # Rosso
        rgb[..., 1] = o_rez     # Verde
        
        ax[2].imshow(rgb, origin='lower')
        ax[2].set_title("Overlay (Giallo=Match)")
        ax[2].axis('off')

        plt.tight_layout()
        plt.savefig(save_dir / f"check_{idx:05d}.jpg", dpi=80)
        plt.close(fig)
    except: pass

def create_lr_wcs_centered(hr_wcs, lr_size, fov_deg):
    """Crea WCS LR centrato su Hubble per allineamento perfetto"""
    center_world = hr_wcs.pixel_to_world(HR_SIZE/2, HR_SIZE/2)
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [center_world.ra.deg, center_world.dec.deg]
    w.wcs.crpix = [lr_size/2, lr_size/2]
    scale = fov_deg / lr_size
    w.wcs.cdelt = [-scale, scale]
    w.wcs.pc = np.eye(2)
    return w

# ================= WORKER LOGIC (TURBO) =================

def init_worker(d_h, w_h, h_fov, out_fits, out_png, grid_coords, grid_indices, counter, lock):
    shared_hubble['data'] = d_h
    shared_hubble['wcs'] = w_h
    shared_hubble['fov'] = h_fov
    shared_hubble['out_fits'] = out_fits
    shared_hubble['out_png'] = out_png
    shared_hubble['grid_coords'] = grid_coords
    shared_hubble['grid_indices'] = grid_indices
    
    global global_counter, global_lock
    global_counter = counter
    global_lock = lock

def process_observatory_file(obs_path):
    saved_count = 0
    try:
        # 1. Carica File Osservatorio una volta sola
        with fits.open(obs_path) as hdul:
            data_o = np.nan_to_num(hdul[0].data)
            if data_o.ndim > 2: data_o = data_o[0]
            wcs_o = WCS(hdul[0].header)
            
        oh, ow = data_o.shape[-2:]
        
        # 2. Filtro Vettoriale (Sovrapposizione)
        world_coords = shared_hubble['grid_coords']
        pix_coords = wcs_o.wcs_world2pix(world_coords, 0)
        
        margin = 128 
        # Seleziona le patch Hubble che cadono dentro questa immagine Obs
        valid_mask = (pix_coords[:,0] > -margin) & (pix_coords[:,0] < ow + margin) & \
                     (pix_coords[:,1] > -margin) & (pix_coords[:,1] < oh + margin)
        valid_idxs = np.where(valid_mask)[0]
        
        if len(valid_idxs) == 0: return 0
            
        h_data = shared_hubble['data']
        h_wcs = shared_hubble['wcs']
        h_fov = shared_hubble['fov']
        
        # 3. Estrazione
        for i in valid_idxs:
            hy, hx = shared_hubble['grid_indices'][i]
            
            # Estrai HR
            patch_h = h_data[hy:hy+HR_SIZE, hx:hx+HR_SIZE]
            if np.count_nonzero(patch_h > MIN_PIXEL_VALUE) / patch_h.size < MIN_COVERAGE:
                continue

            # Crea LR corrispondente
            h_local_wcs = h_wcs.deepcopy()
            h_local_wcs.wcs.crpix -= np.array([hx, hy])
            
            lr_wcs = create_lr_wcs_centered(h_local_wcs, AI_LR_SIZE, h_fov)
            
            patch_o_lr, _ = reproject_interp(
                (data_o, wcs_o), lr_wcs, shape_out=(AI_LR_SIZE, AI_LR_SIZE), order='bilinear'
            )
            patch_o_lr = np.nan_to_num(patch_o_lr)
            
            if np.count_nonzero(patch_o_lr > MIN_PIXEL_VALUE) < (AI_LR_SIZE**2 * MIN_COVERAGE):
                continue
                
            # Salvataggio
            with global_lock:
                pair_id = global_counter.value
                global_counter.value += 1
            
            pair_dir = shared_hubble['out_fits'] / f"pair_{pair_id:06d}"
            pair_dir.mkdir(exist_ok=True)
            
            fits.PrimaryHDU(patch_h.astype(np.float32), header=h_local_wcs.to_header()).writeto(pair_dir/"hubble.fits", overwrite=True)
            fits.PrimaryHDU(patch_o_lr.astype(np.float32), header=lr_wcs.to_header()).writeto(pair_dir/"observatory.fits", overwrite=True)
            
            saved_count += 1
            
            if pair_id < DEBUG_SAMPLES:
                save_clean_debug_png(patch_h, patch_o_lr, pair_id, shared_hubble['out_png'])
                
    except Exception: return 0
    return saved_count

# ================= MAIN =================

def main():
    print(f"\n🚀 GENERATORE DATASET TURBO 4x (Clean + ZIP)")
    print(f"   HR: {HR_SIZE}px | LR: {AI_LR_SIZE}px | Stride: {STRIDE}")
    
    try:
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs', '__pycache__']]
    except: subdirs = []
    if not subdirs: return print("❌ Nessun target.")
    
    print("\nTarget disponibili:")
    for i, d in enumerate(subdirs): print(f" {i+1}: {d.name}")
    try:
        t_idx = int(input("Scelta: ")) - 1
        target_dir = subdirs[t_idx]
    except: return

    path_h = target_dir / '3_registered_native' / 'hubble'
    path_o = target_dir / '3_registered_native' / 'observatory'
    out_fits = target_dir / '6_patches_final'
    out_png = target_dir / '6_debug_visuals'
    
    if out_fits.exists(): shutil.rmtree(out_fits)
    out_fits.mkdir(parents=True)
    if out_png.exists(): shutil.rmtree(out_png)
    out_png.mkdir(parents=True)

    # Carica Hubble Master
    h_files = sorted(list(path_h.glob("*.fit*")))
    if not h_files: return print("❌ Nessun file Hubble.")
    
    print(f"\n📖 Caricamento Hubble Master ({h_files[0].name})...")
    with fits.open(h_files[0]) as h:
        d_h = np.nan_to_num(h[0].data)
        if d_h.ndim > 2: d_h = d_h[0]
        w_h = WCS(h[0].header)
    
    h_scale = get_pixel_scale_deg(w_h)
    h_fov = h_scale * HR_SIZE
    
    # Pre-calcolo Griglia Hubble
    print("📐 Generazione griglia coordinate...")
    hh, hw = d_h.shape
    y_list = range(0, hh - HR_SIZE + 1, STRIDE)
    x_list = range(0, hw - HR_SIZE + 1, STRIDE)
    
    grid_indices = []
    grid_centers_pix = []
    for y in y_list:
        for x in x_list:
            grid_indices.append([y, x])
            grid_centers_pix.append([x + HR_SIZE/2, y + HR_SIZE/2])
            
    grid_centers_pix = np.array(grid_centers_pix)
    grid_coords_world = w_h.wcs_pix2world(grid_centers_pix, 0)
    
    # File Osservatorio
    o_files = sorted(list(path_o.glob("*.fit*")))
    print(f"   🔭 File Osservatorio da analizzare: {len(o_files)}")
    
    # Multiprocessing
    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()
    
    print(f"\n🔥 Avvio {os.cpu_count()} worker...")
    with ProcessPoolExecutor(initializer=init_worker, 
                             initargs=(d_h, w_h, h_fov, out_fits, out_png, 
                                       grid_coords_world, np.array(grid_indices), 
                                       counter, lock)) as ex:
        
        results = list(tqdm(ex.map(process_observatory_file, o_files), 
                            total=len(o_files), unit="file", ncols=100))
        
    total_pairs = counter.value
    print(f"\n✅ GENERAZIONE COMPLETATA: {total_pairs} coppie.")

    # ZIP Automatica
    if total_pairs > 0:
        zip_name = target_dir / f"visual_debug_{target_dir.name}"
        print(f"\n📦 Creazione ZIP debug: {zip_name}.zip ...")
        shutil.make_archive(str(zip_name), 'zip', out_png)
        print("   ✅ ZIP creato.")

        # Lancio step successivo
        next_script = CURRENT_SCRIPT_DIR / 'Modello_2_pre_da_usopatch_dataset_step3.py'
        print(f"\nVuoi lanciare lo split dataset? (Invio=SI)")
        if input(">> ").strip() == "":
            subprocess.run([sys.executable, str(next_script), str(target_dir)])

if __name__ == "__main__":
    main()