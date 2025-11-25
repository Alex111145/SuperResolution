"""
STEP 3 (FINAL): ESTRAZIONE PATCH TURBO + CONTEXT MAP + ZIP
------------------------------------------------------------------------
AGGIORNAMENTI:
- 4° Pannello nei Debug: Mappa completa di Hubble con rettangolo rosso tratteggiato
  che indica la posizione della patch.
- Rimosso i cerchi gialli (Clean Debug).
- Ottimizzazione File-Centric (Veloce).
- ZIP AUTOMATICO alla fine.

INPUT:  data/TARGET/3_registered_native/...
OUTPUT: data/TARGET/6_patches_final/...
        data/TARGET/visual_debug_TARGETNAME.zip
------------------------------------------------------------------------
"""

import os
import sys
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

# --- PARAMETRI ---
HR_SIZE = 512          # Hubble Patch
AI_LR_SIZE = 128       # Obs Patch (512 / 4)
STRIDE = 32            # Sovrapposizione
MIN_COVERAGE = 0.95    # Qualità alta (max 5% nero)
MIN_PIXEL_VALUE = 0.0001 
DEBUG_SAMPLES = 999999 # Salva TUTTI i debug

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

def save_debug_with_context(patch_h, patch_o_lr, idx, save_dir, hx, hy):
    """
    Salva PNG con 4 pannelli: HR, LR, Overlay e Mappa di Contesto.
    """
    try:
        # Layout: 4 colonne
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        
        h_v = normalize_local_stretch(patch_h)
        o_v = normalize_local_stretch(patch_o_lr)
        
        # 1. Hubble Patch
        ax[0].imshow(h_v, origin='lower', cmap='inferno')
        ax[0].set_title(f"Hubble HR ({HR_SIZE}px)")
        ax[0].axis('off')

        # 2. Osservatorio Patch
        ax[1].imshow(o_v, origin='lower', cmap='viridis')
        ax[1].set_title(f"Obs LR ({AI_LR_SIZE}px)")
        ax[1].axis('off')

        # 3. Overlay
        o_rez = resize(o_v, (HR_SIZE, HR_SIZE), order=0)
        rgb = np.zeros((HR_SIZE, HR_SIZE, 3))
        rgb[..., 0] = h_v
        rgb[..., 1] = o_rez
        ax[2].imshow(rgb, origin='lower')
        ax[2].set_title("Match (Rosso+Verde=Giallo)")
        ax[2].axis('off')

        # 4. MAPPA DI CONTESTO (Hubble intera con box)
        preview = shared_hubble['preview']
        ds = shared_hubble['preview_ds'] # Fattore di downsample
        
        ax[3].imshow(preview, origin='lower', cmap='gray')
        ax[3].set_title("Posizione Patch")
        ax[3].axis('off')
        
        # Disegna il rettangolo "rotto" (tratteggiato)
        # Coordinate scalate in base al downsample
        rect_x = hx / ds
        rect_y = hy / ds
        rect_w = HR_SIZE / ds
        rect_h = HR_SIZE / ds
        
        rect = patches.Rectangle((rect_x, rect_y), rect_w, rect_h, 
                                 linewidth=2, edgecolor='red', facecolor='none', 
                                 linestyle='--') # <-- Questo fa il "quadrato rotto"
        ax[3].add_patch(rect)

        plt.tight_layout()
        plt.savefig(save_dir / f"check_{idx:05d}.jpg", dpi=80)
        plt.close(fig)
    except Exception as e:
        pass

def create_lr_wcs_centered(hr_wcs, lr_size, fov_deg):
    center_world = hr_wcs.pixel_to_world(HR_SIZE/2, HR_SIZE/2)
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [center_world.ra.deg, center_world.dec.deg]
    w.wcs.crpix = [lr_size/2, lr_size/2]
    scale = fov_deg / lr_size
    w.wcs.cdelt = [-scale, scale]
    w.wcs.pc = np.eye(2)
    return w

# ================= WORKER LOGIC =================

def init_worker(d_h, w_h, h_fov, out_fits, out_png, grid_coords, grid_indices, counter, lock, preview, preview_ds):
    # Salva i dati condivisi nel processo
    shared_hubble['data'] = d_h
    shared_hubble['wcs'] = w_h
    shared_hubble['fov'] = h_fov
    shared_hubble['out_fits'] = out_fits
    shared_hubble['out_png'] = out_png
    shared_hubble['grid_coords'] = grid_coords
    shared_hubble['grid_indices'] = grid_indices
    
    # Dati per la mappa di contesto
    shared_hubble['preview'] = preview
    shared_hubble['preview_ds'] = preview_ds
    
    global global_counter, global_lock
    global_counter = counter
    global_lock = lock

def process_observatory_file(obs_path):
    saved_count = 0
    try:
        with fits.open(obs_path) as hdul:
            data_o = np.nan_to_num(hdul[0].data)
            if data_o.ndim > 2: data_o = data_o[0]
            wcs_o = WCS(hdul[0].header)
            
        oh, ow = data_o.shape[-2:]
        
        # Filtro Vettoriale
        world_coords = shared_hubble['grid_coords']
        pix_coords = wcs_o.wcs_world2pix(world_coords, 0)
        
        margin = 128 
        valid_mask = (pix_coords[:,0] > -margin) & (pix_coords[:,0] < ow + margin) & \
                     (pix_coords[:,1] > -margin) & (pix_coords[:,1] < oh + margin)
        valid_idxs = np.where(valid_mask)[0]
        
        if len(valid_idxs) == 0: return 0
            
        h_data = shared_hubble['data']
        h_wcs = shared_hubble['wcs']
        h_fov = shared_hubble['fov']
        
        for i in valid_idxs:
            hy, hx = shared_hubble['grid_indices'][i]
            
            # Estrai HR
            patch_h = h_data[hy:hy+HR_SIZE, hx:hx+HR_SIZE]
            if np.count_nonzero(patch_h > MIN_PIXEL_VALUE) / patch_h.size < MIN_COVERAGE:
                continue

            h_local_wcs = h_wcs.deepcopy()
            h_local_wcs.wcs.crpix -= np.array([hx, hy])
            
            lr_wcs = create_lr_wcs_centered(h_local_wcs, AI_LR_SIZE, h_fov)
            
            patch_o_lr, _ = reproject_interp(
                (data_o, wcs_o), lr_wcs, shape_out=(AI_LR_SIZE, AI_LR_SIZE), order='bilinear'
            )
            patch_o_lr = np.nan_to_num(patch_o_lr)
            
            if np.count_nonzero(patch_o_lr > MIN_PIXEL_VALUE) < (AI_LR_SIZE**2 * MIN_COVERAGE):
                continue
                
            with global_lock:
                pair_id = global_counter.value
                global_counter.value += 1
            
            pair_dir = shared_hubble['out_fits'] / f"pair_{pair_id:06d}"
            pair_dir.mkdir(exist_ok=True)
            
            fits.PrimaryHDU(patch_h.astype(np.float32), header=h_local_wcs.to_header()).writeto(pair_dir/"hubble.fits", overwrite=True)
            fits.PrimaryHDU(patch_o_lr.astype(np.float32), header=lr_wcs.to_header()).writeto(pair_dir/"observatory.fits", overwrite=True)
            
            saved_count += 1
            
            if pair_id < DEBUG_SAMPLES:
                # Passiamo hx, hy per disegnare il quadrato sulla mappa
                save_debug_with_context(patch_h, patch_o_lr, pair_id, shared_hubble['out_png'], hx, hy)
                
    except Exception: return 0
    return saved_count

# ================= MAIN =================

def main():
    print(f"\n🚀 GENERATORE DATASET 4x (Context Map + ZIP)")
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

    # Carica Hubble
    h_files = sorted(list(path_h.glob("*.fit*")))
    if not h_files: return print("❌ Nessun file Hubble.")
    
    print(f"\n📖 Caricamento Hubble Master ({h_files[0].name})...")
    with fits.open(h_files[0]) as h:
        d_h = np.nan_to_num(h[0].data)
        if d_h.ndim > 2: d_h = d_h[0]
        w_h = WCS(h[0].header)
    
    # --- CREAZIONE MAPPA CONTESTO (PREVIEW) ---
    print("🗺️  Creazione Mappa di Contesto per i Debug...")
    # Calcoliamo un fattore di downsample per avere una preview larga max 2000px
    h_h, h_w = d_h.shape
    preview_ds = max(1, int(max(h_h, h_w) / 2000))
    
    # Creiamo la preview (normalizzata per essere visibile)
    preview_data = normalize_local_stretch(d_h[::preview_ds, ::preview_ds])
    print(f"   Preview Factor: 1/{preview_ds}x")

    h_scale = get_pixel_scale_deg(w_h)
    h_fov = h_scale * HR_SIZE
    
    # Griglia
    print("📐 Generazione griglia...")
    y_list = range(0, h_h - HR_SIZE + 1, STRIDE)
    x_list = range(0, h_w - HR_SIZE + 1, STRIDE)
    
    grid_indices = []
    grid_centers_pix = []
    for y in y_list:
        for x in x_list:
            grid_indices.append([y, x])
            grid_centers_pix.append([x + HR_SIZE/2, y + HR_SIZE/2])
            
    grid_centers_pix = np.array(grid_centers_pix)
    grid_coords_world = w_h.wcs_pix2world(grid_centers_pix, 0)
    
    # File Obs
    o_files = sorted(list(path_o.glob("*.fit*")))
    print(f"   🔭 File Osservatorio: {len(o_files)}")
    
    # Processing
    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()
    
    print(f"\n🔥 Avvio {os.cpu_count()} worker...")
    with ProcessPoolExecutor(initializer=init_worker, 
                             initargs=(d_h, w_h, h_fov, out_fits, out_png, 
                                       grid_coords_world, np.array(grid_indices), 
                                       counter, lock, preview_data, preview_ds)) as ex:
        
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

        next_script = CURRENT_SCRIPT_DIR / 'Modello_2_pre_da_usopatch_dataset_step3.py'
        print(f"\nVuoi lanciare lo split dataset? (Invio=SI)")
        if input(">> ").strip() == "":
            subprocess.run([sys.executable, str(next_script), str(target_dir)])

if __name__ == "__main__":
    main()