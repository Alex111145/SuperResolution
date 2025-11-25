import os
import sys
import shutil
import numpy as np
import matplotlib
# Use 'Agg' backend to be thread-safe and not require a display
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.visualization import PercentileInterval
from skimage.transform import resize
from tqdm import tqdm
import warnings
from concurrent.futures import ProcessPoolExecutor
from reproject import reproject_interp 
import threading 
import math

warnings.filterwarnings('ignore')

# ================= GLOBAL CONFIGURATION =================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# DATASET PARAMETERS
HR_SIZE = 512          # Hubble Patch Size
AI_LR_SIZE = 128       # Fixed output size for AI (Input)
STRIDE = 64           # High overlap (approx 90%) for Data Augmentation
MIN_COVERAGE = 0.95    # Relaxed threshold to accept clean edges
MIN_PIXEL_VALUE = 0.0001 

# DEBUG PARAMETERS
DEBUG_SAMPLES = 10     # Number of visual examples to generate

# Lock for thread-safe operations (logging, counters)
log_lock = threading.Lock()
# ========================================================

# Global variables for worker processes
shared_data = {}
patch_index_counter = 0

# --- UTILITY FUNCTIONS ---

def get_pixel_scale_deg(wcs):
    """Returns the mean pixel scale in degrees."""
    scales = proj_plane_pixel_scales(wcs)
    return np.mean(scales)

def normalize_local_stretch(data: np.ndarray) -> np.ndarray:
    """Robust normalization (0-1) for visualization."""
    try:
        d = np.nan_to_num(data)
        # Clip outliers (1st and 99.5th percentile) for better contrast
        vmin, vmax = np.percentile(d, [1, 99.5])
        if vmax <= vmin: return np.zeros_like(d)
        clipped = np.clip((d - vmin) / (vmax - vmin), 0, 1)
        return np.sqrt(clipped) # Square root stretch for astronomical data
    except:
        return np.zeros_like(data)

def get_corner_coords(wcs, shape):
    """Returns RA/DEC coordinates for the 4 corners and center."""
    ny, nx = shape
    # Pixel coordinates: TL(0,0), TR(w,0), BL(0,h), BR(w,h)
    corner_pixels = np.array([
        [0, 0],      # Top-Left
        [nx, 0],     # Top-Right
        [0, ny],     # Bottom-Left
        [nx, ny]     # Bottom-Right
    ])
    world = wcs.pixel_to_world(corner_pixels[:,0], corner_pixels[:,1])
    
    labels = ['TL', 'TR', 'BL', 'BR']
    coords = {}
    for i, lbl in enumerate(labels):
        coords[lbl] = f"RA:{world[i].ra.deg:.5f}, DEC:{world[i].dec.deg:.5f}"
    
    center = wcs.pixel_to_world(nx/2, ny/2)
    return coords, center

# --- VISUAL REPORT GENERATION (PNG) ---
def save_debug_png(patch_h, patch_o_lr, wcs_h, wcs_o, raw_crop_size, save_path, pair_id):
    """
    Generates a 3-panel PNG showing the HR patch, LR patch, and Overlay
    with WCS coordinate data printed on the image.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # Normalize data
    h_show = normalize_local_stretch(patch_h)
    o_show = normalize_local_stretch(patch_o_lr)
    o_resized = resize(o_show, h_show.shape, order=1) # Resized only for overlay visualization
    
    # Calculate Coordinates
    h_corners, h_center = get_corner_coords(wcs_h, patch_h.shape)
    # Note: patch_o_lr comes with the Target WCS, so its coords are correct (forced)
    o_corners, o_center = get_corner_coords(wcs_o, patch_o_lr.shape)
    
    # Calculate Mismatch (Offset)
    offset_arcsec = h_center.separation(o_center).arcsec

    # --- Panel 1: Hubble (Target) ---
    axes[0].imshow(h_show, origin='lower', cmap='inferno')
    axes[0].set_title("Hubble (HR Target) 512x512", color='red', fontweight='bold')
    axes[0].axis('off')
    # Add Text
    axes[0].text(0.02, 0.98, f"TL: {h_corners['TL']}", transform=axes[0].transAxes, color='cyan', fontsize=8, va='top', weight='bold')
    axes[0].text(0.02, 0.02, f"BR: {h_corners['BR']}", transform=axes[0].transAxes, color='cyan', fontsize=8, va='bottom', weight='bold')
    axes[0].text(0.5, 0.02, f"Center RA: {h_center.ra.deg:.5f}", transform=axes[0].transAxes, color='white', fontsize=8, ha='center', va='bottom')

    # --- Panel 2: Observatory (Input) ---
    axes[1].imshow(o_show, origin='lower', cmap='viridis')
    axes[1].set_title(f"Observatory (LR Input) {AI_LR_SIZE}x{AI_LR_SIZE}\n(Derived from ~{raw_crop_size}px raw crop)", color='green', fontweight='bold')
    axes[1].axis('off')
    axes[1].text(0.02, 0.98, f"TL: {o_corners['TL']}", transform=axes[1].transAxes, color='cyan', fontsize=8, va='top', weight='bold')
    axes[1].text(0.02, 0.02, f"BR: {o_corners['BR']}", transform=axes[1].transAxes, color='cyan', fontsize=8, va='bottom', weight='bold')
    axes[1].text(0.5, 0.02, f"Center RA: {o_center.ra.deg:.5f}", transform=axes[1].transAxes, color='white', fontsize=8, ha='center', va='bottom')

    # --- Panel 3: Overlay ---
    rgb = np.zeros((h_show.shape[0], h_show.shape[1], 3))
    rgb[..., 0] = h_show      # Red Channel
    rgb[..., 1] = o_resized   # Green Channel
    
    axes[2].imshow(rgb, origin='lower')
    axes[2].set_title(f"Overlay Check (Pair {pair_id})", color='white', fontweight='bold')
    axes[2].axis('off')
    
    # Match verdict
    color_verdict = 'lime' if offset_arcsec < 2.0 else 'red'
    # Use raw string for LaTeX Delta symbol if needed, or just text
    axes[2].text(0.5, 0.98, f"Mismatch: {offset_arcsec:.2f}\"", transform=axes[2].transAxes, color=color_verdict, fontsize=12, ha='center', va='top', weight='bold')
    axes[2].text(0.5, 0.02, "Yellow = Perfect Match", transform=axes[2].transAxes, color='yellow', fontsize=10, ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)

# --- WORKER LOGIC ---

def init_worker(d_h, hdr_h, w_h, out_fits, out_png, h_fov_deg, o_files):
    """Initializer for worker processes to setup shared data."""
    global patch_index_counter
    shared_data['h'] = d_h
    shared_data['header_h'] = hdr_h
    shared_data['wcs_h'] = w_h
    shared_data['out_fits'] = out_fits
    shared_data['out_png'] = out_png
    shared_data['h_fov_deg'] = h_fov_deg
    shared_data['o_files'] = o_files
    patch_index_counter = 0

def create_lr_wcs(hr_wcs, lr_size, fov_deg):
    """
    Crea un NUOVO WCS pulito per la patch LR, centrato esattamente dove punta HR.
    """
    # 1. Calcola il centro esatto della patch HR in coordinate cielo (RA, DEC)
    # Il centro della patch HR (512x512) è a pixel (256, 256)
    hr_center_world = hr_wcs.pixel_to_world(HR_SIZE / 2, HR_SIZE / 2)
    
    # 2. Crea un nuovo WCS da zero per la patch LR
    w = WCS(naxis=2)
    
    # Imposta il centro del nuovo WCS alle coordinate cielo di HR
    w.wcs.crval = [hr_center_world.ra.deg, hr_center_world.dec.deg]
    # w.wcs.ctype = hr_wcs.wcs.ctype # Opzionale: usa proiezione HR o standard
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    
    # Imposta il punto di riferimento pixel al centro della patch LR (64, 64)
    w.wcs.crpix = [lr_size / 2, lr_size / 2]
    
    # Calcola la scala pixel necessaria per coprire lo stesso FOV
    # Scala = FOV / Pixel
    scale = fov_deg / lr_size
    
    # Imposta la scala (con segno standard per l'astronomia: RA decresce)
    # Nota: Forziamo una rotazione zero (PC=Identity) per semplificare l'apprendimento
    w.wcs.cdelt = [-scale, scale]
    w.wcs.pc = np.eye(2)
    
    return w

def process_single_patch_multi(args):
    """Worker function: Extracts one HR patch and matches all valid LR files."""
    global patch_index_counter
    h_path, y, x = args
    
    # Load Hubble Data
    data_h = shared_data['h']
    wcs_h = shared_data['wcs_h']

    # 1. Extract HR Patch
    patch_h = data_h[y:y+HR_SIZE, x:x+HR_SIZE]
    
    # Quality Check: Skip empty or black edges
    if np.count_nonzero(patch_h > MIN_PIXEL_VALUE) / patch_h.size < MIN_COVERAGE:
        return 0

    # Create local WCS for this HR patch
    patch_h_wcs = wcs_h.deepcopy()
    patch_h_wcs.wcs.crpix -= np.array([x, y])

    # 2. Create Target LR WCS (Dynamic Size -> 128x128)
    # CORRETTO: Usa la nuova logica create_lr_wcs che forza il centro
    lr_target_wcs = create_lr_wcs(patch_h_wcs, AI_LR_SIZE, shared_data['h_fov_deg'])
    
    saved_count = 0
    
    # 3. Iterate over ALL aligned Observatory files
    for o_path in shared_data['o_files']:
        try:
            with fits.open(o_path) as o:
                data_o = np.nan_to_num(o[0].data)
                if data_o.ndim > 2: data_o = data_o[0]
                wcs_o = WCS(o[0].header)

            # 4. Reproject (The Geometric Matching)
            # Reproject using the FORCED WCS -> This guarantees alignment
            patch_o_lr, footprint = reproject_interp(
                (data_o, wcs_o),
                lr_target_wcs,
                shape_out=(AI_LR_SIZE, AI_LR_SIZE),
                order='bilinear'
            )
            patch_o_lr = np.nan_to_num(patch_o_lr)

            # Quality Check LR: Must have data (overlap)
            valid_mask = (patch_o_lr > MIN_PIXEL_VALUE)
            if np.sum(valid_mask) < (AI_LR_SIZE**2 * MIN_COVERAGE):
                continue

            # 5. Save Pair
            with log_lock:
                idx = patch_index_counter
                patch_index_counter += 1
            
            pair_dir = shared_data['out_fits'] / f"pair_{idx:06d}"
            pair_dir.mkdir(exist_ok=True)
            
            # Save FITS
            fits.PrimaryHDU(patch_h.astype(np.float32), header=patch_h_wcs.to_header()).writeto(pair_dir/"hubble.fits", overwrite=True)
            fits.PrimaryHDU(patch_o_lr.astype(np.float32), header=lr_target_wcs.to_header()).writeto(pair_dir/"observatory.fits", overwrite=True)
            
            saved_count += 1
            
            # 6. Create Debug PNG (Only for the first N samples)
            if idx < DEBUG_SAMPLES:
                # Calculate raw size just for info
                obs_scale = get_pixel_scale_deg(wcs_o)
                raw_size = int(shared_data['h_fov_deg'] / obs_scale)
                
                png_path = shared_data['out_png'] / f"check_pair_{idx:06d}.jpg"
                save_debug_png(patch_h, patch_o_lr, patch_h_wcs, lr_target_wcs, raw_size, png_path, idx)

        except Exception:
            continue
            
    return saved_count

# ================= MAIN =================

def select_target_directory():
    subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs']]
    if not subdirs: return None
    print("\nSELEZIONA TARGET:")
    for i, d in enumerate(subdirs): print(f" {i+1}: {d.name}")
    try:
        idx = int(input("Scelta: ")) - 1
        return subdirs[idx] if 0 <= idx < len(subdirs) else None
    except: return None

def main():
    print(f"🚀 ESTRAZIONE DINAMICA PATCH (HUBBLE vs OBS)")
    print(f"   Config: HR={HR_SIZE}px, LR={AI_LR_SIZE}px")
    print(f"   Overlap Stride: {STRIDE}px")
    
    target_dir = ROOT_DATA_DIR / "M1" # Default fallback
    if len(sys.argv) > 1: 
        target_dir = Path(sys.argv[1])
    else:
        sel = select_target_directory()
        if sel: target_dir = sel
    
    print(f"\n📂 Target selezionato: {target_dir.name}")
    
    input_h = target_dir / '3_registered_native' / 'hubble'
    input_o = target_dir / '3_registered_native' / 'observatory'
    
    out_fits = target_dir / '6_patches_final'
    out_png = target_dir / '6_debug_visuals' 
    
    if out_fits.exists(): shutil.rmtree(out_fits)
    out_fits.mkdir(parents=True)
    if out_png.exists(): shutil.rmtree(out_png)
    out_png.mkdir(parents=True)
    
    # 2. Caricamento File e Filtro Distanza
    h_files = sorted(list(input_h.glob("*.fits")))
    o_files_all = sorted(list(input_o.glob("*.fits")))
    
    if not h_files or not o_files_all:
        print("❌ File mancanti in 3_registered_native")
        return

    # Master Hubble (per griglia e WCS riferimento)
    h_master_path = h_files[0]
    try:
        with fits.open(h_master_path) as h:
            d_h = np.nan_to_num(h[0].data)
            if d_h.ndim > 2: d_h = d_h[0]
            w_h = WCS(h[0].header)
            h_head = h[0].header
            
        h_scale = get_pixel_scale_deg(w_h)
        h_fov_deg = h_scale * HR_SIZE
        h_center = w_h.wcs.crval
        print(f"   Hubble Master OK. Center: {h_center[0]:.4f}, {h_center[1]:.4f}")
        
    except Exception as e:
        print(f"❌ Errore lettura Hubble: {e}")
        return

    # Filtro file Osservatorio (Distanza < 0.1 deg)
    o_files_good = []
    print(f"   🔍 Filtraggio file non allineati...")
    for f in o_files_all:
        try:
            with fits.open(f) as o:
                w = WCS(o[0].header)
                # Distanza euclidea approssimata
                dist = np.sqrt((w.wcs.crval[0]-h_center[0])**2 + (w.wcs.crval[1]-h_center[1])**2)
                if dist < 0.1: # 0.1 gradi = 6 arcominuti
                    o_files_good.append(f)
        except: pass
        
    print(f"   ✅ File validi trovati: {len(o_files_good)} (scartati {len(o_files_all)-len(o_files_good)})")
    
    if not o_files_good:
        print("❌ Nessun file dell'osservatorio è centrato su Hubble. Controlla la registrazione.")
        return

    # 3. Generazione Task (Griglia su Hubble)
    h_h, h_w = d_h.shape
    tasks = []
    for y in range(0, h_h - HR_SIZE + 1, STRIDE):
        for x in range(0, h_w - HR_SIZE + 1, STRIDE):
            tasks.append((h_master_path, y, x))
            
    print(f"   📦 Patch Hubble da processare: {len(tasks)}")
    print(f"   ⏱️  Stima operazioni: {len(tasks) * len(o_files_good)} riproiezioni potenziali")
    
    # 4. Esecuzione Parallela
    print(f"\n🚀 Avvio estrazione...")
    total_saved = 0
    
    # Inizializza il pool con i dati condivisi
    with ProcessPoolExecutor(initializer=init_worker,
                             initargs=(d_h, h_head, w_h, out_fits, out_png, h_fov_deg, o_files_good)) as ex:
        
        results = list(tqdm(ex.map(process_single_patch_multi, tasks), total=len(tasks), ncols=100))
        total_saved = sum(results)
        
    print(f"\n✅ COMPLETATO.")
    print(f"   Coppie salvate: {total_saved}")
    print(f"   Dataset: {out_fits}")
    print(f"   Validation Images: {out_png}")

if __name__ == "__main__":
    main()