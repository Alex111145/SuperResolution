import os
import sys
import shutil
import numpy as np
import matplotlib
# Imposta il backend non interattivo per il multiprocessing
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.visualization import PercentileInterval, ZScaleInterval
from skimage.transform import resize
from skimage.registration import phase_cross_correlation
from tqdm import tqdm
import warnings
from concurrent.futures import ProcessPoolExecutor
from reproject import reproject_interp 
import threading 
import math
import traceback

warnings.filterwarnings('ignore')

# ================= CONFIGURAZIONE GLOBALE =================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# PARAMETRI DATASET
HR_SIZE = 512 
AI_LR_SIZE = 128
STRIDE = 51 
MIN_COVERAGE = 0.50 
MIN_PIXEL_VALUE = 0.0001 
DEBUG_SAMPLES = 10

# LIMITI PER L'AUTO-ALLINEAMENTO
MAX_SHIFT_PIXELS = 30  # Se lo shift calcolato è maggiore di 30px LR, lo ignoriamo (probabile errore)

log_lock = threading.Lock()
# ==========================================================

# Variabile globale per i processi worker
shared_data = {}
patch_index_counter = 0

# --- FUNZIONI DI UTILITY ---

def get_pixel_scale_deg(wcs):
    scales = proj_plane_pixel_scales(wcs)
    return np.mean(scales)

def get_robust_preview(data, size=None):
    try:
        data = np.nan_to_num(data)
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(data)
        clipped = np.clip((data - vmin) / (vmax - vmin), 0, 1)
        if size:
            return resize(clipped, (size, size), anti_aliasing=True)
        return clipped
    except:
        return np.zeros_like(data)

def calculate_wcs_corners(wcs, x_min, y_min, size):
    corner_pixels = np.array([ [0, 0], [size, 0], [0, size], [size, size] ])
    world_coords = wcs.pixel_to_world(corner_pixels[:, 0], corner_pixels[:, 1])
    formatted_coords = {}
    labels = ['TL', 'TR', 'BL', 'BR']
    for i, label in enumerate(labels):
        formatted_coords[label] = f"RA:{world_coords[i].ra.deg:.5f} DEC:{world_coords[i].dec.deg:.5f}"
    center_world = wcs.pixel_to_world(size/2, size/2)
    return formatted_coords, center_world.ra.deg, center_world.dec.deg

# --- DEBUG PNG ---
def save_diagnostic_card(data_h_orig, data_o_raw_orig, 
                         patch_h, patch_o_lr, 
                         x, y, wcs_h, wcs_o_raw,
                         lr_wcs_target,
                         raw_crop_size, h_fov_deg, save_path, shift_applied):
    try:
        fig = plt.figure(figsize=(20, 12)) 
        gs = fig.add_gridspec(2, 3)

        # WCS Info
        h_patch_wcs = wcs_h.deepcopy()
        h_patch_wcs.wcs.crpix = h_patch_wcs.wcs.crpix - np.array([x, y])
        
        h_coords_str, h_ra, h_dec = calculate_wcs_corners(h_patch_wcs, 0, 0, HR_SIZE)
        lr_coords_str, lr_ra, lr_dec = calculate_wcs_corners(lr_wcs_target, 0, 0, AI_LR_SIZE)
        
        mismatch_ra = abs(h_ra - lr_ra) * 3600
        mismatch_dec = abs(h_dec - lr_dec) * 3600

        # RIGA 1: Contesto
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        h_small = get_robust_preview(data_h_orig, 512)
        ax1.imshow(h_small, origin='lower', cmap='inferno')
        ax1.set_title("GLOBAL HUBBLE MAP", color='red', fontweight='bold')
        
        # Box Patch
        scale_y = 512 / data_h_orig.shape[0]
        scale_x = 512 / data_h_orig.shape[1]
        rect_h = patches.Rectangle((x*scale_x, y*scale_y), HR_SIZE*scale_x, HR_SIZE*scale_y, 
                                   linewidth=2, edgecolor='cyan', facecolor='none')
        ax1.add_patch(rect_h)

        ax2 = fig.add_subplot(gs[0, 1])
        o_small = get_robust_preview(data_o_raw_orig, 512)
        ax2.imshow(o_small, origin='lower', cmap='viridis')
        ax2.set_title("GLOBAL OBS MAP", color='green', fontweight='bold')
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        txt_coords = (f"📍 ALIGNMENT CHECK (Pair {save_path.stem})\n"
                      f"-----------------------------------\n"
                      f"SHIFT APPLIED (Y, X): {shift_applied}\n"
                      f"RAW CROP SIZE: ~{raw_crop_size} px\n"
                      f"MISMATCH (Post-Fix): {mismatch_ra:.2f}\" / {mismatch_dec:.2f}\"\n")
        ax3.text(0.01, 0.5, txt_coords, fontsize=12, verticalalignment='center', family='monospace')

        # RIGA 2: Dettaglio
        inp_s = resize(get_robust_preview(patch_o_lr), (HR_SIZE, HR_SIZE), order=1)
        tar_n = get_robust_preview(patch_h)
        rgb = np.zeros((HR_SIZE, HR_SIZE, 3))
        rgb[..., 0] = tar_n * 0.9  
        rgb[..., 1] = inp_s * 0.9  

        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow(tar_n, origin='lower', cmap='inferno')
        ax4.set_title("1. Hubble Patch (Target)", color='red')
        ax4.axis('off')

        ax5 = fig.add_subplot(gs[1, 1])
        ax5.imshow(get_robust_preview(patch_o_lr), origin='lower', cmap='viridis')
        ax5.set_title(f"2. Obs Patch (Aligned) | {AI_LR_SIZE}px", color='green')
        ax5.axis('off')

        ax6 = fig.add_subplot(gs[1, 2])
        ax6.imshow(rgb, origin='lower')
        ax6.set_title(f"3. Overlay (Detail)", color='white')
        ax6.text(0.5, 0.05, "Yellow = Match", color='yellow', ha='center', transform=ax6.transAxes)
        ax6.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=90)
        plt.close(fig)
    except Exception as e:
        print(f"Error saving PNG: {e}")

# --- WORKER LOGIC ---

def init_worker(d_h, hdr_h, w_h, out_fits, out_png, h_fov_deg, o_files):
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
    scale = fov_deg / lr_size
    w = WCS(naxis=2)
    w.wcs.crval = hr_wcs.wcs.crval
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [lr_size / 2.0, lr_size / 2.0]
    w.wcs.cdelt = [-scale, scale]
    w.wcs.pc = np.eye(2)
    return w

def process_single_patch_multi(args):
    global patch_index_counter
    h_path, y, x = args
    
    data_h = shared_data['h']
    wcs_h = shared_data['wcs_h']

    # 1. Estrazione HR
    patch_h = data_h[y:y+HR_SIZE, x:x+HR_SIZE]
    if np.count_nonzero(patch_h > MIN_PIXEL_VALUE) / patch_h.size < MIN_COVERAGE:
        return 0

    patch_h_wcs = wcs_h.deepcopy()
    patch_h_wcs.wcs.crpix -= np.array([x, y])

    # 2. Creazione WCS Target Iniziale
    lr_target_wcs = create_lr_wcs(patch_h_wcs, AI_LR_SIZE, shared_data['h_fov_deg'])
    
    saved_count = 0
    
    for o_path in shared_data['o_files']:
        try:
            with fits.open(o_path) as o:
                data_o = np.nan_to_num(o[0].data)
                if data_o.ndim > 2: data_o = data_o[0]
                wcs_o = WCS(o[0].header)

            # 3. Riproiezione INIZIALE (Grezza)
            patch_o_lr_raw, _ = reproject_interp(
                (data_o, wcs_o),
                lr_target_wcs,
                shape_out=(AI_LR_SIZE, AI_LR_SIZE),
                order='bilinear'
            )
            patch_o_lr_raw = np.nan_to_num(patch_o_lr_raw)

            if np.sum(patch_o_lr_raw) < 1.0: continue

            # 4. AUTO-ALLINEAMENTO (Cross-Correlation)
            # Ridimensioniamo Hubble per matchare la LR
            h_small = resize(patch_h, (AI_LR_SIZE, AI_LR_SIZE), anti_aliasing=True)
            
            # Normalizziamo per il confronto
            h_norm = (h_small - h_small.min()) / (h_small.max() - h_small.min() + 1e-6)
            o_norm = (patch_o_lr_raw - patch_o_lr_raw.min()) / (patch_o_lr_raw.max() - patch_o_lr_raw.min() + 1e-6)
            
            # Calcoliamo lo shift (dy, dx)
            shift, error, diffphase = phase_cross_correlation(h_norm, o_norm, upsample_factor=10)
            dy, dx = shift
            
            # Se lo shift è ragionevole, correggiamo il WCS e riproiettiamo
            shift_applied = (0.0, 0.0)
            patch_final = patch_o_lr_raw
            
            if abs(dy) < MAX_SHIFT_PIXELS and abs(dx) < MAX_SHIFT_PIXELS:
                # Applichiamo lo shift al CRPIX del target WCS
                # Se dx > 0, l'immagine Obs è spostata a destra -> dobbiamo spostare il centro a destra
                lr_target_wcs_corrected = lr_target_wcs.deepcopy()
                lr_target_wcs_corrected.wcs.crpix[0] -= dx 
                lr_target_wcs_corrected.wcs.crpix[1] -= dy
                
                # Riproiezione FINALE (Fine)
                patch_final, _ = reproject_interp(
                    (data_o, wcs_o),
                    lr_target_wcs_corrected,
                    shape_out=(AI_LR_SIZE, AI_LR_SIZE),
                    order='bilinear'
                )
                patch_final = np.nan_to_num(patch_final)
                shift_applied = (dy, dx)
                
                # Aggiorniamo il WCS finale per il salvataggio
                lr_target_wcs = lr_target_wcs_corrected

            # 5. Salvataggio
            with log_lock:
                idx = patch_index_counter
                patch_index_counter += 1
            
            pair_dir = shared_data['out_fits'] / f"pair_{idx:06d}"
            pair_dir.mkdir(exist_ok=True)
            
            fits.PrimaryHDU(patch_h.astype(np.float32), header=patch_h_wcs.to_header()).writeto(pair_dir/"hubble.fits", overwrite=True)
            # Aggiungiamo lo shift all'header per tracciabilità
            hdr_lr = lr_target_wcs.to_header()
            hdr_lr['SHIFT_Y'] = (shift_applied[0], 'Auto-alignment shift Y')
            hdr_lr['SHIFT_X'] = (shift_applied[1], 'Auto-alignment shift X')
            
            fits.PrimaryHDU(patch_final.astype(np.float32), header=hdr_lr).writeto(pair_dir/"observatory.fits", overwrite=True)
            
            saved_count += 1
            
            # Debug PNG
            if idx < DEBUG_SAMPLES:
                try:
                    obs_scale = get_pixel_scale_deg(wcs_o)
                    raw_size = int(shared_data['h_fov_deg'] / obs_scale)
                    png_path = shared_data['out_png'] / f"check_pair_{idx:06d}.jpg"
                    
                    save_diagnostic_card(
                        data_h, data_o,
                        patch_h, patch_final,
                        x, y, wcs_h, wcs_o,
                        lr_target_wcs, raw_size, shared_data['h_fov_deg'], 
                        png_path, shift_applied
                    )
                except Exception: pass

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
    print(f"🚀 ESTRAZIONE DINAMICA + AUTO-ALLINEAMENTO PATCH")
    
    target_dir = ROOT_DATA_DIR / "M1" 
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
    
    h_files = sorted(list(input_h.glob("*.fits")))
    o_files_all = sorted(list(input_o.glob("*.fits")))
    
    if not h_files or not o_files_all:
        print("❌ File mancanti in 3_registered_native")
        return

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
        
    except Exception as e:
        print(f"❌ Errore lettura Hubble: {e}")
        return

    o_files_good = []
    for f in o_files_all:
        try:
            with fits.open(f) as o:
                w = WCS(o[0].header)
                dist = np.sqrt((w.wcs.crval[0]-h_center[0])**2 + (w.wcs.crval[1]-h_center[1])**2)
                if dist < 0.1: 
                    o_files_good.append(f)
        except: pass
        
    print(f"   ✅ File Obs allineati: {len(o_files_good)}")
    
    h_h, h_w = d_h.shape
    tasks = []
    for y in range(0, h_h - HR_SIZE + 1, STRIDE):
        for x in range(0, h_w - HR_SIZE + 1, STRIDE):
            tasks.append((h_master_path, y, x))
            
    print(f"   📦 Patch Hubble da processare: {len(tasks)}")
    print(f"\n🚀 Avvio estrazione e allineamento fine...")
    total_saved = 0
    
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