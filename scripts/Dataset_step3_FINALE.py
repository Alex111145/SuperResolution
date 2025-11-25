"""
STEP 3 (FINAL): MULTI-PROCESS VERSION + ZIP CON NOME TARGET
------------------------------------------------------------------------
OTTIMIZZAZIONE:
- Esecuzione PARALLELA su tutti i core della CPU.
- Matplotlib backend 'Agg' per rendering thread-safe.
- Zip automatico della cartella PNG con nome del target.
- VISUALIZZAZIONE PULITA: 2 Mosaici Contesto + 3 Patch Dettaglio.
- AUMENTO DATI: Stride ridotto (16px) per massimizzare le patch.
- PROGRESS BAR: Attiva durante l'elaborazione parallela.

INPUT: 
  - final_mosaic_hubble.fits (HR Reference Grid)
  - final_mosaic_observatory.fits (LR Input)
  
OUTPUT: 
  - Dati Training: 6_patches_final/pair_XXXXX/ (FITS)
  - Controllo Visivo: coppie_patch_png/pair_XXXXX_context.png (PNG)
  - Archivio: coppie_patch_png_NOMETAARGET.zip
------------------------------------------------------------------------
"""

import os
import sys
import shutil
import numpy as np
import matplotlib
# Imposta il backend non interattivo PRIMA di importare pyplot per il multiprocessing
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import PercentileInterval
from skimage.transform import resize
from tqdm import tqdm
import warnings
import subprocess
from scipy.ndimage import maximum_filter
from concurrent.futures import ProcessPoolExecutor
from reproject import reproject_interp 

warnings.filterwarnings('ignore')

# ================= CONFIGURAZIONE GLOBALE =================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# PARAMETRI DATASET (MASSIMIZZAZIONE DATI)
HR_SIZE = 512         
LR_SIZE = 80          
STRIDE = 16         # <--- STRIDE BASSO PER MOLTE PATCH

# SOGLIE DI QUALITÀ
MIN_COVERAGE = 0.97      # <--- TOLLERANZA ALTA (accetta un po' di bordo nero)
MIN_PIXEL_VALUE = 0.0001 
SAVE_MAP_EVERY_N = 1    # Salva un PNG ogni 20 patch per non rallentare troppo

# PARAMETRI VISUALIZZAZIONE STELLE
PEAK_THRESHOLD = 0.4      
ALIGNMENT_THRESHOLD = 0.4 
MARKER_COLOR = 'yellow'   
MARKER_SIZE = 50          
# ==========================================================

# Variabile globale per i processi worker
shared_data = {}

# --- FUNZIONI DI UTILITY ---

def get_robust_normalization(data: np.ndarray) -> tuple[float, float]:
    data = np.nan_to_num(data)
    valid_mask = data > MIN_PIXEL_VALUE
    if np.sum(valid_mask) < 100: return 0.0, 1.0
    valid_pixels = data[valid_mask]
    interval = PercentileInterval(99.5)
    vmin, vmax = interval.get_limits(valid_pixels)
    if vmax <= vmin: return np.min(valid_pixels), np.max(valid_pixels)
    return vmin, vmax

def normalize_with_stretch(data: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    data = np.nan_to_num(data)
    if vmax <= vmin: return np.zeros_like(data)
    clipped = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    return np.sqrt(clipped) 

def normalize_local_stretch(data: np.ndarray) -> np.ndarray:
    try:
        data = np.nan_to_num(data)
        interval = PercentileInterval(99.5)
        vmin, vmax = interval.get_limits(data)
        if vmax <= vmin: return np.zeros_like(data)
        clipped = np.clip((data - vmin) / (vmax - vmin), 0, 1)
        return np.sqrt(clipped)
    except:
        return np.zeros_like(data)

def find_aligned_stars(patch_h: np.ndarray, patch_o_in: np.ndarray) -> tuple:
    tar_n = normalize_local_stretch(patch_h)
    inp_s = resize(normalize_local_stretch(patch_o_in), (HR_SIZE, HR_SIZE), order=0)
    overlap_intensity = np.minimum(inp_s, tar_n)
    footprint = np.ones((3, 3))
    local_max = maximum_filter(tar_n, footprint=footprint)
    is_peak = (tar_n == local_max) & (tar_n > PEAK_THRESHOLD * tar_n.max())
    aligned_mask = is_peak & (overlap_intensity > ALIGNMENT_THRESHOLD)
    y, x = np.where(aligned_mask)
    return x, y

# ==================================================================================
# FUNZIONE DI VISUALIZZAZIONE AGGIORNATA (Pulita: Solo Config e Stats)
# ==================================================================================
def save_8panel_card(mosaic_h, mosaic_o_raw, 
                     patch_h, patch_o_in, 
                     x, y, wcs_h, wcs_o_raw,
                     vmin_h, vmax_h, vmin_o, vmax_o, save_path):
    
    fig = plt.figure(figsize=(24, 12)) 
    coverage_perc = np.count_nonzero(patch_h > MIN_PIXEL_VALUE) / (HR_SIZE * HR_SIZE) * 100
    x_stars, y_stars = find_aligned_stars(patch_h, patch_o_in)
    
    # Griglia 2x4.
    gs = fig.add_gridspec(2, 4)
    sc = 10 
    
    target_name = save_path.parent.parent.parent.name if save_path.parent.parent.parent.name else "N/A"
    fig.suptitle(f"DATASET SAMPLE: Target {target_name} | X={x} Y={y} | Coverage: {coverage_perc:.1f}%", 
                 fontsize=18, fontweight='bold')

    def plot_mosaic_context(ax, data, title, box_col, v_min, v_max, wcs_curr, wcs_h):
        display_data = data[::sc, ::sc]
        ax.imshow(normalize_with_stretch(display_data, v_min, v_max), origin='lower', cmap='gray')
        ax.set_title(title, color='black', fontsize=12)
        ax.axis('off')
        
        try:
            corners_pix_h = np.array([[x, y], [x+HR_SIZE, y], [x+HR_SIZE, y+HR_SIZE], [x, y+HR_SIZE]])
            corners_world = wcs_h.pixel_to_world(corners_pix_h[:,0], corners_pix_h[:,1])
            px_curr = wcs_curr.world_to_pixel(corners_world)
            poly_coords = np.column_stack((px_curr[0]/sc, px_curr[1]/sc))
            rect = patches.Polygon(poly_coords, linewidth=2, edgecolor=box_col, facecolor='none')
            ax.add_patch(rect)
        except: pass

    ### RIGA 1: MOSAICI CENTRATI ###
    plot_mosaic_context(fig.add_subplot(gs[0, 1]), mosaic_h, "**1. Hubble Master (HR)**", 'lime', vmin_h, vmax_h, wcs_h, wcs_h)
    plot_mosaic_context(fig.add_subplot(gs[0, 2]), mosaic_o_raw, "**2. Obs Master (LR, Raw WCS)**", 'cyan', vmin_o, vmax_o, wcs_o_raw, wcs_h)
    
    ### RIGA 2: PATCH (Dettaglio Training) ###
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(normalize_local_stretch(patch_h), origin='lower', cmap='gray')
    ax5.scatter(x_stars, y_stars, s=MARKER_SIZE, facecolors='none', edgecolors=MARKER_COLOR, marker='o', linewidths=1.5, alpha=0.9)
    ax5.set_title("**3. Hubble Target (HR) / Ground Truth**", color='black')
    ax5.axis('off')

    x_stars_lr = x_stars * (LR_SIZE / HR_SIZE)
    y_stars_lr = y_stars * (LR_SIZE / HR_SIZE)
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(normalize_local_stretch(patch_o_in), origin='lower', cmap='gray')
    ax6.scatter(x_stars_lr, y_stars_lr, s=MARKER_SIZE, facecolors='none', edgecolors=MARKER_COLOR, marker='o', linewidths=1.5, alpha=0.9)
    ax6.set_title("**4. Obs Input (LR) / Aligned**", color='black')
    ax6.axis('off')

    ax7 = fig.add_subplot(gs[1, 2])
    inp_s = resize(normalize_local_stretch(patch_o_in), (HR_SIZE, HR_SIZE), order=0)
    tar_n = normalize_local_stretch(patch_h)
    rgb = np.zeros((HR_SIZE, HR_SIZE, 3))
    rgb[..., 0] = inp_s * 0.9 
    rgb[..., 1] = tar_n * 0.9 
    ax7.imshow(rgb, origin='lower')
    ax7.scatter(x_stars, y_stars, s=MARKER_SIZE, facecolors='none', edgecolors=MARKER_COLOR, marker='o', linewidths=1.5, alpha=0.9)
    ax7.set_title("**5. RGB Overlay (LR vs HR)**", color='black')
    ax7.axis('off')

    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis('off')
    txt = (f"CONFIG PATCH:\n"
           f"- HR Size: {HR_SIZE}x{HR_SIZE}\n"
           f"- LR Size: {LR_SIZE}x{LR_SIZE}\n"
           f"- Stride:  {STRIDE}\n\n"
           f"STATISTICHE:\n"
           f"- Copertura Pixel (>0): {coverage_perc:.2f}%\n"
           f"- Stelle Allineate: {len(x_stars)}")
    ax8.text(0.1, 0.5, txt, fontsize=14, family='monospace', verticalalignment='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=90)
    plt.close(fig)

# --- WORKER SETUP & JOB ---

def init_worker(d_h, d_o_raw, hdr_h, w_h, w_o_raw, v_h, V_h, v_o, V_o, out_fits, out_png):
    shared_data['h'] = d_h
    shared_data['o_raw'] = d_o_raw
    shared_data['header_h'] = hdr_h
    shared_data['wcs_h'] = w_h
    shared_data['wcs_o_raw'] = w_o_raw
    shared_data['vmin_h'] = v_h
    shared_data['vmax_h'] = V_h
    shared_data['vmin_o'] = v_o
    shared_data['vmax_o'] = V_o
    shared_data['out_fits'] = out_fits
    shared_data['out_png'] = out_png

def create_lr_wcs(hr_wcs, hr_shape, lr_shape):
    scale_factor = hr_shape[0] / lr_shape[0]
    lr_wcs = hr_wcs.deepcopy()
    lr_wcs.wcs.cdelt[0] *= scale_factor 
    lr_wcs.wcs.cdelt[1] *= scale_factor
    lr_wcs.wcs.crpix[0] /= scale_factor
    lr_wcs.wcs.crpix[1] /= scale_factor
    return lr_wcs

def process_single_patch(args):
    y, x, idx = args
    data_h = shared_data['h']
    data_o_raw = shared_data['o_raw']
    wcs_h = shared_data['wcs_h']
    wcs_o_raw = shared_data['wcs_o_raw']
    
    # 1. Check rapido copertura (prima della riproiezione costosa)
    patch_h = data_h[y:y+HR_SIZE, x:x+HR_SIZE]
    valid_px = np.count_nonzero(patch_h > MIN_PIXEL_VALUE)
    if (valid_px / patch_h.size) < MIN_COVERAGE:
        return False 
    
    # 2. Riproiezione
    hr_patch_wcs = wcs_h.deepcopy()
    hr_patch_wcs.wcs.crpix = hr_patch_wcs.wcs.crpix - np.array([x, y])
    lr_target_wcs = create_lr_wcs(hr_patch_wcs, (HR_SIZE, HR_SIZE), (LR_SIZE, LR_SIZE))
    
    try:
        patch_o_lr, _ = reproject_interp(
            (data_o_raw, wcs_o_raw),
            lr_target_wcs,
            shape_out=(LR_SIZE, LR_SIZE),
            order='bilinear'
        )
        patch_o_lr = np.nan_to_num(patch_o_lr).astype(np.float32)
    except Exception as e:
        return False
    
    # 3. Salvataggio
    pair_path = shared_data['out_fits'] / f"pair_{idx:05d}"
    pair_path.mkdir(exist_ok=True)
    
    header_h = shared_data['header_h']
    h_hr = hr_patch_wcs.to_header()
    fits.PrimaryHDU(patch_h.astype(np.float32), header=h_hr).writeto(pair_path/"hubble.fits", overwrite=True)
    h_lr = lr_target_wcs.to_header()
    fits.PrimaryHDU(patch_o_lr, header=h_lr).writeto(pair_path/"observatory.fits", overwrite=True)
    
    if idx % SAVE_MAP_EVERY_N == 0:
        save_path = shared_data['out_png'] / f"pair_{idx:05d}_context.png"
        save_8panel_card(
            data_h, shared_data['o_raw'],
            patch_h, patch_o_lr,
            x, y, shared_data['wcs_h'], shared_data['wcs_o_raw'],
            shared_data['vmin_h'], shared_data['vmax_h'], shared_data['vmin_o'], shared_data['vmax_o'],
            save_path
        )
    
    return True

# ================= MAIN PROCESSING =================

def create_dataset_filtered(base_dir: Path) -> tuple[bool, Path] | None:
    
    mosaics_dir = base_dir / '5_mosaics'
    output_fits_dir = base_dir / '6_patches_final'      
    output_png_dir = base_dir / 'coppie_patch_png'      
    
    f_h_raw   = mosaics_dir / 'final_mosaic_hubble.fits'
    f_o_raw   = mosaics_dir / 'final_mosaic_observatory.fits'

    missing = [f.name for f in [f_h_raw, f_o_raw] if not f.exists()]
    if missing:
        print(f"\n❌ ERRORE: Mancano i file: {missing}")
        return None

    if output_fits_dir.exists(): shutil.rmtree(output_fits_dir)
    output_fits_dir.mkdir(parents=True, exist_ok=True)
    if output_png_dir.exists(): shutil.rmtree(output_png_dir)
    output_png_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n⚙️  Caricamento Dati Main Process...")
    try:
        def load_fits(p):
            with fits.open(p) as h:
                d = np.nan_to_num(h[0].data)
                if d.ndim > 2: d = d[0]
                return d, WCS(h[0].header), h[0].header

        data_h, wcs_h, header_h = load_fits(f_h_raw)
        data_o_raw, wcs_o_raw, _ = load_fits(f_o_raw)
        
    except Exception as e:
        print(f"❌ Errore caricamento FITS: {e}")
        return None

    print("⚖️  Calcolo livelli di normalizzazione...")
    vmin_h, vmax_h = get_robust_normalization(data_h)
    vmin_o, vmax_o = get_robust_normalization(data_o_raw)

    h_dim, w_dim = data_h.shape
    y_list = list(range(0, h_dim - HR_SIZE + 1, STRIDE))
    x_list = list(range(0, w_dim - HR_SIZE + 1, STRIDE))
    
    tasks = []
    task_id = 0
    # Calcolo preliminare dei task (veloce)
    for y in y_list:
        for x in x_list:
            tasks.append((y, x, task_id))
            task_id += 1
            
    print(f"\n🚀 Avvio Processing Parallelo (Stride={STRIDE}) su {os.cpu_count()} core...")
    print(f"   Totale Patch Candidate da analizzare: {len(tasks)}")
    print("   (La barra di avanzamento partirà ora...)")

    processed_count = 0
    
    # ESECUZIONE PARALLELA CON BARRA DI AVANZAMENTO
    with ProcessPoolExecutor(max_workers=os.cpu_count(), 
                             initializer=init_worker, 
                             initargs=(data_h, data_o_raw, 
                                       header_h, wcs_h, wcs_o_raw, 
                                       vmin_h, vmax_h, vmin_o, vmax_o, 
                                       output_fits_dir, output_png_dir)) as executor:
        
        # tqdm avvolge l'iteratore dei risultati
        results = list(tqdm(executor.map(process_single_patch, tasks), 
                            total=len(tasks), 
                            unit="patch",
                            desc="Elaborazione",
                            ncols=100))
        processed_count = sum(results)

    # === ZIP AUTOMATICO PNG CON NOME TARGET ===
    target_name = base_dir.name
    zip_filename = f"coppie_patch_png_{target_name}"
    
    print(f"\n📦 Creazione archivio ZIP '{zip_filename}.zip'...")
    zip_path = base_dir / zip_filename
    shutil.make_archive(str(zip_path), 'zip', output_png_dir)
    
    # ================= RIEPILOGO FINALE =================
    print("\n" + "="*50)
    print(f"✅ COMPLETATO! REPORT FINALE PER: {target_name}")
    print("="*50)
    print(f"🔢 STRIDE UTILIZZATO: {STRIDE} px")
    print(f"🖼️  COPPIE GENERATE:  {processed_count}")
    print("="*50)
    print(f"📂 Dataset: {output_fits_dir}")
    print(f"🗜️  Zip PNG: {zip_path}.zip")
    
    return True, base_dir

def select_target_directory():
    subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs']]
    if not subdirs: return None
    print("\nSELEZIONA TARGET:")
    for i, d in enumerate(subdirs): print(f" {i+1}: {d.name}")
    try:
        idx = int(input("Scelta: ")) - 1
        return subdirs[idx] if 0 <= idx < len(subdirs) else None
    except: return None

def ask_continue(base_dir):
    next_script = CURRENT_SCRIPT_DIR / 'Modello_2_pre_da_usopatch_dataset_step3.py'
    print(f"\nVuoi lanciare lo split dataset? (Script: {next_script.name})")
    if input("S/n: ").lower() in ['s', '', 'y']:
        subprocess.run([sys.executable, str(next_script.resolve()), str(base_dir.resolve())])

if __name__ == "__main__":
    target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else select_target_directory()
    if target:
        res = create_dataset_filtered(target)
        if res:
            ask_continue(res[1])