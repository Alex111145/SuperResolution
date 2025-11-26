import os
import sys
import shutil
import numpy as np
import matplotlib
# Imposta il backend non interattivo per massimizzare le risorse
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from skimage.transform import resize
from skimage.feature import match_template, peak_local_max
from skimage import exposure
from scipy.ndimage import median_filter
from tqdm import tqdm
import warnings
from concurrent.futures import ProcessPoolExecutor
from reproject import reproject_interp 
import threading 
import traceback

warnings.filterwarnings('ignore')

# ================= CONFIGURAZIONE AVANZATA =================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# PARAMETRI DATASET (Ottimizzati per M33)
HR_SIZE = 512           # Dimensione Hubble
AI_LR_SIZE = 128        # Dimensione Osservatorio (Input Rete)
STRIDE = 8              # Salto tra una patch e l'altra (Genera circa 2600+ patch)
MIN_COVERAGE = 0.40     # Minimo % di pixel validi (Cattura anche i bordi)
MIN_PIXEL_VALUE = 0.0001 

# PARAMETRI "SEARCH & LOCK"
SEARCH_FACTOR = 3.0     # Cerca in un'area 3 volte più grande
MIN_CORRELATION = 0.35  # Soglia di somiglianza

log_lock = threading.Lock()
# ==========================================================

# Variabile globale per i processi worker
shared_data = {}
patch_index_counter = 0

# --- FUNZIONI DI UTILITY & VISUALIZZAZIONE ---

def get_pixel_scale_deg(wcs):
    scales = proj_plane_pixel_scales(wcs)
    return np.mean(scales)

def get_robust_preview(data, size=None):
    try:
        data = np.nan_to_num(data)
        data = median_filter(data, size=3)
        vmin, vmax = data.min(), data.max()
        if vmax - vmin > 0:
            data = (data - vmin) / (vmax - vmin)
        data = exposure.equalize_adapthist(data, clip_limit=0.05)
        if size:
            # Upscale Bicubico (order=3) per nitidezza
            return resize(data, (size, size), order=3, anti_aliasing=(size < data.shape[0]))
        return data
    except:
        return np.zeros((size, size)) if size else np.zeros_like(data)

def create_custom_wcs(ref_wcs, center_pix, new_size, fov_deg):
    sky_center = ref_wcs.pixel_to_world(center_pix[0], center_pix[1])
    scale = fov_deg / new_size
    w = WCS(naxis=2)
    w.wcs.crval = [sky_center.ra.deg, sky_center.dec.deg]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [new_size / 2.0, new_size / 2.0]
    w.wcs.cdelt = [-scale, scale]
    w.wcs.pc = np.eye(2)
    return w

def find_common_anchors(img_h, img_o, tolerance=15):
    """
    Trova picchi luminosi (stelle/nodi) presenti in ENTRAMBE le immagini.
    SETTINGS MODIFICATI: MOLTO PIÙ SENSIBILE
    """
    peaks_h = peak_local_max(img_h, min_distance=1, threshold_rel=0.02, num_peaks=1000)
    peaks_o = peak_local_max(img_o, min_distance=1, threshold_rel=0.02, num_peaks=1000)

    common_points = []
    
    # Matching
    for yh, xh in peaks_h:
        if len(peaks_o) == 0: break
        dists = np.sqrt((peaks_o[:, 0] - yh)**2 + (peaks_o[:, 1] - xh)**2)
        idx_best = np.argmin(dists)
        if dists[idx_best] < tolerance:
            yo, xo = peaks_o[idx_best]
            # Salva le coordinate per la patch HR (512x512)
            common_points.append((xh, yh)) 
    return common_points

# --- DEBUG PNG ---
def save_diagnostic_card(patch_h, patch_o, 
                         search_area_o, template_match_val,
                         save_path, shift_applied):
    try:
        fig = plt.figure(figsize=(18, 10)) 
        gs = fig.add_gridspec(2, 3)
        BOX_COLOR = '#9932CC' # Viola

        prev_h = get_robust_preview(patch_h, size=HR_SIZE)
        prev_o = get_robust_preview(patch_o, size=HR_SIZE) 
        prev_search = get_robust_preview(search_area_o, size=HR_SIZE)

        # RIGA 1
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(prev_h, origin='lower', cmap='inferno')
        ax1.set_title("1. Hubble Target", color='red')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(prev_search, origin='lower', cmap='viridis')
        
        sh, sw = search_area_o.shape
        cy, cx = sh//2, sw//2
        dy, dx = shift_applied
        scale_factor = HR_SIZE / sh 
        rect_x = (cx + dx - AI_LR_SIZE/2) * scale_factor
        rect_y = (cy + dy - AI_LR_SIZE/2) * scale_factor
        rect_w = AI_LR_SIZE * scale_factor
        
        rect = patches.Rectangle((rect_x, rect_y), rect_w, rect_w, 
                                 linewidth=3, edgecolor=BOX_COLOR, facecolor='none')
        ax2.add_patch(rect)
        ax2.set_title("2. Obs Search Area", color=BOX_COLOR)
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        
        # Calcolo ancore qui per averne il numero
        common_anchors = find_common_anchors(prev_h, prev_o)
        
        txt_info = (f"🔍 SEARCH & LOCK DEBUG\n"
                    f"PAIR ID: {save_path.stem}\n"
                    f"SCORE: {template_match_val:.3f} (Req: {MIN_CORRELATION})\n"
                    f"STATUS: ✅ GOOD MATCH\n\n"
                    f"🟡 ANCHORS FOUND: {len(common_anchors)}")
        ax3.text(0.05, 0.5, txt_info, fontsize=14, va='center', color='green', fontweight='bold', family='monospace')

        # RIGA 2
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow(prev_h, origin='lower', cmap='inferno')
        if common_anchors:
            xs, ys = zip(*common_anchors)
            ax4.scatter(xs, ys, c='yellow', s=80, edgecolors='black', marker='o') # Pallini leggermente più piccoli
        border_h = patches.Rectangle((2, 2), HR_SIZE-4, HR_SIZE-4, linewidth=4, edgecolor=BOX_COLOR, facecolor='none')
        ax4.add_patch(border_h)
        ax4.set_title("Final Hubble", color=BOX_COLOR)
        ax4.axis('off')

        ax5 = fig.add_subplot(gs[1, 1])
        ax5.imshow(prev_o, origin='lower', cmap='viridis')
        if common_anchors:
            xs, ys = zip(*common_anchors)
            ax5.scatter(xs, ys, c='yellow', s=80, edgecolors='black', marker='o')
        border_o = patches.Rectangle((2, 2), HR_SIZE-4, HR_SIZE-4, linewidth=4, edgecolor=BOX_COLOR, facecolor='none')
        ax5.add_patch(border_o)
        ax5.set_title(f"Final Obs", color=BOX_COLOR)
        ax5.axis('off')

        ax6 = fig.add_subplot(gs[1, 2])
        rgb = np.zeros((HR_SIZE, HR_SIZE, 3))
        rgb[..., 0] = prev_h 
        rgb[..., 1] = prev_o
        ax6.imshow(rgb, origin='lower')
        if common_anchors:
            xs, ys = zip(*common_anchors)
            ax6.scatter(xs, ys, c='yellow', s=80, edgecolors='black', marker='o')
        ax6.set_title("Overlay", color='yellow')
        ax6.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
    except Exception as e:
        print(f"Error saving PNG: {e}")
        traceback.print_exc()

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

def process_single_patch_search_and_lock(args):
    global patch_index_counter
    h_path, y, x = args
    
    data_h = shared_data['h']
    wcs_h = shared_data['wcs_h']
    out_fits_dir = shared_data['out_fits']
    out_png_dir = shared_data['out_png']

    patch_h = data_h[y:y+HR_SIZE, x:x+HR_SIZE]
    if np.count_nonzero(patch_h > MIN_PIXEL_VALUE) / patch_h.size < MIN_COVERAGE:
        return 0

    center_x_h = x + HR_SIZE / 2.0
    center_y_h = y + HR_SIZE / 2.0

    h_template = resize(patch_h, (AI_LR_SIZE, AI_LR_SIZE), anti_aliasing=True)
    h_template_norm = (h_template - h_template.mean()) / (h_template.std() + 1e-6)

    SEARCH_SIZE = int(AI_LR_SIZE * SEARCH_FACTOR) 
    wcs_search = create_custom_wcs(wcs_h, (center_x_h, center_y_h), SEARCH_SIZE, shared_data['h_fov_deg'] * SEARCH_FACTOR)
    
    saved_count = 0
    
    for o_path in shared_data['o_files']:
        try:
            with fits.open(o_path) as o:
                data_o = np.nan_to_num(o[0].data)
                if data_o.ndim > 2: data_o = data_o[0]
                wcs_o = WCS(o[0].header)

            search_area, _ = reproject_interp(
                (data_o, wcs_o), wcs_search, shape_out=(SEARCH_SIZE, SEARCH_SIZE), order='bilinear'
            )
            search_area = np.nan_to_num(search_area)
            if np.max(search_area) <= 0: continue

            search_norm = (search_area - search_area.mean()) / (search_area.std() + 1e-6)
            result = match_template(search_norm, h_template_norm, pad_input=True)
            ij = np.unravel_index(np.argmax(result), result.shape)
            best_y, best_x = ij
            max_corr = result[best_y, best_x]
            
            # FILTRO CORRELAZIONE
            if max_corr < MIN_CORRELATION:
                continue

            shift_y = best_y - SEARCH_SIZE//2
            shift_x = best_x - SEARCH_SIZE//2
            
            wcs_final = create_custom_wcs(wcs_h, (center_x_h, center_y_h), AI_LR_SIZE, shared_data['h_fov_deg'])
            wcs_final.wcs.crpix[0] -= shift_x
            wcs_final.wcs.crpix[1] -= shift_y
            
            patch_final, _ = reproject_interp(
                (data_o, wcs_o), wcs_final, shape_out=(AI_LR_SIZE, AI_LR_SIZE), order='bilinear'
            )
            patch_final = np.nan_to_num(patch_final)

            with log_lock:
                idx = patch_index_counter
                patch_index_counter += 1
            
            pair_dir = out_fits_dir / f"pair_{idx:06d}"
            pair_dir.mkdir(exist_ok=True)
            
            h_wcs_crop = shared_data['wcs_h'].deepcopy()
            h_wcs_crop.wcs.crpix -= np.array([x, y])
            fits.PrimaryHDU(patch_h.astype(np.float32), header=h_wcs_crop.to_header()).writeto(pair_dir/"hubble.fits", overwrite=True)
            
            obs_hdr = wcs_final.to_header()
            obs_hdr['MATCH_SC'] = (max_corr, "Template Matching Score")
            fits.PrimaryHDU(patch_final.astype(np.float32), header=obs_hdr).writeto(pair_dir/"observatory.fits", overwrite=True)
            
            saved_count += 1
            # === MODIFICA QUI: Salva PNG solo se l'indice è multiplo di 10 ===
            if idx % 10 == 0:
                png_path = out_png_dir / f"check_pair_{idx:06d}.jpg"
                save_diagnostic_card(patch_h, patch_final, search_area, max_corr, png_path, (shift_y, shift_x))
            # ================================================================

            
            png_path = out_png_dir / f"check_pair_{idx:06d}.jpg"
            save_diagnostic_card(patch_h, patch_final, search_area, max_corr, png_path, (shift_y, shift_x))
            
            break 

        except Exception: continue
            
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
    print(f"🚀 ESTRAZIONE PATCH v6: HIGH SENSITIVITY ANCHORS")
    
    # Rilevamento automatico CPU per BOOST
    max_cpu = os.cpu_count()
    print(f"🔥 BOOST ATTIVO: Utilizzo di {max_cpu} CPU in parallelo + Max RAM disponibile.")
    
    target_dir = ROOT_DATA_DIR / "M33" 
    if len(sys.argv) > 1: target_dir = Path(sys.argv[1])
    else:
        sel = select_target_directory()
        if sel: target_dir = sel
    
    print(f"\n📂 Target: {target_dir.name}")
    # Nota: Path di Hubble e Obs sono rimasti originali nello script fornito
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
    
    if not h_files or not o_files_all: return

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
    except Exception: return

    o_files_good = []
    for f in o_files_all:
        try:
            with fits.open(f) as o:
                w = WCS(o[0].header)
                dist = np.sqrt((w.wcs.crval[0]-h_center[0])**2 + (w.wcs.crval[1]-h_center[1])**2)
                if dist < 0.6: o_files_good.append(f)
        except: pass
        
    tasks = []
    h_h, h_w = d_h.shape
    for y in range(0, h_h - HR_SIZE + 1, STRIDE):
        for x in range(0, h_w - HR_SIZE + 1, STRIDE):
            tasks.append((h_master_path, y, x))
            
    print(f"   🔍 Avvio (Soglia stelle ridotta per più pallini)...")
    
    # MODIFICA BOOST: max_workers=max_cpu invece di 6
    with ProcessPoolExecutor(max_workers=max_cpu, initializer=init_worker,
                             initargs=(d_h, h_head, w_h, out_fits, out_png, h_fov_deg, o_files_good)) as ex:
        results = list(tqdm(ex.map(process_single_patch_search_and_lock, tasks), total=len(tasks), ncols=100))
        
    print(f"\n✅ COMPLETATO. Controlla {out_png}")

    # ================= ZIPPING AUTOMATICO MODIFICATO =================
    print(f"\n📦 Creazione archivi ZIP...")
    
    # 1. ZIP dei dati FITS
    zip_name_fits = target_dir / f"{target_dir.name}_dataset_fits"
    try:
        shutil.make_archive(str(zip_name_fits), 'zip', str(out_fits))
        print(f"✅ Archivio FITS salvato correttamente in: {zip_name_fits}.zip")
    except Exception as e:
        print(f"❌ Errore durante la creazione dello ZIP FITS: {e}")
        
    # 2. ZIP delle immagini PNG di debug
    zip_name_png = target_dir / f"{target_dir.name}_debug_png"
    try:
        shutil.make_archive(str(zip_name_png), 'zip', str(out_png))
        print(f"✅ Archivio PNG salvato correttamente in: {zip_name_png}.zip")
    except Exception as e:
        print(f"❌ Errore durante la creazione dello ZIP PNG: {e}")
        
    # =================================================================
    
if __name__ == "__main__":
    main()