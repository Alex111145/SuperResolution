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
from astropy.visualization import ZScaleInterval
from skimage.transform import resize
from tqdm import tqdm
import warnings
from concurrent.futures import ProcessPoolExecutor
from reproject import reproject_interp 
import threading

warnings.filterwarnings('ignore')

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

HR_SIZE = 512 
AI_LR_SIZE = 128
STRIDE = 150
MIN_COVERAGE = 0.50 
MIN_PIXEL_VALUE = 0.0001 
DEBUG_SAMPLES = 50

log_lock = threading.Lock()

shared_data = {}
patch_index_counter = 0

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

def calculate_wcs_corners(wcs, size):
    corner_pixels = np.array([
        [0, 0], [size, 0], [0, size], [size, size]
    ])
    world_coords = wcs.pixel_to_world(corner_pixels[:, 0], corner_pixels[:, 1])
    
    center_world = wcs.pixel_to_world(size/2, size/2)
    return center_world.ra.deg, center_world.dec.deg

def save_diagnostic_card(data_h_orig, data_o_raw_orig, 
                         patch_h, patch_o_lr, 
                         x, y, wcs_h, wcs_o_raw,
                         lr_wcs_target, 
                         h_fov_deg, save_path):
    
    try:
        fig = plt.figure(figsize=(20, 12), facecolor='#1e1e1e') 
        gs = fig.add_gridspec(2, 3)

        h_patch_wcs = wcs_h.deepcopy()
        h_patch_wcs.wcs.crpix -= np.array([x, y])
        
        h_ra, h_dec = calculate_wcs_corners(h_patch_wcs, HR_SIZE)
        lr_ra, lr_dec = calculate_wcs_corners(lr_wcs_target, AI_LR_SIZE)
        
        mismatch_ra = abs(h_ra - lr_ra) * 3600
        mismatch_dec = abs(h_dec - lr_dec) * 3600

        lr_corners_pix = np.array([
            [0, 0], [AI_LR_SIZE, 0], [AI_LR_SIZE, AI_LR_SIZE], [0, AI_LR_SIZE]
        ])
        lr_corners_world = lr_wcs_target.pixel_to_world(lr_corners_pix[:, 0], lr_corners_pix[:, 1])
        obs_corners_pix_raw = wcs_o_raw.world_to_pixel(lr_corners_world)
        
        scale_ox = 512 / data_o_raw_orig.shape[1]
        scale_oy = 512 / data_o_raw_orig.shape[0]
        polygon_verts = np.stack([obs_corners_pix_raw[0] * scale_ox, obs_corners_pix_raw[1] * scale_oy], axis=1)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        h_small = get_robust_preview(data_h_orig, 512)
        ax1.imshow(h_small, origin='lower', cmap='inferno')
        ax1.set_title("GLOBAL HUBBLE MAP", color='cyan', fontweight='bold')
        
        scale_hy = 512 / data_h_orig.shape[0]
        scale_hx = 512 / data_h_orig.shape[1]
        rect_h = patches.Rectangle((x*scale_hx, y*scale_hy), HR_SIZE*scale_hx, HR_SIZE*scale_hy, 
                                   linewidth=2, edgecolor='cyan', facecolor='none')
        ax1.add_patch(rect_h)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        o_small = get_robust_preview(data_o_raw_orig, 512)
        ax2.imshow(o_small, origin='lower', cmap='viridis')
        ax2.set_title("GLOBAL OBS MAP", color='lime', fontweight='bold')
        poly_o = patches.Polygon(polygon_verts, linewidth=2, edgecolor='lime', facecolor='none')
        ax2.add_patch(poly_o)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        color_status = 'lime' if (mismatch_ra < 1.0 and mismatch_dec < 1.0) else 'red'
        txt_coords = (f"📍 ALIGNMENT CHECK (Pair {save_path.stem})\n"
                      f"-----------------------------------\n"
                      f"HUBBLE RA Center: {h_ra:.6f}°\n"
                      f"LR RA Center:     {lr_ra:.6f}°\n"
                      f"MISMATCH:         {mismatch_ra:.3f}\" / {mismatch_dec:.3f}\"\n"
                      f"-----------------------------------\n"
                      f"STATUS: ")
        ax3.text(0.05, 0.6, txt_coords, fontsize=12, color='white', verticalalignment='center', family='monospace')
        ax3.text(0.3, 0.47, "PERFECT" if color_status=='lime' else "MISMATCH", fontsize=14, color=color_status, fontweight='bold', family='monospace')

        ax4 = fig.add_subplot(gs[1, 0])
        tar_n = get_robust_preview(patch_h)
        ax4.imshow(tar_n, origin='lower', cmap='inferno')
        ax4.set_title("Hubble Patch (HR)", color='white')
        ax4.axis('off')

        ax5 = fig.add_subplot(gs[1, 1])
        inp_s = get_robust_preview(patch_o_lr)
        ax5.imshow(inp_s, origin='lower', cmap='viridis')
        ax5.set_title(f"Obs Patch (LR)", color='white')
        ax5.axis('off')

        inp_resized = resize(inp_s, (HR_SIZE, HR_SIZE), order=0, anti_aliasing=False)
        rgb = np.zeros((HR_SIZE, HR_SIZE, 3))
        rgb[..., 0] = tar_n
        rgb[..., 1] = inp_resized
        
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.imshow(rgb, origin='lower')
        ax6.set_title(f"Overlay (R=HST, G=OBS)", color='white')
        ax6.text(0.02, 0.02, f"Err: {max(mismatch_ra, mismatch_dec):.2f}\"", transform=ax6.transAxes, color=color_status, fontsize=10, fontweight='bold')
        ax6.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, facecolor='#1e1e1e')
        plt.close(fig)
        
    except Exception as e:
        print(f"\n❌ ERRORE PNG: {e}")

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

def create_aligned_lr_wcs(hr_patch_wcs, hr_size, lr_size):
    factor = hr_size / lr_size
    
    w_lr = hr_patch_wcs.deepcopy()
    
    if w_lr.wcs.has_cd():
        w_lr.wcs.cd *= factor
    else:
        w_lr.wcs.cdelt *= factor
        
    w_lr.wcs.crpix /= factor
    
    return w_lr

def process_single_patch_multi(args):
    global patch_index_counter
    h_path, y, x = args
    
    data_h = shared_data['h']
    wcs_h = shared_data['wcs_h']

    patch_h = data_h[y:y+HR_SIZE, x:x+HR_SIZE]
    
    if np.count_nonzero(patch_h > MIN_PIXEL_VALUE) / patch_h.size < MIN_COVERAGE:
        return 0

    patch_h_wcs = wcs_h[y:y+HR_SIZE, x:x+HR_SIZE]

    lr_target_wcs = create_aligned_lr_wcs(patch_h_wcs, HR_SIZE, AI_LR_SIZE)
    
    saved_count = 0
    
    for o_path in shared_data['o_files']:
        try:
            with fits.open(o_path) as o:
                data_o = np.nan_to_num(o[0].data)
                if data_o.ndim > 2: data_o = data_o[0]
                wcs_o = WCS(o[0].header)

            patch_o_lr, footprint = reproject_interp(
                (data_o, wcs_o),
                lr_target_wcs,
                shape_out=(AI_LR_SIZE, AI_LR_SIZE),
                order='bilinear'
            )
            patch_o_lr = np.nan_to_num(patch_o_lr)

            valid_mask = (patch_o_lr > MIN_PIXEL_VALUE)
            if np.sum(valid_mask) < (AI_LR_SIZE**2 * MIN_COVERAGE):
                continue

            with log_lock:
                idx = patch_index_counter
                patch_index_counter += 1
            
            pair_dir = shared_data['out_fits'] / f"pair_{idx:06d}"
            pair_dir.mkdir(exist_ok=True)
            
            fits.PrimaryHDU(patch_h.astype(np.float32), header=patch_h_wcs.to_header()).writeto(pair_dir/"hubble.fits", overwrite=True)
            fits.PrimaryHDU(patch_o_lr.astype(np.float32), header=lr_target_wcs.to_header()).writeto(pair_dir/"observatory.fits", overwrite=True)
            
            saved_count += 1
            
            if idx < DEBUG_SAMPLES:
                png_path = shared_data['out_png'] / f"check_pair_{idx:06d}.jpg"
                save_diagnostic_card(
                    data_h, data_o,
                    patch_h, patch_o_lr,
                    x, y, wcs_h, wcs_o,
                    lr_target_wcs, 
                    shared_data['h_fov_deg'], png_path
                )

        except Exception:
            continue
            
    return saved_count

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
    print(f"🚀 ESTRAZIONE PATCH WCS-AWARE")
    print(f"   Config: HR={HR_SIZE}px, LR={AI_LR_SIZE}px")
    
    target_dir = ROOT_DATA_DIR / "M1" 
    if len(sys.argv) > 1: 
        target_dir = Path(sys.argv[1])
    else:
        sel = select_target_directory()
        if sel: target_dir = sel
    
    print(f"\n📂 Target: {target_dir.name}")
    
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
    print(f"   🔍 Filtraggio file non allineati...")
    for f in o_files_all:
        try:
            with fits.open(f) as o:
                w = WCS(o[0].header)
                dist = np.sqrt((w.wcs.crval[0]-h_center[0])**2 + (w.wcs.crval[1]-h_center[1])**2)
                if dist < 0.1: 
                    o_files_good.append(f)
        except: pass
        
    print(f"   ✅ File validi: {len(o_files_good)}")
    
    if not o_files_good:
        print("❌ Nessun file osservatorio centrato su Hubble.")
        return

    h_h, h_w = d_h.shape
    tasks = []
    for y in range(0, h_h - HR_SIZE + 1, STRIDE):
        for x in range(0, h_w - HR_SIZE + 1, STRIDE):
            tasks.append((h_master_path, y, x))
            
    print(f"   📦 Patch da processare: {len(tasks)}")
    
    print(f"\n🚀 Avvio estrazione...")
    total_saved = 0
    
    with ProcessPoolExecutor(initializer=init_worker,
                             initargs=(d_h, h_head, w_h, out_fits, out_png, h_fov_deg, o_files_good)) as ex:
        
        results = list(tqdm(ex.map(process_single_patch_multi, tasks), total=len(tasks), ncols=100))
        total_saved = sum(results)
        
    print(f"\n✅ COMPLETATO.")
    print(f"   Coppie salvate: {total_saved}")
    print(f"   Dataset: {out_fits}")
    print(f"   Debug: {out_png}")

    target_name = target_dir.name
    
    zip_fits_name = target_dir / f"{target_name}_patches"
    print(f"   🗜️  ZIP Dataset: {zip_fits_name}.zip")
    shutil.make_archive(str(zip_fits_name), 'zip', str(out_fits))
    
    zip_png_name = target_dir / f"{target_name}_debug_visuals"
    print(f"   🗜️  ZIP Debug: {zip_png_name}.zip")
    shutil.make_archive(str(zip_png_name), 'zip', str(out_png))

if __name__ == "__main__":
    main()