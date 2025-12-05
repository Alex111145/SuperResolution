import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval
from reproject import reproject_interp
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ================= CONFIGURAZIONE =================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

def select_target_directory():
    print("\n" + "="*35)
    print("CONTROLLO ALLINEAMENTO (MODALITÀ MOSAICO)".center(70))
    print("="*35)
    try:
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs']]
    except: return None
    
    print("\nTarget disponibili:")
    for i, d in enumerate(subdirs): print(f" {i+1}: {d.name}")
    try:
        idx = int(input("Scelta: ")) - 1
        return subdirs[idx] if 0 <= idx < len(subdirs) else None
    except: return None

def get_image_data(hdul):
    """Estrae i dati validi (2D) dall'HDU list."""
    for hdu in hdul:
        if hdu.data is not None and hdu.data.ndim >= 2:
            return hdu.data, hdu.header
    return hdul[0].data, hdul[0].header

def load_observatory_master(folder_path):
    """Carica l'immagine dell'osservatorio per usarla come 'Tela Master'."""
    files = sorted(list(folder_path.glob('*.fits')) + list(folder_path.glob('*.fit')))
    if not files: return None, None, None
    
    # Usiamo la prima immagine (o uno stack veloce) come riferimento geometrico
    print(f"Caricamento Riferimento Osservatorio ({len(files)} files)...")
    
    stack_data = []
    ref_wcs = None
    ref_shape = None
    
    # Carichiamo max 10 immagini per fare la media veloce del fondo
    for f in tqdm(files[:10], desc="   Stacking Obs"):
        with fits.open(f) as h:
            d, head = get_image_data(h)
            if ref_wcs is None:
                ref_wcs = WCS(head)
                ref_shape = d.shape
            
            # Resize se necessario (non dovrebbe servire se registrati, ma per sicurezza)
            if d.shape == ref_shape:
                if d.ndim == 3: d = np.mean(d, axis=0)
                stack_data.append(d)

    master_data = np.nanmedian(np.array(stack_data), axis=0)
    master_data = np.nan_to_num(master_data, nan=0.0)
    return master_data, ref_wcs

def create_hubble_mosaic(folder_path, target_wcs, target_shape):
    """
    Invece di stackare, PROIETTA ogni pezzo di Hubble sulla griglia dell'Osservatorio.
    """
    files = sorted(list(folder_path.glob('*.fits')) + list(folder_path.glob('*.fit')))
    if not files: return None

    print(f"Creazione Mosaico Hubble da {len(files)} tasselli...")
    
    # Creiamo una tela vuota con le dimensioni dell'Osservatorio
    mosaic_canvas = np.zeros(target_shape, dtype=np.float32)
    
    # Per ogni file Hubble
    for f in tqdm(files, desc="   Stitching Hubble"):
        try:
            with fits.open(f) as hdul:
                data, head = get_image_data(hdul)
                wcs = WCS(head)
                
                if data is None or wcs is None: continue
                if data.ndim == 3: data = data[0] # Prendi solo un canale se RGB
                
                # Riproietta questo tassello sulla tela globale
                # Usiamo 'reproject_interp' per velocità
                reproj_data, footprint = reproject_interp(
                    (data, wcs), 
                    target_wcs, 
                    shape_out=target_shape
                )
                
                # Sovrapponiamo alla tela (usiamo np.nanmax per 'vincere' sui pixel neri)
                reproj_data = np.nan_to_num(reproj_data, nan=0.0)
                mosaic_canvas = np.maximum(mosaic_canvas, reproj_data)
                
        except Exception as e:
            # print(f"Skipped {f.name}: {e}")
            pass
            
    return mosaic_canvas

def normalize_zscale(data):
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(data)
    norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    return norm

def main():
    target_dir = select_target_directory()
    if not target_dir: return

    # Nota: Usiamo i file SOLVED (Step 1) o REGISTERED (Step 2)?
    # Per M42 Mosaic, è meglio usare i file '2_solved_astap' di Hubble 
    # e proiettarli sul '3_registered_native' dell'Osservatorio.
    # Ma per coerenza col tuo flusso, usiamo le cartelle standard.
    
    dir_hubble = target_dir / '2_solved_astap/hubble' # Usiamo i solved originali per non avere crop
    if not dir_hubble.exists() or not list(dir_hubble.glob('*.fits')):
        # Fallback se non esistono
        dir_hubble = target_dir / '3_registered_native/hubble'

    dir_observatory = target_dir / '3_registered_native/observatory'
    
    out_dir = target_dir / '4_quality_check'
    out_dir.mkdir(exist_ok=True)

    # 1. Carica Master Osservatorio (La Tela)
    obs_data, obs_wcs = load_observatory_master(dir_observatory)
    if obs_data is None:
        print("Dati Osservatorio mancanti.")
        return

    # 2. Crea Mosaico Hubble sulla tela dell'Osservatorio
    hubble_mosaic = create_hubble_mosaic(dir_hubble, obs_wcs, obs_data.shape)
    if hubble_mosaic is None or np.max(hubble_mosaic) == 0:
        print("Impossibile creare mosaico Hubble (dati vuoti o WCS errati).")
        return

    # 3. Normalizzazione
    print("Generazione Overlay...")
    img_o = normalize_zscale(obs_data)
    img_h = normalize_zscale(hubble_mosaic)

    # 4. Creazione Overlay
    # Verde = Hubble, Magenta = Osservatorio
    rgb_overlay = np.zeros((img_o.shape[0], img_o.shape[1], 3), dtype=np.float32)
    rgb_overlay[..., 1] = img_h * 0.8       # Green
    rgb_overlay[..., 0] = img_o * 0.5       # Red
    rgb_overlay[..., 2] = img_o * 0.5       # Blue

    # 5. Plotting
    fig = plt.figure(figsize=(18, 6), facecolor='black')

    # Hubble Mosaic
    ax1 = plt.subplot(1, 3, 1, projection=obs_wcs)
    ax1.imshow(img_h, cmap='gray', origin='lower', vmin=0, vmax=1)
    ax1.set_title("Hubble Mosaic (Projected)", color='white')
    ax1.coords.grid(color='white', alpha=0.2)
    ax1.set_facecolor('black')

    # Observatory
    ax2 = plt.subplot(1, 3, 2, projection=obs_wcs)
    ax2.imshow(img_o, cmap='magma', origin='lower', vmin=0, vmax=1)
    ax2.set_title("Observatory Master", color='white')
    ax2.coords.grid(color='white', alpha=0.2)
    ax2.set_facecolor('black')

    # Overlay
    ax3 = plt.subplot(1, 3, 3, projection=obs_wcs)
    ax3.imshow(rgb_overlay, origin='lower')
    ax3.set_title(f"Mosaic Check ({target_dir.name})", color='white', fontweight='bold')
    ax3.set_xlabel("V=Hubble | M=Obs", color='white')
    ax3.coords.grid(color='white', alpha=0.2)
    ax3.set_facecolor('black')

    out_file = out_dir / f"{target_dir.name}_mosaic_check.png"
    plt.savefig(out_file, dpi=150, bbox_inches='tight', facecolor='black')
    print(f"\n Controllo completato! Mosaico generato:\n   {out_file}")

if __name__ == "__main__":
    main()