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

# --- CONFIGURAZIONE GLOBALE ---
warnings.filterwarnings('ignore')

# Definizioni Path (Assicurati che l'ambiente sia configurato correttamente)
# La linea successiva presuppone che lo script venga eseguito da un contesto specifico, 
# se non è così, potrebbe essere necessario modificarla.
try:
    CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # Caso in cui lo script viene eseguito in un ambiente interattivo (es. Jupyter)
    CURRENT_SCRIPT_DIR = Path.cwd() 
    
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# Parametri Dataset
HR_SIZE = 512           # Dimensione patch Alta Risoluzione (Hubble)
AI_LR_SIZE = 128        # Dimensione patch Bassa Risoluzione (Target per Riproiezione)
STRIDE = 50             # Passo per l'estrazione delle patch (sovrapposizione)
MIN_COVERAGE = 0.50     # Copertura minima richiesta (0.0 a 1.0)
MIN_PIXEL_VALUE = 0.0001 # Valore minimo per considerare un pixel "valido"
DEBUG_SAMPLES = 50      # Numero di schede diagnostiche da salvare

# Globali per il multiprocessing
log_lock = threading.Lock()
shared_data = {}
patch_index_counter = 0

# --- FUNZIONI DI UTILITÀ ASTRONOMICA E DIAGNOSTICA ---

def get_pixel_scale_deg(wcs):
    """Calcola la scala media dei pixel in gradi."""
    scales = proj_plane_pixel_scales(wcs)
    return np.mean(scales)

def get_robust_preview(data, size=None):
    """Normalizza e opzionalmente ridimensiona i dati per la visualizzazione PNG."""
    try:
        data = np.nan_to_num(data)
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(data)
        # Normalizzazione Min-Max dopo ZScale e clipping
        clipped = np.clip((data - vmin) / (vmax - vmin), 0, 1)
        
        if size:
            return resize(clipped, (size, size), anti_aliasing=True)
        return clipped
    except:
        return np.zeros_like(data)

def calculate_wcs_corners(wcs, size):
    """Calcola le coordinate RA/Dec centrali di un'immagine data WCS e dimensione."""
    # Uso (size-1)/2 per centrare correttamente sui pixel (0-based indexing)
    center_world = wcs.pixel_to_world((size-1)/2, (size-1)/2) 
    return center_world.ra.deg, center_world.dec.deg

def save_diagnostic_card(data_h_orig, data_o_raw_orig, 
                         patch_h, patch_o_lr, 
                         x, y, wcs_h, wcs_o_raw,
                         lr_wcs_target, 
                         h_fov_deg, save_path):
    """Genera e salva una scheda diagnostica PNG per la verifica dell'allineamento."""
    
    try:
        fig = plt.figure(figsize=(20, 12), facecolor='#1e1e1e') 
        gs = fig.add_gridspec(2, 3)

        # WCS della patch HR ritagliata
        h_patch_wcs = wcs_h.deepcopy()
        # Modifica CRPIX per centrare il WCS sull'angolo (x, y) del ritaglio
        h_patch_wcs.wcs.crpix -= np.array([x, y]) 
        
        # Calcolo Centri
        h_ra, h_dec = calculate_wcs_corners(h_patch_wcs, HR_SIZE)
        lr_ra, lr_dec = calculate_wcs_corners(lr_wcs_target, AI_LR_SIZE)
        
        # Mismatch in secondi d'arco (3600"/grado)
        mismatch_ra = abs(h_ra - lr_ra) * 3600
        mismatch_dec = abs(h_dec - lr_dec) * 3600

        # Calcolo Vertici Patch LR in coordinate Osservatorio RAW (per Plot 2)
        lr_corners_pix = np.array([
            [0, 0], [AI_LR_SIZE, 0], [AI_LR_SIZE, AI_LR_SIZE], [0, AI_LR_SIZE], [0, 0] # Chiusura del poligono
        ])
        lr_corners_world = lr_wcs_target.pixel_to_world(lr_corners_pix[:, 0], lr_corners_pix[:, 1])
        obs_corners_pix_raw = wcs_o_raw.world_to_pixel(lr_corners_world)
        
        # Ridimensionamento per la vista globale (dimensione fissa 512x512)
        scale_ox = 512 / data_o_raw_orig.shape[1]
        scale_oy = 512 / data_o_raw_orig.shape[0]
        polygon_verts = np.stack([obs_corners_pix_raw[0] * scale_ox, obs_corners_pix_raw[1] * scale_oy], axis=1)

        # Plot 1: Mappa Hubble con riquadro patch
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        h_small = get_robust_preview(data_h_orig, 512)
        ax1.imshow(h_small, origin='lower', cmap='inferno')
        ax1.set_title("GLOBAL HUBBLE MAP (HR Source)", color='cyan', fontweight='bold')
        
        scale_hy = 512 / data_h_orig.shape[0]
        scale_hx = 512 / data_h_orig.shape[1]
        rect_h = patches.Rectangle((x*scale_hx, y*scale_hy), HR_SIZE*scale_hx, HR_SIZE*scale_hy, 
                                 linewidth=2, edgecolor='cyan', facecolor='none')
        ax1.add_patch(rect_h)

        # Plot 2: Mappa Osservatorio con riquadro patch riproiettata
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        o_small = get_robust_preview(data_o_raw_orig, 512)
        ax2.imshow(o_small, origin='lower', cmap='viridis')
        ax2.set_title("GLOBAL OBS MAP (LR Source)", color='lime', fontweight='bold')
        poly_o = patches.Polygon(polygon_verts, linewidth=2, edgecolor='lime', facecolor='none')
        ax2.add_patch(poly_o)

        # Plot 3: Allineamento e Metadati
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        # Allineamento considerato buono se l'errore massimo è inferiore a 1 secondo d'arco
        color_status = 'lime' if (mismatch_ra < 1.0 and mismatch_dec < 1.0) else 'red'
        txt_coords = (f"📍 ALIGNMENT CHECK (Pair {save_path.stem})\n"
                      f"-----------------------------------\n"
                      f"HUBBLE Center (RA/Dec): {h_ra:.6f}° / {h_dec:.6f}°\n"
                      f"LR Center (RA/Dec):     {lr_ra:.6f}° / {lr_dec:.6f}°\n"
                      f"MISMATCH (RA/Dec):      {mismatch_ra:.3f}\" / {mismatch_dec:.3f}\"\n"
                      f"-----------------------------------\n"
                      f"STATUS: ")
        ax3.text(0.05, 0.6, txt_coords, fontsize=12, color='white', verticalalignment='center', family='monospace')
        ax3.text(0.3, 0.4, "PERFECT" if color_status=='lime' else "MISMATCH", fontsize=14, color=color_status, fontweight='bold', family='monospace')

        # Plot 4: Patch HR (Hubble)
        ax4 = fig.add_subplot(gs[1, 0])
        tar_n = get_robust_preview(patch_h)
        ax4.imshow(tar_n, origin='lower', cmap='inferno')
        ax4.set_title(f"Hubble Patch (HR {HR_SIZE}x{HR_SIZE})", color='white')
        ax4.axis('off')

        # Plot 5: Patch LR (Osservatorio Riproiettato)
        ax5 = fig.add_subplot(gs[1, 1])
        inp_s = get_robust_preview(patch_o_lr)
        ax5.imshow(inp_s, origin='lower', cmap='viridis')
        ax5.set_title(f"Obs Patch (LR {AI_LR_SIZE}x{AI_LR_SIZE})", color='white')
        ax5.axis('off')

        # Plot 6: Overlay (RGB - Allineamento Visivo)
        # Interpolazione semplice (nearest) dell'LR a dimensione HR per l'overlay
        inp_resized = resize(inp_s, (HR_SIZE, HR_SIZE), order=0, anti_aliasing=False)
        rgb = np.zeros((HR_SIZE, HR_SIZE, 3))
        rgb[..., 0] = tar_n # Rosso = Hubble (HR)
        rgb[..., 1] = inp_resized # Verde = Osservatorio (LR)
        
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.imshow(rgb, origin='lower')
        ax6.set_title(f"Overlay (R=HST, G=OBS)", color='white')
        ax6.text(0.02, 0.02, f"Max Err: {max(mismatch_ra, mismatch_dec):.2f}\"", 
                 transform=ax6.transAxes, color=color_status, fontsize=10, fontweight='bold')
        ax6.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, facecolor='#1e1e1e')
        plt.close(fig)
        
    except Exception as e:
        # Errore nella creazione del PNG, non blocca il flusso principale
        print(f"\n❌ ERRORE PNG per patch ({x},{y}): {e}")

# --- LOGICA MULTIPROCESSING ---

def init_worker(d_h, hdr_h, w_h, out_fits, out_png, h_fov_deg, o_files):
    """Funzione di inizializzazione per i processi worker."""
    global patch_index_counter
    shared_data['h'] = d_h
    shared_data['header_h'] = hdr_h
    shared_data['wcs_h'] = w_h
    shared_data['out_fits'] = out_fits
    shared_data['out_png'] = out_png
    shared_data['h_fov_deg'] = h_fov_deg
    shared_data['o_files'] = o_files
    # Reimposta il contatore all'inizio di ogni target
    patch_index_counter = 0 

def create_aligned_lr_wcs(hr_patch_wcs, hr_size, lr_size):
    """Crea un oggetto WCS per la patch LR allineato alla patch HR."""
    # Il fattore S è la scala di risoluzione spaziale
    factor = hr_size / lr_size
    
    w_lr = hr_patch_wcs.deepcopy()
    
    # Adatta i parametri WCS per la nuova dimensione di output LR
    if w_lr.wcs.has_cd():
        # Aggiorna la matrice di trasformazione CD
        w_lr.wcs.cd *= factor
    else:
        # Aggiorna la scala del pixel
        w_lr.wcs.cdelt *= factor
        
    # Il punto di riferimento CRPIX viene scalato per riflettere il nuovo numero di pixel
    w_lr.wcs.crpix /= factor
    
    return w_lr

def process_single_patch_multi(args):
    """Elabora una singola patch e riproietta i dati dell'osservatorio."""
    global patch_index_counter
    h_path, y, x = args
    
    data_h = shared_data['h']
    wcs_h = shared_data['wcs_h']

    patch_h = data_h[y:y+HR_SIZE, x:x+HR_SIZE]
    
    # 1. Controllo copertura HR (Hubble)
    if np.count_nonzero(patch_h > MIN_PIXEL_VALUE) / patch_h.size < MIN_COVERAGE:
        return 0

    # Ricava la WCS per la patch HR
    # Nota: Astropy WCS slicing supporta solo il slicing 1D (per CRPIX), 
    # ma il slicing dell'array qui è solo per riferimento, la WCS corretta
    # viene calcolata in save_diagnostic_card (sottraendo [x,y] da CRPIX)
    patch_h_wcs = wcs_h[y:y+HR_SIZE, x:x+HR_SIZE] 

    # 2. Creazione WCS Target LR Allineato
    lr_target_wcs = create_aligned_lr_wcs(patch_h_wcs, HR_SIZE, AI_LR_SIZE)
    
    saved_count = 0
    
    # 3. Elaborazione di tutti i file Osservatorio disponibili
    for o_path in shared_data['o_files']:
        try:
            with fits.open(o_path) as o:
                data_o = np.nan_to_num(o[0].data)
                if data_o.ndim > 2: data_o = data_o[0]
                wcs_o = WCS(o[0].header)

            # Riproiezione (reproject_interp): Mappa data_o/wcs_o su lr_target_wcs
            patch_o_lr, footprint = reproject_interp(
                (data_o, wcs_o),
                lr_target_wcs,
                shape_out=(AI_LR_SIZE, AI_LR_SIZE),
                order='bilinear'
            )
            patch_o_lr = np.nan_to_num(patch_o_lr)

            # 4. Controllo copertura LR riproiettata
            valid_mask = (patch_o_lr > MIN_PIXEL_VALUE)
            if np.sum(valid_mask) / patch_o_lr.size < MIN_COVERAGE:
                continue

            # Incremento Indice (thread safe)
            with log_lock:
                idx = patch_index_counter
                patch_index_counter += 1
            
            pair_dir = shared_data['out_fits'] / f"pair_{idx:06d}"
            pair_dir.mkdir(exist_ok=False) # Deve essere una nuova directory
            
            # 5. Salvataggio FITS (HR e LR)
            # Salvataggio HR con la WCS corretta (basata su x,y)
            fits.PrimaryHDU(patch_h.astype(np.float32), header=patch_h_wcs.to_header()).writeto(pair_dir/"hubble.fits", overwrite=True)
            # Salvataggio LR con la WCS target allineata
            fits.PrimaryHDU(patch_o_lr.astype(np.float32), header=lr_target_wcs.to_header()).writeto(pair_dir/"observatory.fits", overwrite=True)
            
            saved_count += 1
            
            # 6. Salvataggio diagnostico (solo i primi DEBUG_SAMPLES)
            if idx < DEBUG_SAMPLES:
                png_path = shared_data['out_png'] / f"check_pair_{idx:06d}.jpg"
                save_diagnostic_card(
                    data_h, data_o, # Dati Originali (per global map)
                    patch_h, patch_o_lr, # Dati Patch
                    x, y, wcs_h, wcs_o, # WCS Originali
                    lr_target_wcs, # WCS Target (per allineamento)
                    shared_data['h_fov_deg'], png_path
                )

        except Exception as e:
            # Errore specifico durante il processing di una singola patch
            # Questo può includere errori di WCS o di I/O.
            # print(f"Errore su patch ({x},{y}) con {o_path.name}: {e}")
            continue
            
    return saved_count

# --- LOGICA DI ELABORAZIONE SINGOLO TARGET ---

def process_single_target(target_dir):
    """Logica principale per l'estrazione delle patch su una singola directory."""
    input_h = target_dir / '3_registered_native' / 'hubble'
    input_o = target_dir / '3_registered_native' / 'observatory'
    
    out_fits = target_dir / '6_patches_final'
    out_png = target_dir / '6_debug_visuals' 
    
    # Pulizia e creazione directory di output
    if out_fits.exists(): shutil.rmtree(out_fits)
    out_fits.mkdir(parents=True)
    if out_png.exists(): shutil.rmtree(out_png)
    out_png.mkdir(parents=True)
    
    h_files = sorted(list(input_h.glob("*.fits")))
    o_files_all = sorted(list(input_o.glob("*.fits")))
    
    if not h_files or not o_files_all:
        print("❌ File mancanti in 3_registered_native. Salto.")
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
        print(f"❌ Errore lettura Hubble Master file: {e}. Salto.")
        return

    # Filtro file Osservatorio per centralità
    o_files_good = []
    print(f" 	 🔍 Filtraggio file Osservatorio non allineati...")
    for f in o_files_all:
        try:
            with fits.open(f) as o:
                w = WCS(o[0].header)
                # Calcola la distanza angolare tra i centri WCS in gradi
                dist = np.sqrt((w.wcs.crval[0]-h_center[0])**2 + (w.wcs.crval[1]-h_center[1])**2)
                # Assunzione: l'osservatorio deve essere sufficientemente centrato (entro 0.1 gradi)
                if dist < 0.1: 
                    o_files_good.append(f)
        except: 
            # Ignora file FITS corrotti
            continue 
        
    print(f" 	 ✅ File Osservatorio validi: {len(o_files_good)}")
    
    if not o_files_good:
        print("❌ Nessun file osservatorio centrato su Hubble. Salto.")
        return

    # Definizione delle task (coordinate Y, X per le patch HR)
    h_h, h_w = d_h.shape
    tasks = []
    for y in range(0, h_h - HR_SIZE + 1, STRIDE):
        for x in range(0, h_w - HR_SIZE + 1, STRIDE):
            tasks.append((h_master_path, y, x))
            
    print(f" 	 📦 Patch HR potenziali da analizzare: {len(tasks)}")
    
    print(f"\n🚀 Avvio estrazione multi-processo...")
    total_saved = 0
    
    # Esecuzione in parallelo
    max_workers = os.cpu_count() or 4 # Usa tutti i core disponibili
    with ProcessPoolExecutor(max_workers=max_workers,
                             initializer=init_worker,
                             initargs=(d_h, h_head, w_h, out_fits, out_png, h_fov_deg, o_files_good)) as ex:
        
        # Mappa le task ai worker e usa tqdm per mostrare l'avanzamento sulla linea di comando
        results = list(tqdm(ex.map(process_single_patch_multi, tasks), total=len(tasks), ncols=100,
                            desc=f"Patching {target_dir.name}"))
        total_saved = sum(results)
        
    print(f"\n✅ COMPLETATO TARGET {target_dir.name}.")
    print(f" 	 Coppie salvate (Hubble + Osservatorio): {total_saved}")
    
    # Compressione finale
    target_name = target_dir.name
    zip_fits_name = target_dir / f"{target_name}_patches"
    shutil.make_archive(str(zip_fits_name), 'zip', str(out_fits))
    print(f" 	 🗜️ 	ZIP Dataset: {zip_fits_name}.zip")
    
    zip_png_name = target_dir / f"{target_name}_debug_visuals"
    shutil.make_archive(str(zip_png_name), 'zip', str(out_png))
    print(f" 	 🗜️ 	ZIP Debug: {zip_png_name}.zip")


# --- MENU SELEZIONE AGGIORNATO (Multi-Target) ---

def select_target_directories(required_subdir='3_registered_native'):
    """
    Consente la selezione di una, più o tutte le directory target che contengono la sottocartella richiesta.
    """
    # Filtra solo le directory che non sono speciali (es. 'splits', 'logs')
    all_subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs']]
    
    # Filtra solo le directory che contengono la sottocartella richiesta
    valid_subdirs = [
        d for d in all_subdirs 
        if (d / required_subdir).exists() and (d / required_subdir).is_dir()
    ]

    if not valid_subdirs: 
        print(f"\n❌ Nessuna directory valida trovata in {ROOT_DATA_DIR}")
        print(f"(Nessuna cartella contiene '{required_subdir}')")
        return []

    print(f"\nSELEZIONA TARGET (Solo quelle con '{required_subdir}'):")
    valid_subdirs.sort(key=lambda x: x.name)
    for i, d in enumerate(valid_subdirs): 
        print(f" {i+1}: {d.name}")
        
    print(f" A: TUTTE ({len(valid_subdirs)} in totale)")
    print(" (Es. '1,3,4' per selezione multipla, o 'A' per tutte)")
        
    try:
        raw_val = input("Scelta: ").strip().upper()
        if not raw_val: return []

        if raw_val == 'A':
            return valid_subdirs
        
        selected_indices = []
        for part in raw_val.split(','):
            try:
                idx = int(part) - 1
                if 0 <= idx < len(valid_subdirs):
                    selected_indices.append(idx)
            except ValueError:
                continue

        # Rimuove duplicati e mantiene l'ordine di selezione/ordinamento
        unique_indices = []
        for idx in selected_indices:
            if idx not in unique_indices:
                unique_indices.append(idx)
                
        return [valid_subdirs[i] for i in unique_indices]
        
    except Exception: 
        return []

# --- MAIN CONTROLLER (Modificato per Avanzamento) ---

def main():
    """Funzione di controllo che gestisce l'input e itera sui target."""
    print(f"🚀 ESTRAZIONE PATCH WCS-AWARE")
    print(f" 	 Config: HR={HR_SIZE}px, LR={AI_LR_SIZE}px")
    
    # Gestione argomento da riga di comando (priorità) o menu interattivo
    if len(sys.argv) > 1: 
        # Tenta di prendere il primo argomento come path
        target_dirs = [Path(sys.argv[1])]
    else:
        target_dirs = select_target_directories('3_registered_native')
        
    if not target_dirs:
        print("❌ Nessun target selezionato. Esco.")
        return

    total_targets = len(target_dirs) # Conteggio totale
    
    for i, target_dir in enumerate(target_dirs): # Iterazione con indice
        current_index = i + 1 
        progress_label = f"({current_index}/{total_targets})" # Esempio: (1/4)
        
        print(f"\n=======================================================")
        print(f"📂 INIZIO LAVORAZIONE TARGET {progress_label}: {target_dir.name}")
        print(f"=======================================================")
        
        try:
            process_single_target(target_dir)
        except Exception as e:
            print(f"⚠️ Errore critico nel processo di {target_dir.name}: {e}")
            
    print("\n\n🎉 TUTTE LE ESTRAZIONI COMPLETATE.")

if __name__ == "__main__":
    main()