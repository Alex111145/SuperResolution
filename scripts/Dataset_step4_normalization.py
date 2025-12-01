import os
import sys
import shutil
import numpy as np
from astropy.io import fits
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ================= CONFIGURAZIONE =================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# Impostazioni Normalizzazione
# Tagliamo l'1% più scuro (fondo cielo sporco) e lo 0.1% più luminoso (stelle bruciate)
LOWER_PERCENTILE = 1.0 
UPPER_PERCENTILE = 99.9 

def select_target_directory():
    print("\n" + "⚖️"*35)
    print("NORMALIZZAZIONE DATASET (FITS -> TIFF 16-bit)".center(70))
    print("⚖️"*35)
    
    subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs']]
    if not subdirs: return None
    
    print("\nTarget disponibili (con cartella '6_patches_final'):")
    valid_targets = []
    for i, d in enumerate(subdirs):
        if (d / '6_patches_final').exists():
            print(f" {len(valid_targets)+1}: {d.name}")
            valid_targets.append(d)
            
    if not valid_targets:
        print("❌ Nessun target ha le patch estratte. Esegui lo Step 3 prima.")
        return None

    try:
        idx = int(input("Scelta: ")) - 1
        return valid_targets[idx] if 0 <= idx < len(valid_targets) else None
    except: return None

def normalize_to_uint16(data):
    """
    Normalizza un array float (qualsiasi range) in uint16 (0-65535).
    Usa clipping basato sui percentili per massimizzare il contrasto utile.
    """
    # 1. Gestione NaN e Infinti
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 2. Calcolo range dinamico utile
    vmin = np.percentile(data, LOWER_PERCENTILE)
    vmax = np.percentile(data, UPPER_PERCENTILE)
    
    # Protezione da immagini piatte/nere
    if vmax <= vmin:
        return np.zeros(data.shape, dtype=np.uint16)
    
    # 3. Normalizzazione 0.0 - 1.0 con clipping
    # I valori sotto vmin diventano 0, sopra vmax diventano 1
    norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    
    # 4. Espansione a 16-bit
    return (norm * 65535).astype(np.uint16)

def main():
    target_dir = select_target_directory()
    if not target_dir: return

    input_dir = target_dir / '6_patches_final'
    output_dir = target_dir / '7_dataset_ready'

    # Pulizia e creazione cartella output principale
    if output_dir.exists(): shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    pairs = sorted(list(input_dir.glob("pair_*")))
    print(f"\n🚀 Avvio normalizzazione su {len(pairs)} coppie...")
    print(f"   Input:  {input_dir}")
    print(f"   Output: {output_dir} (Struttura a coppie)")

    count = 0
    
    # Loop principale
    for pair_folder in tqdm(pairs, ncols=100):
        try:
            p_id = pair_folder.name # es. pair_000001
            
            f_hubble = pair_folder / "hubble.fits"
            f_obs = pair_folder / "observatory.fits"
            
            if not f_hubble.exists() or not f_obs.exists(): continue

            # 1. Caricamento FITS
            with fits.open(f_hubble) as h: d_hr = h[0].data
            with fits.open(f_obs) as o:    d_lr = o[0].data
            
            # --- STATS PRE-NORMALIZZAZIONE ---
            raw_h_min, raw_h_max = np.nanmin(d_hr), np.nanmax(d_hr)
            raw_o_min, raw_o_max = np.nanmin(d_lr), np.nanmax(d_lr)

            # 2. Normalizzazione Indipendente
            img_hr_u16 = normalize_to_uint16(d_hr)
            img_lr_u16 = normalize_to_uint16(d_lr)
            
            # --- STATS POST-NORMALIZZAZIONE ---
            norm_h_min, norm_h_max = img_hr_u16.min(), img_hr_u16.max()
            norm_o_min, norm_o_max = img_lr_u16.min(), img_lr_u16.max()

            # --- LOGGING (Solo primi 5 campioni per evitare spam) ---
            if count < 5:
                tqdm.write(f"\n🔎 [Check {p_id}]")
                tqdm.write(f"   Hubble RAW: {raw_h_min:.4e} -> {raw_h_max:.4e}  |  TIFF16: {norm_h_min} -> {norm_h_max}")
                tqdm.write(f"   Obs    RAW: {raw_o_min:.4e} -> {raw_o_max:.4e}  |  TIFF16: {norm_o_min} -> {norm_o_max}")
                tqdm.write("-" * 60)

            # 3. Creazione Cartella Coppia e Salvataggio
            # Creiamo la cartella output_dir/pair_XXXXXX
            current_pair_out = output_dir / p_id
            current_pair_out.mkdir(parents=True, exist_ok=True)

            # Salviamo i file dentro la cartella specifica della coppia
            Image.fromarray(img_hr_u16, mode='I;16').save(current_pair_out / "hubble.tiff")
            Image.fromarray(img_lr_u16, mode='I;16').save(current_pair_out / "observatory.tiff")
            
            count += 1
            
        except Exception as e:
            tqdm.write(f"⚠️ Errore su {pair_folder.name}: {e}")

    print(f"\n✅ DATASET COMPLETATO!")
    print(f"   Coppie salvate: {count}")
    print(f"   Cartella dati: {output_dir}")

    # ================= ZIPPING SECTION =================
    target_name = target_dir.name # Es. M42, M1, ecc.
    zip_filename = f"{target_name}_patches" # Non serve aggiungere .zip, lo fa shutil
    zip_output_path = target_dir / zip_filename
    
    print(f"\n🗜️  Compressione in corso...")
    print(f"   Sorgente:   {output_dir}")
    print(f"   Destinazione: {zip_output_path}.zip")
    
    try:
        shutil.make_archive(str(zip_output_path), 'zip', str(output_dir))
        print(f"   ✅ Archivio ZIP creato con successo!")
    except Exception as e:
        print(f"   ❌ Errore durante la creazione dello ZIP: {e}")
    # ===================================================

if __name__ == "__main__":
    main()