import os
import numpy as np
from astropy.io import fits
from pathlib import Path

# ================= CONFIGURAZIONE =================
# Percorso relativo (o metti il percorso assoluto se preferisci)
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

def check_folder(path_dir):
    print(f"\n--- CONTROLLO: {path_dir.name} ---")
    files = list(path_dir.glob('*.fits'))
    if not files:
        print("❌ Nessun file trovato.")
        return

    # Controlliamo il primo file della cartella
    f = files[0]
    print(f"📄 File ispezionato: {f.name}")
    
    try:
        with fits.open(f) as hdul:
            # Info generali
            hdul.info()
            
            # Cerchiamo i dati
            data = None
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim >= 2:
                    data = hdu.data
                    header = hdu.header
                    break
            
            if data is None:
                print("❌ ERRORE: Nessun dato immagine trovato nel file!")
                return

            # Statistiche Pixel
            d_min = np.nanmin(data)
            d_max = np.nanmax(data)
            d_mean = np.nanmean(data)
            
            print(f"📊 Dimensioni: {data.shape}")
            print(f"📉 Min: {d_min:.4f} | 📈 Max: {d_max:.4f} | ∅ Media: {d_mean:.4f}")
            
            if d_max == 0:
                print("⚠️  ALLERTA: L'immagine è completamente NERA (tutti zeri).")
                print("    Causa probabile: Nessuna sovrapposizione WCS durante la registrazione.")
            elif np.isnan(d_max):
                print("⚠️  ALLERTA: L'immagine contiene solo NaN.")
            else:
                print("✅ L'immagine contiene dati.")

            # Verifica Coordinate Header
            if 'CRVAL1' in header:
                print(f"🌍 Centro WCS: RA={header['CRVAL1']:.4f}, DEC={header['CRVAL2']:.4f}")
            else:
                print("❌ ERRORE: Nessuna coordinata WCS nell'header.")

    except Exception as e:
        print(f"❌ Errore apertura file: {e}")

def main():
    # Trova la cartella M1
    target_dir = ROOT_DATA_DIR / "M1" 
    
    if not target_dir.exists():
        print(f"Cartella non trovata: {target_dir}")
        return

    # Controlla input e output
    check_folder(target_dir / '2_solved_astap/observatory')     # Step 1 Output
    check_folder(target_dir / '3_registered_native/observatory') # Step 2 Output
    check_folder(target_dir / '3_registered_native/hubble')      # Step 2 Output (Reference)

if __name__ == "__main__":
    main()