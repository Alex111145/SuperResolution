import os
import sys
from pathlib import Path
import numpy as np
from astropy.io import fits
from tqdm import tqdm
import warnings

# Ignora i warning di Astropy
warnings.filterwarnings('ignore', category=UserWarning, module='astropy')

# ============================================================================
# CONFIGURAZIONE PATH E PARAMETRI
# ============================================================================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# PARAMETRI DI CLASSIFICAZIONE STRUTTURALE (Empirici per distinguere Background)
# La Deviazione Standard misura la quantità di struttura (contrasto / dettaglio)
MIN_STD_DEV_FOR_SIGNAL = 0.05  # Soglia empirica (0.05 è tipico per distinguere background da struttura)

# Definisci i parametri base di copertura solo per riferimento:
MIN_PIXEL_VALUE = 0.0001
MIN_COVERAGE = 0.40

# ============================================================================
# FUNZIONI DI UTILITÀ
# ============================================================================

def select_target_directory():
    """Mostra un menu per selezionare una cartella target (es. M33)."""
    # ... (omessa per brevità, la funzione è la stessa di prima)
    print("\n" + "📂"*35)
    print("ANALISI PATCHES: SELEZIONE CARTELLA TARGET".center(70))
    print("📂"*35)
    
    subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs', '__pycache__']]
    if not subdirs:
        print(f"❌ ERRORE: Nessuna cartella target trovata in {ROOT_DATA_DIR}")
        return None

    print("\nCartelle target disponibili:")
    for i, dir_path in enumerate(subdirs):
        print(f"   {i+1}: {dir_path.name}")

    while True:
        try:
            choice_str = input(f"👉 Seleziona un numero (1-{len(subdirs)}) o 'q' per uscire: ").strip()
            if choice_str.lower() == 'q': return None 
            choice = int(choice_str)
            choice_idx = choice - 1
            if 0 <= choice_idx < len(subdirs):
                return subdirs[choice_idx]
            else:
                print("❌ Scelta non valida.")
        except ValueError:
            print("❌ Input non valido.")


def analyze_fits_patch_structure(hubble_path):
    """
    Carica il file FITS di Hubble e classifica la patch in base alla Deviazione Standard.
    """
    try:
        with fits.open(hubble_path) as hdul:
            data = hdul[0].data
            
            # Pulisci NaN e Inf
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            if data.ndim > 2:
                data = data[0]

            # Normalizzazione approssimativa al range [0, 1] se non è già normalizzato
            # (Assumiamo che il flusso massimo sia 1.0 o usiamo il max del batch se conosciuto)
            # Per questa analisi, la Deviazione Standard è robusta anche senza normalizzazione perfetta.
            
            # Calcola la Deviazione Standard: questa misura la struttura e il contrasto.
            std_dev = np.std(data)
            
            # Calcola la copertura 'tecnica' per riferimento
            coverage_check = np.count_nonzero(data > MIN_PIXEL_VALUE) / data.size
            
            # Classifica la patch
            if std_dev >= MIN_STD_DEV_FOR_SIGNAL:
                return 'valid_signal', std_dev, coverage_check
            else:
                return 'black_background', std_dev, coverage_check
            
    except Exception as e:
        # print(f"Errore lettura FITS {hubble_path.name}: {e}")
        return 'error', 0.0, 0.0

# ============================================================================
# FUNZIONE PRINCIPALE
# ============================================================================

def main():
    target_dir = select_target_directory()
    if not target_dir: return

    patches_path = target_dir / "6_patches_final"
    
    if not patches_path.is_dir():
        print(f"\n❌ ERRORE: Cartella delle patch non trovata in: {patches_path}")
        print("   Esegui prima 'Dataset_step3_extractpatches_Gaia copy.py'.")
        return

    print(f"\n🔍 Analisi struttura patch FITS (Hubble) in: {patches_path.name}")

    pair_folders = sorted(list(patches_path.glob("pair_*")))
    
    if not pair_folders:
        print(f"⚠️ Nessuna cartella 'pair_XXX' trovata in {patches_path}")
        return

    total_pairs = len(pair_folders)
    count = {'valid_signal': 0, 'black_background': 0, 'error': 0}
    
    results_list = []

    for folder in tqdm(pair_folders, desc="Analisi struttura patch"):
        hubble_file = folder / "hubble.fits"
        
        if hubble_file.exists():
            result, std_dev, coverage = analyze_fits_patch_structure(hubble_file)
            count[result] += 1
            results_list.append((folder.name, result, std_dev, coverage))
        else:
            count['error'] += 1

    # ================= STAMPA RISULTATI =================
    print("\n" + "="*70)
    print("📊 REPORT ANALISI STRUTTURA PATCHES (Hubble)".center(70))
    print("="*70)
    
    perc_signal = count['valid_signal'] / total_pairs * 100
    perc_black = count['black_background'] / total_pairs * 100
    
    print(f"   Totale Coppie Analizzate: {total_pairs}")
    print(f"   Soglia Deviazione Standard (STRUTTURA): >= {MIN_STD_DEV_FOR_SIGNAL:.4f}")
    
    print("\n   CLASSIFICAZIONE STRUTTURALE:")
    print(f"   ✅ Patch con Dettagli/Struttura: {count['valid_signal']} ({perc_signal:.1f}%)")
    print(f"   ◼️ Patch Nere/Background Piatti: {count['black_background']} ({perc_black:.1f}%)")
    
    if count['error'] > 0:
        print(f"   ❓ Errori (File mancanti/corrotti): {count['error']}")
    
    # Stampa un esempio di patch per chiarezza
    print("\n   Esempi di Classificazione:")
    
    valid_examples = [r for r in results_list if r[1] == 'valid_signal'][:3]
    black_examples = [r for r in results_list if r[1] == 'black_background'][:3]

    for name, _, std_dev, coverage in valid_examples:
        print(f"     - {name} (Valida/Struttura): Dev.Std={std_dev:.4f}, Copertura={coverage*100:.1f}%")
        
    for name, _, std_dev, coverage in black_examples:
        print(f"     - {name} (Nera/Background): Dev.Std={std_dev:.4f}, Copertura={coverage*100:.1f}%")
        
    print("="*70)
    
    if perc_black > 40:
        print(f"💡 Suggerimento: La percentuale di patch 'Nere/Background' è alta ({perc_black:.1f}%).")
        print("  Questo è normale e salutare (insegna il background) ma se è troppo alto,")
        print("  potrebbe significare che STRIDE è troppo piccolo o che la tua MIN_COVERAGE è troppo bassa.")
        print(f"  Hai già un buon mix. Ora aumenta STRIDE a 8 (o meno) per avere più coppie.")


if __name__ == "__main__":
    main()