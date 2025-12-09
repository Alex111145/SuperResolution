# 🌌 SuperResolution: Modello Ibrido per Astro-Fotografia

Questo progetto implementa una pipeline completa per l'Astro-fotografia Super Resolution (SR), utilizzando un'architettura ibrida all'avanguardia (RRDBNet + HAT) e una complessa pipeline di elaborazione dati basata sull'allineamento celeste (WCS) per migliorare immagini astronomiche a bassa risoluzione (LR) basate su osservatori terrestri, utilizzando come riferimento l'altissima risoluzione del telescopio spaziale Hubble (HR).

---

## 1. ⚙️ Setup e Installazione

Questa sezione spiega come configurare l'ambiente di lavoro, installare le dipendenze Python e configurare il solutore astrometrico ASTAP, un componente cruciale della pipeline.

### 1.1. Dipendenze Python

È fortemente consigliato l'uso di un ambiente virtuale (venv).

```bash
# 1. Spostati nella cartella del progetto (SuperResolution-GTX5000-TRAIN-E-SANITY)
cd SuperResolution-GTX5000-TRAIN-E-SANITY

# 2. Crea e attiva l'ambiente virtuale
python3 -m venv venv
source venv/bin/activate
# Oppure (su Windows): .\venv\Scripts\activate

# 3. Installa le librerie (incluso PyTorch con supporto CUDA 11.8)
# Nota: La riga 'extra-index-url' nel requirements.txt specifica la versione CUDA.
pip install -r requirements.txt
```

Le dipendenze chiave includono: `torch>=2.0.0` (con CUDA), `astropy`, `reproject`, `basicsr>=1.4.2` (necessario per SwinIR/RRDBNet) e `tqdm`.

### 1.2. Setup di ASTAP (Astrometry Solving)

ASTAP è essenziale per risolvere l'Astrometria (WCS) dei file FITS grezzi e allineare geometricamente le immagini LR (Osservatorio) con quelle HR (Hubble).

Il progetto si aspetta che l'eseguibile e il database siano accessibili, preferibilmente tramite la cartella `astap_local` creata accanto alla cartella principale del progetto.

**Pulizia (Opzionale):**

```bash
rm -rf astap_local temp_db astap_old.deb d50_star_database.deb
```

**Download dei pacchetti:** Scarica il programma (`astap_amd64.deb` o versione compatibile) e il database stellare (`d50_star_database.deb`).

**Estrazione e Setup (Consigliato):**

```bash
# 1. Crea la cartella locale
mkdir astap_local

# 2. Estrai il programma ASTAP
dpkg -x astap_old.deb astap_local

# 3. Dai i permessi di esecuzione
chmod +x astap_local/opt/astap/astap

# 4. Estrai il Database in una cartella temporanea
mkdir temp_db
dpkg -x d50_star_database.deb temp_db

# 5. Sposta il Database accanto al programma (nella struttura interna)
mv temp_db/opt/astap/* astap_local/opt/astap/

# 6. Rimuovi la cartella temporanea
rm -rf temp_db
```

I file Python come `Dataset_1.py` cercheranno automaticamente l'eseguibile in questa struttura.

---

## 2. 🚀 Pipeline di Preparazione Dati (Dataset)

Il processo di creazione del dataset di Super Resolution è diviso in 4 fasi principali, tutte eseguite tramite gli script nella cartella `scripts/`.

### FASE 1: Risoluzione Astrometrica (WCS)

**Script:** `scripts/Dataset_1.py` (o `Dataset_step1_datasetwcs.py`)

**Azione:** Applica la risoluzione astrometrica (WCS) a tutti i file FITS grezzi (sia Osservatorio che Hubble) usando ASTAP.

**Dettagli:** Utilizza la funzione `solve_with_astap` con due tentativi (veloce e blind solve con FOV manuale) per garantire l'associazione delle coordinate celesti ad ogni immagine. Successivamente, allinea (registra) tutte le immagini (LR e HR) sulla griglia WCS del primo file Hubble, salvandole in `3_registered_native`.

### FASE 2: Controllo Mosaico (Qualità Allineamento)

**Script:** `scripts/Dataset_2.py` (o `Dataset_step2_mosaicHSTObs.py`)

**Azione:** Genera un'immagine di overlay tra l'immagine master dell'Osservatorio e un mosaico di tutti i tasselli Hubble riproiettati sulla stessa griglia WCS.

**Dettagli:** Usa la riproiezione (`reproject_interp`) per creare il mosaico Hubble. L'overlay finale usa il Verde per Hubble e il Magenta per l'Osservatorio per evidenziare visivamente la qualità dell'allineamento geometrico prima dell'estrazione delle patch.

### FASE 3: Estrazione delle Patch

**Script:** `scripts/Dataset_3.py` (o `Dataset_step3_extractpatches.py`)

**Azione:** Taglia le immagini registrate (HR: 512x512, LR: 128x128) in piccole coppie di patch adatte al training della AI.

**Dettagli:** L'estrazione è WCS-Aware: la WCS della patch HR viene usata per derivare la WCS della patch LR, garantendo che i pixel si riferiscano alla stessa area del cielo, anche se hanno risoluzioni diverse. Vengono create anche card diagnostiche (`save_diagnostic_card`) per controllare l'errore di allineamento in secondi d'arco.

### FASE 4: Normalizzazione Robusta (Deep Black)

**Script:** `scripts/Dataset_4.py` (o `Dataset_step4_normalization.py`)

**Azione:** Converte i file FITS (float a 32-bit) in file TIFF a 16-bit con normalizzazione 0-65535 (corrispondente a 0.0-1.0 per la AI).

**Dettagli Tecnici (Il "Cervello" del Contrasto):**

- **Log Stretch:** Applica `log1p` ai dati per comprimere la dinamica, rendendo visibili le nebulose deboli accanto alle stelle luminose.
- **Percentile Clip:** La funzione `calculate_robust_stats` campiona i pixel e calcola i limiti globali: il Nero (`global_min`) è fissato al 4.0° percentile (`BLACK_CLIP_PERCENTILE`), che rimuove il rumore di fondo pur preservando i dettagli del gas.
- **Output 16-bit:** Le immagini finali sono salvate come TIFF 16-bit, che offrono 65.536 livelli di grigio, essenziali per le sfumature delle nebulose.

---

## 3. 🧠 Architettura del Modello e Funzioni Core

Il progetto utilizza un modello di Super Resolution ibrido e altamente ottimizzato, definito nei moduli `src/`.

### 3.1. Architettura Ibrida (`src/architecture_train.py`)

Il modello combina due stadi per massimizzare sia la ricostruzione della struttura che la texture fine:

- **Stage 1: RRDBNet (Base):** Utilizzato per la ricostruzione iniziale dell'immagine a bassa risoluzione. RRDBNet (Real-ESRGAN/BasicSR) è un'architettura robusta con 23 blocchi.
- **Stage 2: HAT (Texture):** Se disponibile (importato da `models/HAT`), un Transformer gerarchico viene applicato per raffinare ulteriormente i dettagli e le texture sottili, aumentando la risoluzione di un ulteriore fattore 2.
- **AntiCheckerboardLayer:** Un livello di smoothing basato su un kernel Gaussiano (ad esempio, con modalità balanced 5x5) per mitigare gli artefatti a scacchiera tipici degli upsampling in Deep Learning.

### 3.2. Loss e Metriche

| File | Classe | Dettagli |
|------|--------|----------|
| `src/losses_train.py` | `TrainStarLoss` | Una Loss L1 modificata che applica un peso di 500.0x ai pixel luminosi (target > 0.02). Questo star-weighting è fondamentale per garantire che la rete ricostruisca le stelle in modo preciso. |
| `src/losses_sanity.py` | `SanityStarLoss` | Versione aggressiva della loss con lo stesso peso 500.0x. |
| `src/metrics_train.py` | `TrainMetrics` | Calcola PSNR (Peak Signal-to-Noise Ratio) e SSIM (Structural Similarity Index Measure). |

### 3.3. Dataset Loader (`src/dataset.py`)

La classe `AstronomicalDataset` gestisce:

- **Caricamento 16-bit:** Legge i file TIFF a 16-bit generati in Fase 4, dividendo i valori per 65535.0 per riportarli nel range 0.0-1.0.
- **Data Augmentation:** Applica rotazioni di 90° e flip casuali (orizzontali/verticali) durante il Full Training per rendere il modello più robusto.

---

## 4. 💻 Flussi di Lavoro (Esecuzione)

Una volta che il dataset è pronto (dopo la Fase 4), è possibile avviare il training.

### 4.1. Sanity Check (Test di Overfitting)

L'obiettivo è testare rapidamente l'architettura su una singola patch, assicurandosi che il modello possa raggiungere un overfitting perfetto.

**Preparazione:** Seleziona la patch di test.

```bash
python3 scripts/SanityCheck_Prepare_0.py
```

Questo script chiede all'utente di selezionare una singola patch tra quelle disponibili e la duplica per i set Train, Val e Test.

**Lancio Training:**

```bash
python3 scripts/SanityCheck_Launcher_1.py
```

Questo script avvia il `SanityCheck_Worker.py`. Richiede la selezione di una sola GPU (max 1).

Il worker esegue 1000 epoche e salva immagini di debug ogni 20 epoche in `outputs/TARGET_SANITY_CHECK_RUN/images/`.

### 4.2. Full Training

L'addestramento completo del modello sull'intero dataset.

**Preparazione:** Split del dataset.

```bash
python3 scripts/Train_Prepare_0.py
```

Lo script divide tutti i file disponibili in: 80% Train, 10% Val, 10% Test.

**Lancio Training:**

```bash
python3 scripts/Train_Launcher_1.py
```

Il launcher permette la selezione di una o più GPU (`select_gpu_ids`).

Il `Train_Worker.py` esegue 300 epoche con Data Augmentation attivata, salvando i checkpoint del modello e loggando su TensorBoard.

### 4.3. Inference (Test Finale)

**Script:** `scripts/Train_s_Inference_4.py`

**Azione:** Carica il miglior modello salvato (`best.pth`) ed esegue l'inferenza sul set di Test (o Validation), salvando i risultati come TIFF 16-bit (per l'analisi scientifica) e come PNG di preview.

**Output:** Genera le metriche finali (PSNR e SSIM) e salva i risultati in `outputs/TARGET_NAME/test_results_tiff/`.

### 4.4. Utility (Etichettatura Immagine)

**Script:** `scripts/SanityCheck_print_2.py`

**Azione:** Prende un'immagine di collage (ad esempio, l'output PNG del test) e aggiunge un'intestazione etichettando le sezioni "Input", "Risultato", "Target". È utile per la documentazione finale dei risultati.

```bash
python3 scripts/SanityCheck_print_2.py
# Modifica le variabili FILE_INPUT e FILE_OUTPUT nello script prima di lanciare.
```

---

## 5. 🔎 Monitoraggio (TensorBoard)

Durante il training, tutte le metriche (Loss, PSNR, ecc.) sono registrate.

```bash
# Avvia TensorBoard dalla root del progetto
tensorboard --logdir=outputs --port=6006 --bind_all
```

Successivamente, apri l'URL fornito (es. `http://0.0.0.0:6006`) nel tuo browser.