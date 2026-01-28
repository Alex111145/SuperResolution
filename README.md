# Pipeline di Preparazione Dataset: Super-Risoluzione Astronomica (HST vs Obs)

Questo branch contiene una serie di script Python progettati per automatizzare il processo di creazione di un dataset per l'addestramento di modelli di Intelligenza Artificiale dedicati alla Super-Risoluzione di immagini astronomiche.

L'obiettivo è accoppiare immagini ad alta risoluzione del telescopio spaziale Hubble (HST) con immagini a bassa risoluzione acquisite da osservatori terrestri (Obs).

## Struttura della Pipeline

Il processo è suddiviso in 5 step sequenziali:

### 1. Risoluzione Astrometrica e Registrazione (Dataset_step1_datasetwcs.py)
* **Astrometry Solving**: Utilizza l'eseguibile ASTAP per risolvere le coordinate WCS (World Coordinate System) delle immagini grezze (.fits).
* **Riproiezione**: Allinea le immagini Hubble e quelle dell'osservatorio sulla stessa griglia di coordinate celesti utilizzando la libreria reproject.
* **Output**: Salva le immagini risolte e registrate in cartelle dedicate (2_solved_astap e 3_registered_native).

### 2. Controllo Qualità Mosaico (Dataset_step2_mosaicHSTObs.py)
* **Visualizzazione**: Crea un mosaico di Hubble proiettato sulla tela dell'osservatorio per verificare l'allineamento.
* **Overlay**: Genera un'immagine di confronto RGB (R=Hubble, G=Obs) per identificare eventuali errori di registrazione prima dell'estrazione delle patch.

### 3. Estrazione Patch Dinamica (Dataset_step3_extractpatches.py)
* **Multi-processing**: Utilizza ProcessPoolExecutor per estrarre migliaia di coppie di patch dalle immagini originali.
* **Stride Adattivo**: Calcola automaticamente il passo (stride) di campionamento in base al numero di patch desiderate o a dati storici per bilanciare il dataset.
* **Filtro Copertura**: Scarta le patch con segnale insufficiente basandosi sulla costante MIN_COVERAGE.
* **Check Diagnostico**: Salva immagini di debug per ogni patch per verificare la sovrapposizione a livello di pixel.

### 4. Normalizzazione e Conversione TIFF (Dataset_step4_normalization.py)
* **Log-Stretch**: Applica una trasformazione logaritmica (log1p) per comprimere il range dinamico dei dati FITS.
* **Robust Stats**: Calcola i percentili su tutto il dataset per definire il range dinamico (es. BLACK_CLIP_PERCENTILE).
* **16-bit TIFF**: Converte i dati scientifici a 32-bit in immagini TIFF a 16-bit per i modelli di IA.

### 5. Split del Dataset (prepare_data.py)
* **Train/Val/Test**: Suddivide le coppie di immagini in set di addestramento (80%), validazione (10%) e test (10%).
* **JSON Manifest**: Genera file .json con i percorsi dei file per facilitare il caricamento tramite DataLoader.

## Istruzioni per l'uso

### Prerequisiti
* ASTAP installato nel sistema.
* Librerie Python: numpy, astropy, reproject, tqdm, scikit-image, pillow, torch, matplotlib.

### Esecuzione
Eseguire gli script in ordine numerico all'interno della cartella misc:

1. python Dataset_step1_datasetwcs.py
2. python Dataset_step3_extractpatches.py
3. python Dataset_step4_normalization.py
4. python prepare_data.py

## Configurazione
Le impostazioni principali sono definite come costanti all'inizio di ogni script:
* HR_SIZE: 512.
* AI_LR_SIZE: 128.
* TRAIN_RATIO: 0.8.
