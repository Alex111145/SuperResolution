
# Astro Super-Resolution Pipeline

Pipeline completa per la super-risoluzione di immagini astronomiche utilizzando Deep Learning. Il sistema combina dati di telescopi terrestri con immagini Hubble per addestrare modelli di upscaling avanzati.

## 📋 Indice

- [Panoramica](#-panoramica)
- [Struttura del Progetto](#-struttura-del-progetto)
- [Requisiti](#-requisiti)
- [Installazione](#-installazione)
- [Architettura dei Modelli](#-architettura-dei-modelli)
- [Pipeline Completa](#-pipeline-completa)
- [Configurazione Hardware](#-configurazione-hardware)
- [Troubleshooting](#-troubleshooting)
- [Risultati Attesi](#-risultati-attesi)
- [Riferimenti](#-riferimenti)

## 🌌 Panoramica

Questa pipeline trasforma immagini astronomiche a bassa risoluzione in output ad alta risoluzione attraverso:

- **Astrometric Solving**: Calibrazione WCS con ASTAP
- **Image Registration**: Allineamento spaziale tramite riproiezione
- **Patch Extraction**: Estrazione di coppie LR-HR allineate WCS-aware
- **Deep Learning**: Addestramento di modelli ibridi (RRDB + HAT)
- **Inference Scientifica**: Generazione TIFF 16-bit per analisi

## 📁 Struttura del Progetto

```
SuperResolution/
├── data/                          # Dataset organizzati per target
│   └── M42/                       # Esempio: Nebulosa di Orione
│       ├── 1_originarie/
│       │   ├── local_raw/         # Immagini osservatorio grezze
│       │   └── img_lights/        # Immagini Hubble grezze
│       ├── 2_solved_astap/        # Output ASTAP (WCS calibrato)
│       ├── 3_registered_native/   # Immagini riproiettate
│       ├── 4_quality_check/       # Overlay di controllo
│       ├── 6_patches_final/       # Coppie LR-HR estratte
│       ├── 7_dataset_ready_LOG/   # Dataset normalizzato (TIFF 16-bit)
│       └── 8_dataset_split/       # JSON train/val/test
│
├── models/                        # Architetture esterne (CRITICHE)
│   ├── BasicSR/                   # Repository BasicSR (RRDBNet)
│   │   └── basicsr/archs/rrdbnet_arch.py
│   └── HAT/                       # Repository HAT (Hybrid Attention Transformer)
│       └── hat/archs/hat_arch.py
│
├── weights/                       # Pesi pre-addestrati (opzionali)
│   ├── RRDB_pretrained.pth        # Pesi ImageNet per Stage 1
│   └── HAT_pretrained.pth         # Pesi pre-training HAT
│
├── outputs/                       # Risultati training/inference
│   └── M42_GPU_0/
│       ├── checkpoints/
│       │   ├── best_model.pth     # Modello con PSNR migliore
│       │   └── last.pth           # Ultimo checkpoint
│       ├── tensorboard/           # Log TensorBoard
│       └── test_results_tiff/     # Output inferenza
│
├── scripts/                       # Pipeline scripts
│   ├── Dataset_step1_datasetwcs.py      # Solving e Registrazione
│   ├── Dataset_step2_mosaicHSTObs.py    # Controllo allineamento
│   ├── Dataset_step3_extractpatches_Gaia.py  # Estrazione patch
│   ├── Dataset_step4_normalization.py   # Normalizzazione LOG
│   ├── Modello_1.py                     # Setup ambiente
│   ├── Modello_2.py                     # Creazione split
│   ├── Modello_3.py                     # Launcher training
│   ├── Modello_4.py                     # Finalizzazione modello
│   ├── Modello_5.py                     # Inferenza
│   └── Modello_supporto.py              # Worker training
│
└── src/                           # Moduli core
    ├── architecture.py            # Modello ibrido (RRDB+HAT)
    ├── dataset.py                 # Loader TIFF 16-bit
    ├── losses.py                  # Loss functions
    ├── metrics.py                 # PSNR/SSIM
    └── env_setup.py               # Setup percorsi
```

## 🔧 Requisiti

### Software

- Python 3.10+
- ASTAP (Astrometric Solver): [Download](https://www.hnsky.org/astap.htm)
- CUDA 11.8+ (per GPU NVIDIA)

### Hardware Consigliato

- **GPU**: NVIDIA H100/H200 (141 GB VRAM) oppure RTX 4090/A100
- **RAM**: 64 GB+
- **Storage**: 500 GB+ SSD (per dataset grandi)

### Dipendenze Python

```
torch>=2.0.0
torchvision
astropy
reproject
numpy<2.0
Pillow
scikit-image
opencv-python
tqdm
tensorboard
einops
timm
```

## 🚀 Installazione

### 1. Setup Base

```bash
git clone <repository-url>
cd SuperResolution
```

### 2. Installazione Modelli Esterni

**CRITICO**: I modelli BasicSR e HAT devono essere clonati nella cartella `models/`:

```bash
cd models

# BasicSR (per RRDBNet)
git clone https://github.com/XPixelGroup/BasicSR.git
cd BasicSR
pip install -e .
cd ..

# HAT (Hybrid Attention Transformer)
git clone https://github.com/XPixelGroup/HAT.git
cd HAT
pip install -r requirements.txt
cd ../..
```

**Verifica struttura**:

```
models/
├── BasicSR/basicsr/archs/rrdbnet_arch.py  ✓
└── HAT/hat/archs/hat_arch.py              ✓
```

### 3. Setup Ambiente Python

```bash
cd scripts
python Modello_1.py
```

Questo script:
- Installa PyTorch con supporto CUDA
- Configura tutte le dipendenze
- Verifica l'integrità dell'ambiente

### 4. Installazione ASTAP

- **Windows**: Scarica l'installer da [hnsky.org](https://www.hnsky.org/astap.htm)
- **Linux**: `sudo apt install astap` (o compila da sorgente)

Il pipeline cerca ASTAP in:
- `C:\Program Files\astap\astap.exe`
- `C:\Program Files (x86)\astap\astap.exe`
- Path di sistema

## 🧠 Architettura dei Modelli

### Modello Ibrido (HybridSuperResolutionModel)

Il sistema utilizza una **architettura a due stadi**:

```
Input (128×128)
    ↓
┌─────────────────────────┐
│  STAGE 1: RRDBNet       │  ← BasicSR (models/BasicSR)
│  - 23 RRDB Blocks       │
│  - Upscale 2× (→256px)  │
│  - Smoothing Layer      │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  STAGE 2: HAT           │  ← HAT Transformer (models/HAT)
│  - Hybrid Attention     │
│  - 6 Layers × 6 Heads   │
│  - Upscale 2× (→512px)  │
│  - Anti-Checkerboard    │
└─────────────────────────┘
    ↓
Output (512×512)
```

### Componenti Chiave

#### 1. RRDBNet (Stage 1)

- **Sorgente**: `models/BasicSR/basicsr/archs/rrdbnet_arch.py`
- **Funzione**: Estrazione features e primo upscale
- **Parametri**:
  - `num_feat=64`: Feature maps
  - `num_block=23`: Blocchi residui densi
  - `scale=2`: Fattore upscaling

#### 2. HAT (Stage 2)

- **Sorgente**: `models/HAT/hat/archs/hat_arch.py`
- **Funzione**: Raffinamento con Attention
- **Configurazione H200 (Memory-Safe)**:

```python
embed_dim=120        # Divisibile per num_heads=6
depths=[6,6,6,6,6,6] # 6 Transformer layers
window_size=16       # Attention window
```

#### 3. Anti-Checkerboard Layer

- Filtri Gaussiani per eliminare artefatti griglia
- Modalità: 'light', 'balanced', 'strong'

### Loss Function (CombinedLoss)

```python
Total Loss = λ₁·Charbonnier + λ₂·Perceptual + λ₃·Astro
```

- **Charbonnier Loss**: L1 robusta (principale)
- **Perceptual Loss**: VGG19 feature space
- **Astro Loss**: Penalizza errori su stelle/strutture luminose

## 🔄 Pipeline Completa

### FASE 1: Preparazione Dataset

#### Step 1: Astrometric Solving

```bash
python Dataset_step1_datasetwcs.py
```

**Cosa fa**:
- Cerca ASTAP nel sistema
- Risolve WCS per ogni immagine (coordinate celesti)
- Registra Hubble e Osservatorio su griglia comune
- Applica riproiezione via `reproject_interp`

**Input**:
- `data/M42/1_originarie/local_raw/*.fits` (Osservatorio)
- `data/M42/1_originarie/img_lights/*.fits` (Hubble)

**Output**:
- `data/M42/2_solved_astap/` (WCS calibrati)
- `data/M42/3_registered_native/` (Allineati)

**Configurazione FOV (se ASTAP fallisce)**:

```python
FORCE_FOV = 0.46  # Gradi (adatta al tuo telescopio)
USE_MANUAL_FOV = True
```

#### Step 2: Quality Check (Opzionale)

```bash
python Dataset_step2_mosaicHSTObs.py
```

Genera overlay RGB per verificare l'allineamento:
- **Verde**: Hubble
- **Magenta**: Osservatorio

**Output**: `data/M42/4_quality_check/M42_mosaic_check.png`

#### Step 3: Estrazione Patch

```bash
python Dataset_step3_extractpatches_Gaia.py
```

**Cosa fa**:
- Estrae patch sovrapposte da Hubble (512×512)
- Riproietta Osservatorio su WCS allineato (128×128)
- Genera coppie LR-HR con WCS identico
- Crea PNG di debug per validazione

**Parametri**:

```python
HR_SIZE = 512      # Dimensione patch Hubble
AI_LR_SIZE = 128   # Dimensione patch Osservatorio
STRIDE = 150       # Sovrapposizione patch
MIN_COVERAGE = 0.50  # % minima dati validi
```

**Output**:
- `data/M42/6_patches_final/pair_NNNNNN/`
  - `hubble.fits` (512×512)
  - `observatory.fits` (128×128)
- `data/M42/6_debug_visuals/` (prime 50 coppie)

#### Step 4: Normalizzazione LOG

```bash
python Dataset_step4_normalization.py
```

**Trasformazioni**:
- **Log Stretch**: `log(data + ε)` per comprimere dinamica
- **Percentile Clipping**: Taglia rumore e saturazioni
- **Espansione 16-bit**: Output TIFF (0-65535)

**Output**:
- `data/M42/7_dataset_ready_LOG/pair_NNNNNN/`
  - `hubble.tiff` (16-bit)
  - `observatory.tiff` (16-bit)

### FASE 2: Training del Modello

#### Step 1: Creazione Split

```bash
python Modello_2.py
```

Genera JSON per train/val/test (90/10 split):
- `data/M42/8_dataset_split/splits_json/train.json`
- `data/M42/8_dataset_split/splits_json/val.json`

#### Step 2: Configurazione Training

Modifica `Modello_3.py`:

```python
TARGET_NAME = "M42"  # Nome del target
NUM_GPUS = 1         # Numero GPU
```

#### Step 3: Avvio Training

```bash
python Modello_3.py
```

**Hyperparameters** (in `Modello_supporto.py`):

```python
BATCH_SIZE = 3          # Per H200 (141GB VRAM)
ACCUM_STEPS = 20        # Gradient Accumulation
LR = 4e-4               # Learning Rate
TOTAL_EPOCHS = 150
```

**Monitoraggio con TensorBoard**:

```bash
tensorboard --logdir=outputs/M42_GPU_0/tensorboard
```

**Metriche tracciate**:
- Loss totale e componenti (Charbonnier, Astro, Perceptual)
- PSNR/SSIM su validation set
- Learning rate
- Immagini di preview (ogni epoca)

#### Step 4: Finalizzazione

```bash
python Modello_4.py --target M42
```

Copia il best checkpoint in:
- `outputs/M42/final_weights/best.pth`

### FASE 3: Inferenza

```bash
python Modello_5.py
```

**Output**:
- `outputs/M42/test_results_tiff/tiff_science/` (TIFF 16-bit)
- `outputs/M42/test_results_tiff/png_preview/` (Comparazioni visive)

**Formato Output**:
- **TIFF Scientifici**: Range completo 16-bit per analisi
- **PNG Preview**: [LR_upscaled | SR | HR_ground_truth]

## ⚙️ Configurazione Hardware

### GPU NVIDIA H100/H200 (Consigliato)

**Configurazione Memory-Safe**:

```python
# architecture.py
embed_dim=120           # Ridotto da 180
depths=[6,6,6,6,6,6]   # 6 layers invece di 12
```

**Ottimizzazioni**:

```python
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
```

### RTX 4090 / A100 (Alternative)

Riduci batch size e accumulation steps:

```python
BATCH_SIZE = 1          # Invece di 3
ACCUM_STEPS = 60        # Invece di 20
```

### Multi-GPU Training

Modifica `Modello_3.py`:

```python
NUM_GPUS = 4  # Esempio: 4× H100
```

Ogni GPU addestrerà su uno split del dataset.

## 🔍 Troubleshooting

### Problema: ASTAP non trova soluzioni

**Sintomo**: File `.fits` senza WCS dopo Step 1

**Soluzioni**:

1. Verifica FOV manuale:

```python
FORCE_FOV = 0.46  # Calcola: (altezza_sensore_mm / focale_mm) * (180/π)
USE_MANUAL_FOV = True
```

2. Controlla header FITS:

```python
from astropy.io import fits
hdul = fits.open('image.fits')
print(hdul[0].header)  # Cerca FOCALLEN, XPIXSZ
```

### Problema: Patch completamente nere

**Sintomo**: Dataset vuoto o training loss = 0

**Soluzioni**:

1. Controlla normalizzazione:

```bash
python debugmodello.py
```

2. Verifica percentili:

```python
LOWER_PERCENTILE = 1.0   # Aumenta se troppo scuro
UPPER_PERCENTILE = 98.0  # Riduci per evidenziare faint objects
```

### Problema: RuntimeError (Mixed Precision)

**Sintomo**: `Expected tensor for argument to have the same type`

**Soluzione**: Conversione esplicita float32:

```python
# In Modello_supporto.py (già implementato)
metrics.update(v_pred.float(), v_hr.float())
```

### Problema: Import Error (BasicSR/HAT)

**Sintomo**: `ModuleNotFoundError: No module named 'basicsr'`

**Soluzioni**:

1. Verifica cartella `models/`:

```bash
ls -la models/BasicSR/basicsr/archs/rrdbnet_arch.py
ls -la models/HAT/hat/archs/hat_arch.py
```

2. Reinstalla:

```bash
cd models/BasicSR && pip install -e . && cd ../..
```

### Problema: Out of Memory (OOM)

**Sintomo**: `CUDA out of memory`

**Soluzioni**:

1. Riduci batch size:

```python
BATCH_SIZE = 1
```

2. Usa solo Stage 1 (disabilita HAT):

```python
# architecture.py
self.has_stage2 = False
```

3. Riduci dimensione patch:

```python
HR_SIZE = 256    # Invece di 512
AI_LR_SIZE = 64  # Invece di 128
```

## 📊 Risultati Attesi

### Metriche Target

| Metrica | Baseline (Bicubic) | Target Modello |
|---------|-------------------|----------------|
| PSNR    | ~28 dB            | 32-35 dB       |
| SSIM    | ~0.85             | 0.92-0.95      |

### Esempio Output

Input LR (128×128) → Output SR (512×512) → Ground Truth HR

- Dettagli stellari recuperati
- Nebulose con texture preservate
- Assenza di artefatti griglia

## 📚 Riferimenti

### Modelli Utilizzati

- **RRDBNet**: ESRGAN Paper
  - Repository: [BasicSR](https://github.com/XPixelGroup/BasicSR)

- **HAT**: Hybrid Attention Transformer
  - Repository: [HAT Official](https://github.com/XPixelGroup/HAT)

### Dataset

- **Hubble Legacy Archive**: [HST Data](https://hla.stsci.edu/)
- **ASTAP**: [Astrometric Solver](https://www.hnsky.org/astap.htm)
