# 🌌 Guida Completa alla Normalizzazione Astronomica per Deep Learning

Questo documento spiega come trasformiamo i dati grezzi (FITS) in input digeribili dalla Rete Neurale (Tensori 0-1), garantendo che la AI veda correttamente sia il nero profondo dello spazio che i dettagli tenui delle nebulose.

## 1. Il Cuore del Sistema: `Dataset_step4_normalization.py`

Questo script è responsabile della pulizia e della preparazione dei dati. Non si limita a convertire i file, ma applica una trasformazione matematica intelligente basata sull'statistica globale.

### 🛠️ Componenti Chiave del Codice

#### A. Classe `RawFitsDataset` (Efficienza)

Invece di caricare migliaia di immagini pesanti nella RAM tutte insieme, usiamo una classe personalizzata di PyTorch.

```python
class RawFitsDataset(Dataset):
    def __getitem__(self, idx):
        # Apre il file FITS solo quando serve
        with fits.open(path) as hdul:
            data = hdul[0].data
            # Gestisce valori corrotti (NaN/Inf)
            data = np.nan_to_num(data, ...)
            return torch.from_numpy(data)
```

**Cosa fa:** Insegna a PyTorch come leggere un singolo file FITS.

**Perché:** Permette l'uso dei DataLoader, che caricano i dati in parallelo usando più processori (CPU workers).

#### B. `calculate_robust_stats` (Il "Cervello")

Questa funzione determina matematicamente cosa è "Nero" e cosa è "Bianco" per l'intero dataset.

**Campionamento Casuale (Sampling):**
Non legge ogni pixel (sarebbe troppo lento), ma preleva un campione casuale di 4000 pixel da ogni immagine.

```python
indices = torch.randperm(valid_pixels.numel())[:num_take]
```

**Log Stretch (Compressione Dinamica):**
Prima di calcolare le statistiche, applica un logaritmo naturale.

```python
batch = torch.log1p(torch.maximum(batch, torch.tensor(0.0)))
```

**Motivo:** Le stelle sono milioni di volte più luminose delle nebulose. Il logaritmo "avvicina" questi valori, rendendo le nebulose visibili invece che invisibili.

**Percentili (Il Segreto del Contrasto):**
Qui usiamo il parametro `BLACK_CLIP_PERCENTILE` (che abbiamo impostato al 4%).

```python
global_min = np.percentile(full_sample, 4.0)   # Il Nero
global_max = np.percentile(full_sample, 99.99) # Il Bianco
```

- **Min (4%):** Dice: "Tutto ciò che è più scuro del 4% dei pixel totali è rumore elettronico. Cancellalo (fallo diventare 0)". Questo pulisce il fondo cielo.
- **Max (99.99%):** Dice: "Il valore più alto è dato dalle stelle, ma ignoriamo lo 0.01% di pixel 'impazziti' (raggi cosmici)".

#### C. Normalizzazione e Salvataggio (`main`)

Una volta trovati i limiti globali (`global_min`, `global_max`), lo script processa ogni immagine:

**Scaling 0-1 Globale:**

$$Valore = \frac{Pixel - Min}{Max - Min}$$

Tutti i pixel vengono portati matematicamente nel range tra 0.0 e 1.0.

**Clipping:**

```python
d_h_norm = np.clip(d_h_norm, 0, 1)
```

Taglia via definitivamente il rumore sotto il 4% (diventa 0 precisi) e i raggi cosmici (diventano 1 precisi).

**Conversione a 16-bit:**

```python
h_u16 = (d_h_norm * 65535).astype(np.uint16)
```

Moltiplica per 65.535 e salva come TIFF.

**Perché 16-bit?** Un'immagine normale (JPG/PNG) ha 256 livelli di grigio. Il 16-bit ne ha 65.536. Questo permette sfumature incredibilmente morbide per le nebulose.

## 2. L'Ingresso nel Modello: `src/dataset.py`

Quando lanci l'addestramento, il modello non legge i file FITS, ma questi TIFF puliti. Ecco come li "mangia".

### Il Metodo `_load_tiff_as_tensor`

```python
def _load_tiff_as_tensor(self, path):
    # 1. Carica l'immagine TIFF (valori 0 - 65535)
    img = Image.open(path)
    arr = np.array(img, dtype=np.float32)
    
    # 2. RITORNO AL RANGE 0-1 (Cruciale!)
    arr = arr / 65535.0
    
    # 3. Trasforma in Tensore per la GPU
    tensor = torch.from_numpy(arr)
    return tensor
```

- **Lettura:** Carica il file TIFF creato prima.
- **Divisione:** Divide ogni pixel per 65535.0.
  - Se un pixel era 0 (nero) → $0 / 65535 = 0.0$
  - Se un pixel era 65535 (bianco) → $65535 / 65535 = 1.0$
  - Se un pixel era 32768 (grigio medio) → $32768 / 65535 = 0.5$
- **Risultato:** La rete neurale riceve una matrice di numeri puri ("float") pronti per i calcoli matematici dell'intelligenza artificiale.

## 3. Riassunto del Flusso ("La Pipeline")

| Step | File | Cosa succede ai dati? | Valori dei Pixel |
|------|------|----------------------|------------------|
| 1. Input | `.fits` | Dati grezzi dal telescopio, pieni di rumore | -100 a 500.000+ |
| 2. Statistica | `Dataset_step4...` | Calcolo del "nero" (4%) e Logaritmo | Analisi globale |
| 3. Pulizia | `Dataset_step4...` | Sottrazione del rumore e compressione | 0.0 a 1.0 |
| 4. Archiviazione | `.tiff` | Salvataggio ad alta precisione su disco | 0 a 65535 (Interi) |
| 5. Training | `src/dataset.py` | Caricamento e riconversione per la GPU | 0.0 a 1.0 (Float) |

## Perché il 4% è il numero magico?

Abbiamo scelto `BLACK_CLIP_PERCENTILE = 4.0` perché:

- **< 4% (es. 1%):** Lascia passare il rumore elettronico del sensore → Sfondo Grigio/Nebbia.
- **> 10% (es. 40%):** Taglia via i dati reali delle nebulose deboli → Immagini nere vuote.
- **4%:** Taglia solo il rumore, lasciando il nero profondo ma preservando i dettagli del gas.