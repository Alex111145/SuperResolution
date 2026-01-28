# Documentazione Tecnica Dettagliata: HAT-Real (Hybrid Attention Transformer)

Questo repository implementa un framework avanzato per la Super-Risoluzione (SR) di immagini astronomiche, ottimizzato per accoppiare dati di osservatori terrestri con immagini del telescopio spaziale Hubble. Il sistema utilizza un'architettura Hybrid Attention Transformer (HAT) per superare i limiti delle tradizionali reti convoluzionali.

## 1. Architettura del Modello (models/)

Il cuore del sistema è progettato per gestire la complessità dei dati astronomici attraverso tre moduli principali:

* **Generatore HAT (hat_arch.py)**: Implementa il trasformatore ibrido che combina l'efficacia delle convoluzioni nel catturare dettagli locali con la potenza della self-attention per le relazioni globali. Questa struttura permette di ricostruire strutture stellari e nebulose con alta precisione.
* **Sistema Discriminatore (discriminator.py)**: Include diverse architetture per il training avversariale. Utilizza modelli basati su UNet o VGG (srvgg_arch.py) per analizzare l'immagine sia a livello di texture che di struttura complessiva, spingendo il generatore a produrre risultati indistinguibili dalle immagini originali di Hubble.
* **Wrapper Ibrido (hybridmodels.py)**: Una classe di astrazione che facilita la gestione simultanea di generatore e discriminatore, ottimizzando il passaggio dei gradienti e il caricamento dei pesi durante le fasi di addestramento complesso.

## 2. Gestione Dati e Pipeline (dataset/)

La pipeline di caricamento è specificamente progettata per i formati scientifici:

* **astronomical_dataset.py**: Implementa un `DataLoader` personalizzato che legge immagini TIFF a 16-bit. A differenza delle immagini standard a 8-bit, questo preserva l'alta gamma dinamica necessaria per l'astronomia.
* **Data Augmentation**: Il dataset applica trasformazioni casuali (rotazioni di 90°, 180°, 270° e flip orizzontali/verticali) per garantire che il modello sia invariante all'orientamento degli oggetti celesti.

## 3. Logica di Addestramento (train_hat.py)

L'addestramento segue un approccio multi-fase per garantire stabilità e qualità:

* **Funzioni di Perdita (utils/gan_losses.py, losses_train.py)**:
    * **Pixel Loss (L1)**: Assicura che i valori cromatici e di luminosità siano accurati.
    * **Perceptual Loss**: Utilizza una rete VGG pre-addestrata per confrontare le feature estratte, garantendo che le strutture morfologiche siano preservate.
    * **Adversarial Loss**: Fornisce il "tocco finale" per la nitidezza, eliminando l'effetto sfocato tipico delle sole perdite MSE/L1.
* **Metriche di Monitoraggio (utils/metrics.py)**: Vengono calcolati costantemente PSNR (Peak Signal-to-Noise Ratio) e SSIM (Structural Similarity Index) per valutare oggettivamente il miglioramento della qualità.

## 4. Inferenza e Deployment (infer.py)

Lo script di inferenza è ottimizzato per l'uso pratico su dati reali:

* **Processing Batch**: Permette di elaborare intere cartelle di immagini provenienti da sessioni osservative terrestri.
* **Tiling**: Se necessario, gestisce immagini di grandi dimensioni suddividendole in patch per evitare saturazione della memoria GPU, ricomponendole poi senza cuciture visibili.

## 5. Setup e Automazione (start.py)

Il file `start.py` automatizza la preparazione dell'ambiente:
* Verifica la compatibilità dei driver CUDA.
* Configura le cartelle di output per log, checkpoint e immagini campionate.
* Installa le dipendenze mancanti tramite `env_setup.py`.

## Istruzioni per l'uso

1.  **Requisiti**: Python 3.8+, PyTorch 2.0+, GPU con almeno 8GB di VRAM.
2.  **Preparazione**: Assicurarsi che i dati siano stati processati dal branch `misc` (registrazione e normalizzazione).
3.  **Training**: Eseguire `python train_hat.py` per avviare il ciclo di addestramento.
4.  **Inferenza**: Usare `python infer.py --input <cartella_input> --model_path <checkpoint.pth>` per generare i risultati.
