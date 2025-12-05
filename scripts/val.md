1. Parametri di Architettura (SwinIR Model)

Queste variabili definiscono la potenza del tuo modello SwinIR.

embed_dim (in src/architecture.py):

Cosa è: La larghezza interna dei blocchi Transformer. Definisce la dimensione dei vettori di feature elaborati.

Abbassarla comporta: Rende il modello più piccolo (meno parametri) e riduce il consumo di VRAM, ma diminuisce la capacità di apprendimento e la qualità finale.

depths (in src/architecture.py):

Cosa è: La profondità (numero di blocchi Transformer) del modello, tipicamente espressa come una lista per i diversi stadi.

Abbassarla comporta: Rende il modello meno profondo (meno parametri) e riduce la sua capacità di elaborare feature complesse in sequenza, con conseguente calo della qualità.

window_size (in src/architecture.py):

Cosa è: La dimensione della "finestra" locale utilizzata dal meccanismo di Self-Attention all'interno di SwinIR.

Abbassarla comporta: Rende il campo visivo locale del Transformer più piccolo, limitando la sua capacità di correlare informazioni su distanze brevi. (Generalmente non si modifica da 8 in questa architettura).

2. Parametri di Training (scripts/Modello_supporto.py)

Queste variabili controllano la stabilità e la velocità del processo di apprendimento.

BATCH_SIZE (nel worker):

Cosa è: Il numero di patch elaborate in parallelo dalla GPU prima di accumulare i gradienti.

Abbassarla comporta: Riduce drasticamente il consumo di VRAM, ma rallenta il training e rende l'addestramento meno efficiente nell'uso della GPU.

ACCUM_STEPS (nel worker):

Cosa è: Il numero di passi in avanti e indietro eseguiti prima di aggiornare i pesi (calcola l'Effective Batch Size come BATCH_SIZE * ACCUM_STEPS).

Abbassarla comporta: Riduce l'Effective Batch Size, rendendo il gradiente di training più "rumoroso" e meno stabile, potendo peggiorare la qualità e la convergenza.

LR (Learning Rate, nel worker):

Cosa è: Il tasso di apprendimento, ovvero la dimensione del passo con cui i pesi del modello vengono aggiornati.

Abbassarla comporta: Il modello impara più lentamente (richiede più epoche), ma è essenziale per la stabilità di un Transformer e per raggiungere un punto di qualità più alto.

TOTAL_EPOCHS (nel worker):

Cosa è: Il numero totale di cicli completi di training sull'intero dataset.

Abbassarla comporta: Interrompe il training prima, portando a un modello non completamente addestrato (underfitting) e a una qualità finale inferiore.

3. Pesi della Loss Function (src/losses.py)

Queste variabili determinano cosa è più importante per il modello da minimizzare.

l1_w (Peso Charbonnier Loss):

Cosa è: Il peso dato all'errore di base basato sul pixel (Charbonnier Loss).

Abbassarla comporta: Rende l'errore basato sul pixel meno importante. Il modello potrebbe ignorare piccole deviazioni di luminosità in favore di una migliore coerenza strutturale.

perceptual_w (Peso Perceptual Loss):

Cosa è: Il peso dato alla Loss sulla coerenza visiva, calcolata nello spazio delle feature VGG.

Abbassarla comporta: Il modello si preoccupa meno che l'immagine sembri realistica. Rischia di generare output con texture piatte o non coerenti.

astro_w (Peso Astro Loss):

Cosa è: Il peso dato alla Loss specifica che pondera l'errore sulle regioni luminose (stelle e centri di nebulose).

Abbassarla comporta: Il modello presta meno attenzione alla precisione delle stelle, aumentando il rischio di aloni o artefatti intorno ai punti più luminosi.

4. Variabili di Preparazione Dati (Dataset)

Queste influenzano la composizione del dataset.

HR_SIZE (in Dataset_step3...py):

Cosa è: La dimensione in pixel della patch ad Alta Risoluzione (HR) estratta da Hubble (e del tuo target di output).

Abbassarla comporta: Rende le patch HR più piccole. Rallenta il training di poco ma riduce il contesto spaziale per la HR.

STRIDE (in Dataset_step3...py):

Cosa è: Il passo di scorrimento delle patch, che determina quanto si sovrappongono.

Abbassarla comporta: Aumenta la sovrapposizione tra le patch, generando più campioni per l'addestramento.

LOWER_PERCENTILE (in Dataset_step4...py):

Cosa è: Il percentile inferiore usato per il clipping e lo stretch logaritmico durante la normalizzazione.

Abbassarla comporta: Il modello ignora meno il rumore di fondo. (Se scendi troppo, rischia di includere rumore).

TRAIN_RATIO (in scripts/Modello_2.py):

Cosa è: La percentuale del dataset totale riservata all'addestramento (il resto è per la validazione).

Abbassarla comporta: Riduce la dimensione del set di training, potenzialmente peggiorando la capacità di generalizzazione del modello.