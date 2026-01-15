Con le modifiche appena apportate al codice di train_hat.py, il file CSV di log (train_log.csv) conterrà ora le seguenti 8 colonne:

Epoch: Il numero dell'epoca corrente.

G_Total: La perdita (loss) totale del Generatore (somma pesata di Pixel, Perceptual e Adversarial).

L1: Il valore della Pixel Loss pura (differenza media assoluta tra i pixel generati e quelli reali).

G_Adv: La componente avversaria della perdita del Generatore (quanto il generatore è riuscito a ingannare il discriminatore).

D_Total: La perdita totale del Discriminatore (quanto bene il discriminatore distingue tra vero e falso).

LR: Il Learning Rate corrente del Generatore.

PSNR (Nuovo): Il Peak Signal-to-Noise Ratio medio dell'epoca. Indica la qualità della ricostruzione dell'immagine (valori più alti sono migliori).

SSIM (Nuovo): Lo Structural Similarity Index medio dell'epoca. Misura la similarità strutturale tra l'immagine generata e quella originale (valori più vicini a 1 sono migliori).




============================================================
 CONTEGGIO TOTALE PARAMETRI
------------------------------------------------------------
 Hybrid Generator (HAT):   42.54 M  (42,536,738)
 Discriminator (UNet):      4.38 M  (4,375,745)
------------------------------------------------------------
 TOTALE COMPLESSIVO:       46.91 M  (46,912,483)
============================================================
