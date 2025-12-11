1. Creazione e Attivazione Virtual Environment
Questo isola le librerie del progetto dal sistema.

Bash

# Crea l'ambiente virtuale chiamato 'venv'
python3 -m venv venv

# Attiva l'ambiente
source venv/bin/activate
2. Installazione nvtop
nvtop è un monitor di sistema per GPU (simile a htop). Richiede privilegi di root/sudo per l'installazione.

Bash

sudo apt update
sudo apt install -y nvtop
Per usarlo, digita semplicemente nvtop nel terminale.

3. Installazione Dipendenze (requirements.txt)
Il file requirements.txt include già l'indice specifico per PyTorch con CUDA (cu118) e tensorboard.

Bash

# Aggiorna pip per sicurezza
pip install --upgrade pip

# Installa tutto dal file
pip install -r requirements.txt
4. Installazione e Avvio TensorBoard
TensorBoard viene installato automaticamente col comando sopra (è presente nel requirements.txt).

Per avviarlo, devi puntare alla cartella dove lo script train_core.py salva i log. Analizzando il codice, i log vengono salvati in outputs/{TARGET}_DDP/tensorboard.

Per monitorare tutti i training contemporaneamente:

Bash

# Avvia TensorBoard sulla cartella outputs
tensorboard --logdir=./outputs --port 6006 --bind_all
