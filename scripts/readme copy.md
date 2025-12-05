2) cd SuperResolution
2) source venv/bin/activate
3) pip install -r requirements.txt
4) nvtop (per capire che gpu viene usata)
5) pip install tensorboard
6)tensorboard --logdir=outputs --port=6006 --bind_all

PROBLEMI NON FUNZIONA LA DATASET_1
NOTA: SE FAI ANDARE SU WINDOWS MI DA ERRORE COMUNQUE RISOLVI IO FACCIO ALTRO 

DATASET_2 ? 
DATASET_3 OK STRIDE 50 700 PATCH, se dimezzo stride in teoria e x4
DATASET 4 OK 
MODELLO 2 OK 
MODELLO 3 CAMBIATO OK 
SUPPORTO CAMBIATO OK 




problema di out of memory batch o qualita di hat hat s super ecc
provare con 3000 coppie e 300 epoche 