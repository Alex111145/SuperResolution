dopo aver fatto dataset 4 vedere quante patch sono scure e nel caso rifarle aumentando MIN_COVERAGE = 0.40  e diminuendo lo stride 


fa 1000 file ogni 1 min 30 sec

unzip M33_dataset_fits.zip

mv pair* data/M33/6_patches_final/


python Modello_1.py 
python Modello_2.py 
python Modello_3.py 
python Modello_4.py --target M33 --exclude 2 7
python Modello_5.py 

tensorboard --logdir=outputs --port=6006 --bind_all
