import sys
import torch
import torchvision.transforms.functional as TF_functional

# Fix per torchvision
sys.modules['torchvision.transforms.functional_tensor'] = TF_functional

try:
    from models.hybridmodels import HybridHATRealESRGAN
    from models.discriminator import UNetDiscriminatorSN
except ImportError as e:
    print(f"❌ Errore di importazione: {e}")
    sys.exit(1)

# --- CONFIGURAZIONE ---
MODEL_CONFIG = {
    "img_size": 128,
    "in_chans": 1,
    "embed_dim": 180,              
    "depths": (6, 6, 6, 6, 6, 6),  
    "num_heads": (6, 6, 6, 6, 6, 6),
    "window_size": 8,               
    "upscale": 4,
    "num_rrdb": 23,                
    "num_feat": 64,
    "num_grow_ch": 32
}
# ----------------------

def get_total_params(model):
    return sum(p.numel() for p in model.parameters())

def main():
    print("Calcolo parametri in corso...\n")
    
    try:
        # 1. Inizializzazione modelli
        net_g = HybridHATRealESRGAN(**MODEL_CONFIG)
        net_d = UNetDiscriminatorSN(num_in_ch=1, num_feat=64)
        
        # 2. Calcolo totali
        total_g = get_total_params(net_g)
        total_d = get_total_params(net_d)
        grand_total = total_g + total_d

        # 3. Stampa Riepilogo
        print("=" * 60)
        print(" CONTEGGIO TOTALE PARAMETRI")
        print("-" * 60)
        
        print(f" Hybrid Generator (HAT):  {total_g/1e6:6.2f} M  ({total_g:,})")
        print(f" Discriminator (UNet):    {total_d/1e6:6.2f} M  ({total_d:,})")
        print("-" * 60)
        print(f" TOTALE COMPLESSIVO:      {grand_total/1e6:6.2f} M  ({grand_total:,})")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Errore durante il calcolo: {e}")

if __name__ == "__main__":
    main()