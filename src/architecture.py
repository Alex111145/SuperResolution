import torch
import torch.nn as nn
import torch.nn.functional as F
from .env_setup import import_external_archs

# Recuperiamo SwinIR
_, _, SwinIR_Arch = import_external_archs()

class TrainHybridModel(nn.Module):
    def __init__(self, output_size=512, smoothing=None, device='cpu'):
        super().__init__()
        self.output_size = output_size
        
        if SwinIR_Arch is None:
            raise ImportError("❌ ERRORE CRITICO: SwinIR non trovato.")
        
        print("🏗️  Model Arch: SwinIR-Standard (Balanced) x4 [TRAIN MODE]")
        
        # CONFIGURAZIONE SWINIR BILANCIATA (PER GPU < 24GB)
        self.model = SwinIR_Arch(
            upscale=4,             
            in_chans=1,            
            img_size=128,          
            window_size=8,         
            img_range=1.,          
            
            # Parametri Ottimizzati per RTX/GTX
            depths=[8, 8, 8, 8, 8, 8,8,8],   # Profondità invariata (ottima qualità)
            embed_dim=360,                # RIDOTTO da 180 a 96 (Risparmia molta VRAM)
            num_heads=[8, 8, 8, 8, 8, 8, 8, 8],
            
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='1conv'
        )

    def forward(self, x):
        x = self.model(x)
        
        if x.shape[-1] != self.output_size:
            x = F.interpolate(x, size=(self.output_size, self.output_size), 
                              mode='bicubic', align_corners=False, antialias=True)
        return x