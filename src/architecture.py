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
        
        print("🏗️  Model Arch: SwinIR-ULTRA (Astronomical High-Fidelity) [TRAIN MODE]")
        
        # CONFIGURAZIONE "ULTRA" (High Quality / High VRAM usage)
        # Ottimizzata per massimizzare la resa su texture stellari e nebulose
        self.model = SwinIR_Arch(
            upscale=4,             
            in_chans=1,            
            img_size=128,          # Dimensione patch di training (non influenza l'inferenza)
            window_size=8,         
            img_range=1.,          
            
            # --- PARAMETRI POTENZIATI ---
            # Più profondo (9 blocchi) e con MLP Ratio standard (4)
            depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], 
            
            # 192 è il "sweet spot" che permette mlp_ratio=4 su GPU da 12-16GB+
            # Se hai una 3090/4090 (24GB), puoi tentare embed_dim=210 o 240.
            embed_dim=192,         
            
            num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6], # Deve dividere embed_dim (192/6 = 32)
            
            mlp_ratio=4,           # STANDARD ALTA QUALITÀ (era 2) - Fondamentale per la non-linearità
            upsampler='pixelshuffle',
            resi_connection='1conv'
        )

    def forward(self, x):
        x = self.model(x)
        
        # Interpolazione di sicurezza nel caso l'output non sia esattamente 512
        if x.shape[-1] != self.output_size:
            x = F.interpolate(x, size=(self.output_size, self.output_size), 
                              mode='bicubic', align_corners=False, antialias=True)
        return x