import torch
import torch.nn as nn
import torch.nn.functional as F
from .env_setup import import_external_archs

# Recuperiamo SwinIR (terzo valore restituito)
_, _, SwinIR_Arch = import_external_archs()

class HybridSuperResolutionModel(nn.Module):
    def __init__(self, output_size=512, smoothing=None, device='cpu'):
        super().__init__()
        self.output_size = output_size
        
        if SwinIR_Arch is None:
            raise ImportError("❌ ERRORE CRITICO: SwinIR non trovato. Installa 'basicsr' e controlla env_setup.py")
        
        print("🏗️  Model Arch: SwinIR-Light (Vision Transformer) x4")
        
        # CONFIGURAZIONE SWINIR LIGHTWEIGHT
        # Ottimizzata per GPU consumer (6-8GB VRAM) ma con qualità Transformer
        self.model = SwinIR_Arch(
            upscale=4,             # Upscaling 4x (128 -> 512)
            in_chans=1,            # 1 canale (Immagini BW Astronomiche)
            img_size=128,          # Dimensione patch di training
            window_size=8,         # Standard per SwinIR
            img_range=1.,          # Range input 0-1
            
            # Parametri "Light" (Depth e Width ridotti per stabilità VRAM)
            depths=[6, 6, 8, 6],   
            embed_dim=90,          
            num_heads=[6, 6, 6, 6],
            
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='1conv'
        )

    def forward(self, x):
        # SwinIR fa tutto internamente
        x = self.model(x)
        
        # Safety Check: Garantiamo output 512x512 esatto
        if x.shape[-1] != self.output_size:
            x = F.interpolate(x, size=(self.output_size, self.output_size), 
                              mode='bicubic', align_corners=False, antialias=True)
        return x
