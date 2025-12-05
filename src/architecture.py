import torch
import torch.nn as nn
import torch.nn.functional as F
from .env_setup import import_external_archs

# Importiamo SwinIR (terzo valore restituito)
_, _, SwinIR_Arch = import_external_archs()

class HybridSuperResolutionModel(nn.Module):
    def __init__(self, output_size=512, smoothing='balanced', device='cpu'):
        super().__init__()
        self.output_size = output_size
        
        if SwinIR_Arch is None:
            raise ImportError("❌ ERRORE CRITICO: SwinIR non trovato in BasicSR.")
        
        print("🏗️  Model Arch: SwinIR-Light (Vision Transformer) x4")
        
        # CONFIGURAZIONE SWINIR LIGHTWEIGHT
        # Questa è la configurazione ufficiale "lightweight" per SR
        # Parametri ridotti rispetto al modello standard per girare su GPU singola
        self.model = SwinIR_Arch(
            upscale=4,             # Upscaling diretto 4x (128->512)
            in_chans=1,            # 1 canale (Bianco e Nero)
            img_size=128,          # Dimensione patch input (training)
            window_size=8,
            img_range=1.,
            
            # Configurazione "Light":
            depths=[6, 6, 6, 6],   # Profondità ridotta
            embed_dim=60,          # Embedding ridotto
            num_heads=[6, 6, 6, 6],
            
            mlp_ratio=2,
            upsampler='pixelshuffle', # PixelShuffle è standard per SwinIR
            resi_connection='1conv'
        )

    def forward(self, x):
        # SwinIR gestisce tutto internamente
        x = self.model(x)
        
        # Resize di sicurezza se l'output non fosse esattamente 512
        if x.shape[-1] != self.output_size:
            x = F.interpolate(x, size=(self.output_size, self.output_size), 
                              mode='bicubic', align_corners=False, antialias=True)
        return x
