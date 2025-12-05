import torch
import torch.nn as nn
import torch.nn.functional as F
from .env_setup import import_external_archs

# ==========================================================
# 1. RECUPERO ARCHITETTURE ESTERNE
# ==========================================================
# Importiamo solo RRDBNet, ignoriamo HAT
RRDBNet, _ = import_external_archs()

# ==========================================================
# 2. UTILITY LAYERS
# ==========================================================
class AntiCheckerboardLayer(nn.Module):
    def __init__(self, mode='balanced'):
        super().__init__()
        if mode == 'strong':
            k, p, s = 7, 3, 1600.0
            bk = [[1,6,15,20,15,6,1],[6,36,90,120,90,36,6],[15,90,225,300,225,90,15],
                  [20,120,300,400,300,120,20],[15,90,225,300,225,90,15],[6,36,90,120,90,36,6],[1,6,15,20,15,6,1]]
        elif mode == 'balanced':
            k, p, s = 5, 2, 256.0
            bk = [[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]]
        else:
            k, p, s = 3, 1, 16.0
            bk = [[1,2,1],[2,4,2],[1,2,1]]

        kernel = torch.tensor(bk, dtype=torch.float32) / s
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.register_buffer('weight', kernel)
        self.padding = p

    def forward(self, x):
        return F.conv2d(x, self.weight.expand(x.shape[1], -1, -1, -1), 
                        padding=self.padding, groups=x.shape[1])

# ==========================================================
# 3. MODELLO PURO RRDBNet (ESRGAN Style)
# ==========================================================
class HybridSuperResolutionModel(nn.Module):
    def __init__(self, output_size=512, smoothing='balanced', device='cpu'):
        super().__init__()
        self.output_size = output_size
        
        if RRDBNet is None:
            raise ImportError("❌ ERRORE: BasicSR (RRDBNet) non trovato.")
        
        print("🏗️  Model Arch: Pure RRDBNet (ESRGAN Style) x4")
        
        # -------------------------
        # STAGE 1: RRDBNet (Full 4x Upscale)
        # -------------------------
        # Configuriamo per 4x diretto (128 -> 512)
        # num_block=23 è lo standard ESRGAN (Qualità Alta)
        # Se hai ancora problemi di memoria, scendi a num_block=16
        self.stage1 = RRDBNet(
            num_in_ch=1, 
            num_out_ch=1, 
            num_feat=64,      # Standard ESRGAN
            num_block=23,     # Standard ESRGAN
            num_grow_ch=32,   
            scale=4           # <--- FONDAMENTALE: Fa tutto l'upscale qui
        )
        
        # Disabilitiamo Stage 2 (HAT)
        self.has_stage2 = False
        self.stage2 = nn.Identity()

        # Anti-Checkerboard opzionale
        if smoothing != 'none':
            self.sf = AntiCheckerboardLayer('light')
        else:
            self.sf = nn.Identity()

    def forward(self, x):
        # Stage 1 fa tutto il lavoro (128px -> 512px)
        x = self.stage1(x)
        
        # Resize di sicurezza (solo se output size diverso da 512)
        if x.shape[-1] != self.output_size:
            x = F.interpolate(x, size=(self.output_size, self.output_size), 
                              mode='bicubic', align_corners=False, antialias=True)
            
        return self.sf(x)
