import torch
import torch.nn.functional as F
from math import exp
import sys

# Tenta di importare piq per FSIM
try:
    from piq import fsim
    PIQ_AVAILABLE = True
except ImportError:
    PIQ_AVAILABLE = False
    print("⚠️  Libreria 'piq' non trovata. FSIM sarà 0. Installa con: pip install piq")

def ssim_torch(img1, img2, window_size=11):
    """Calcolo SSIM interno (Fallback se piq non va)."""
    c = img1.size(1)
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*1.5**2)) for x in range(window_size)])
    win = (gauss/gauss.sum()).unsqueeze(1).mm((gauss/gauss.sum()).unsqueeze(0)).unsqueeze(0).unsqueeze(0).expand(c,1,window_size,window_size).type_as(img1)
    
    mu1, mu2 = F.conv2d(img1, win, groups=c), F.conv2d(img2, win, groups=c)
    sigma1_sq = F.conv2d(img1*img1, win, groups=c) - mu1.pow(2)
    sigma2_sq = F.conv2d(img2*img2, win, groups=c) - mu2.pow(2)
    sigma12 = F.conv2d(img1*img2, win, groups=c) - mu1*mu2
    
    return (((2*mu1*mu2 + 0.01**2)*(2*sigma12 + 0.03**2))/((mu1.pow(2) + mu2.pow(2) + 0.01**2)*(sigma1_sq + sigma2_sq + 0.03**2))).mean()

class Metrics:
    def __init__(self): 
        self.reset()
        
    def reset(self): 
        self.ssim = 0.0
        self.fsim = 0.0
        self.count = 0
        
    def update(self, p, t):
        # Assicura range [0, 1] e stacca dai grafi di calcolo
        p = p.detach().clamp(0, 1)
        t = t.detach().clamp(0, 1)
        
        batch_size = p.size(0)

        # --- 1. Calcolo SSIM ---
        self.ssim += ssim_torch(p, t).item() * batch_size
        
        # --- 2. Calcolo FSIM ---
        if PIQ_AVAILABLE:
            # Check dimensioni: piq vuole (N, C, H, W)
            if p.ndim == 3: p = p.unsqueeze(0)
            if t.ndim == 3: t = t.unsqueeze(0)
            
            # Rimosso il try/except per vedere l'errore reale
            val_fsim = fsim(p, t, data_range=1.0, reduction='mean')
            self.fsim += val_fsim.item() * batch_size
        
        self.count += batch_size
        
    def compute(self): 
        if self.count == 0:
            return {'ssim': 0, 'fsim': 0}
            
        return {
            'ssim': self.ssim / self.count,
            'fsim': self.fsim / self.count
        }

# Alias per compatibilità
class TrainMetrics(Metrics):
    pass