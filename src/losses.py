import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CharbonnierLoss(nn.Module):
    """L1 Loss robusta (migliore per Super-Resolution rispetto a L1 standard)."""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps)
        return torch.mean(loss)

class CombinedLoss(nn.Module):
    def __init__(self, l1_w=1.0, perceptual_w=0.05, astro_w=0.2):
        super().__init__()
        # Pesi ricalibrati: Meno peso su astro/perceptual all'inizio per favorire la convergenza
        self.weights = (l1_w, perceptual_w, astro_w)
        
        # Uso Charbonnier invece di L1 standard
        self.char = CharbonnierLoss()
        
        # VGG per Perceptual Loss (Feature Extractor)
        # Usiamo i layer fino al 34 (relu5_4) o 18 (relu3_4)
        vgg19 = models.vgg19(weights='DEFAULT').features
        self.vgg = nn.Sequential(*list(vgg19.children())[:18]).eval()
        
        for p in self.vgg.parameters(): 
            p.requires_grad = False
            
        # Normalizzazione ImageNet per VGG
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, pred, target):
        # 1. Pixel Loss (Charbonnier) - Fondamentale per il fondo nero
        char_loss = self.char(pred, target)
        
        # 2. Astro Loss (Penalizza errori sulle stelle)
        # Usiamo Charbonnier pesato invece di L1 raw
        diff = torch.abs(pred - target)
        # Il peso (1 + 10*target) penalizza di più le aree luminose (stelle)
        weight_map = 1.0 + 10.0 * target
        astro_loss = torch.mean(torch.sqrt(diff * diff + 1e-6) * weight_map)
        
        # 3. Perceptual Loss (VGG)
        # IMPORTANTE: Clampiamo a [0,1] per evitare che VGG impazzisca con valori fuori range
        pred_clamped = pred.clamp(0, 1)
        target_clamped = target.clamp(0, 1)
        
        # Convertiamo 1 canale -> 3 canali per VGG
        pr = (pred_clamped.repeat(1,3,1,1) - self.mean) / self.std
        tr = (target_clamped.repeat(1,3,1,1) - self.mean) / self.std
        
        perc_loss = F.l1_loss(self.vgg(pr), self.vgg(tr))
        
        # Somma Ponderata
        total_loss = (self.weights[0] * char_loss + 
                      self.weights[1] * perc_loss + 
                      self.weights[2] * astro_loss)
        
        return total_loss, {
            'char': char_loss, 
            'astro': astro_loss, 
            'perceptual': perc_loss
        }