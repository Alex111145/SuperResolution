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
    def __init__(self, l1_w=1.0, perceptual_w=0.0, astro_w=0.1): 
        super().__init__()
        # TUPLA DEI PESI MODIFICABILE ESTERNAMENTE
        self.weights = (l1_w, perceptual_w, astro_w)
        
        # Loss 1: Ricostruzione (Geometria)
        self.char = CharbonnierLoss()
        
        # Loss 2: Perceptual (VGG19)
        vgg19 = models.vgg19(weights='DEFAULT').features
        self.vgg = nn.Sequential(*list(vgg19.children())[:18]).eval()
        for p in self.vgg.parameters(): 
            p.requires_grad = False
            
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, pred, target):
        # 1. Charbonnier (Loss Principale)
        char_loss = self.char(pred, target)
        
        # 2. Astro Loss (Enfasi su stelle/parti chiare)
        diff = torch.abs(pred - target)
        # Pesa di più le zone dove il target è luminoso (stelle)
        weight_map = 1.0 + 5.0 * target 
        astro_loss = torch.mean(torch.sqrt(diff * diff + 1e-6) * weight_map)
        
        # 3. Perceptual Loss (Opzionale, attivata dopo)
        if self.weights[1] > 0:
            pred_c = pred.clamp(0, 1).repeat(1,3,1,1)
            target_c = target.clamp(0, 1).repeat(1,3,1,1)
            
            pr = (pred_c - self.mean) / self.std
            tr = (target_c - self.mean) / self.std
            perc_loss = F.l1_loss(self.vgg(pr), self.vgg(tr))
        else:
            perc_loss = torch.tensor(0.0, device=pred.device)
        
        # CALCOLO TOTALE PONDERATO
        total_loss = (self.weights[0] * char_loss + 
                      self.weights[1] * perc_loss + 
                      self.weights[2] * astro_loss)
        
        return total_loss, {
            'char': char_loss, 
            'astro': astro_loss, 
            'perceptual': perc_loss
        }