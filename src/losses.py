import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps)
        return torch.mean(loss)

class CombinedLoss(nn.Module):
    def __init__(self, l1_w=1.0, perceptual_w=0.05, astro_w=0.05):
        super().__init__()
        self.weights = (l1_w, perceptual_w, astro_w)
        
        self.char = CharbonnierLoss()
        
        vgg19 = models.vgg19(weights='DEFAULT').features
        self.vgg = nn.Sequential(*list(vgg19.children())[:18]).eval()
        
        for p in self.vgg.parameters(): 
            p.requires_grad = False
            
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, pred, target):
        char_loss = self.char(pred, target)
        
        with torch.no_grad():
            l1_raw = F.l1_loss(pred, target)
        
        diff = torch.abs(pred - target)
        weight_map = 1.0 + 5.0 * target 
        astro_loss = torch.mean(torch.sqrt(diff * diff + 1e-6) * weight_map)
        
        pred_clamped = pred.clamp(0, 1)
        target_clamped = target.clamp(0, 1)
        
        pr = (pred_clamped.repeat(1,3,1,1) - self.mean) / self.std
        tr = (target_clamped.repeat(1,3,1,1) - self.mean) / self.std
        
        perc_loss = F.l1_loss(self.vgg(pr), self.vgg(tr))
        
        total_loss = (self.weights[0] * char_loss + 
                      self.weights[1] * perc_loss + 
                      self.weights[2] * astro_loss)
        
        return total_loss, {
            'char': char_loss, 
            'l1_raw': l1_raw,
            'astro': astro_loss, 
            'perceptual': perc_loss
        }