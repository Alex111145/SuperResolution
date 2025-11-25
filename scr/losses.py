import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CombinedLoss(nn.Module):
    def __init__(self, l1_w=1.0, perceptual_w=0.1, astro_w=1.0):
        super().__init__()
        self.weights = (l1_w, perceptual_w, astro_w)
        self.l1 = nn.L1Loss()
        self.vgg = models.vgg19(weights='DEFAULT').features[:18].eval()
        for p in self.vgg.parameters(): p.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, pred, target):
        l1 = self.l1(pred, target)
        astro = (F.l1_loss(pred, target, reduction='none') * (1.0 + 10.0 * target)).mean()
        
        pr, tr = (pred.repeat(1,3,1,1)-self.mean)/self.std, (target.repeat(1,3,1,1)-self.mean)/self.std
        perc = F.l1_loss(self.vgg(pr), self.vgg(tr))
        
        return self.weights[0]*l1 + self.weights[1]*perc + self.weights[2]*astro, \
               {'l1': l1, 'astro': astro, 'perceptual': perc}