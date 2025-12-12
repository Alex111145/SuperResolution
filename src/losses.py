import torch
import torch.nn as nn
import torchvision.models as models

class VGGLoss(nn.Module):
    def __init__(self, feature_layer=35, use_bn=False):
        super(VGGLoss, self).__init__()
        
        # Caricamento robusto VGG
        try:
            vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        except:
            vgg = models.vgg19(pretrained=True)
            
        if use_bn:
            vgg = models.vgg19_bn(pretrained=True)
            
        # Layer 35 (relu5_4) è profondo -> Ottimo per texture e struttura
        self.features = nn.Sequential(*list(vgg.features.children())[:feature_layer]).eval()
        
        # Congela i pesi (non addestriamo la VGG)
        for param in self.features.parameters():
            param.requires_grad = False
            
        # Normalizzazione VGG (ImageNet stats)
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        # SwinIR esce con 1 canale, VGG ne vuole 3
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
            
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        pred_feat = self.features(pred)
        target_feat = self.features(target)
        
        return nn.functional.l1_loss(pred_feat, target_feat)

class TrainStarLoss(nn.Module):
    def __init__(self, vgg_weight=1.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.vgg_loss = VGGLoss()
        self.vgg_weight = vgg_weight
        
    def forward(self, pred, target):
        # 1. PIXEL LOSS (Struttura Base)
        # Usiamo L1 semplice. SwinIR è potente, non serve complicare con pesi manuali sulle stelle
        # se usiamo la VGG, perché la VGG "vede" già le stelle come feature salienti.
        loss_pixel = self.l1(pred, target)
        
        # 2. PERCEPTUAL LOSS (Nitidezza e Texture)
        loss_perceptual = self.vgg_loss(pred, target)
        
        # 3. TOTALE
        # Pixel Loss (1.0) + VGG (1.0)
        # Questo bilanciamento 1:1 è aggressivo sulla nitidezza.
        total_loss = loss_pixel + (self.vgg_weight * loss_perceptual)
        
        return total_loss, {'total': total_loss, 'pixel': loss_pixel, 'vgg': loss_perceptual}