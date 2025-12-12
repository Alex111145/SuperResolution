import torch
import torch.nn as nn
import torchvision.models as models

class VGGLoss(nn.Module):
    def __init__(self, feature_layer=35, use_bn=False, device='cpu'):
        super(VGGLoss, self).__init__()
        try:
            vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        except:
            vgg = models.vgg19(pretrained=True)
            
        if use_bn: vgg = models.vgg19_bn(pretrained=True)
            
        self.features = nn.Sequential(*list(vgg.features.children())[:feature_layer]).eval()
        for param in self.features.parameters():
            param.requires_grad = False
            
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        # 0. SAFETY FIRST: Rimuove NaN e clamp valori estremi
        pred = torch.nan_to_num(pred)
        pred = pred.clamp(0, 1) # VGG vuole input normali
        target = target.clamp(0, 1)

        # 1. Grayscale -> RGB
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
            
        # 2. Normalizzazione ImageNet
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        # 3. Features & Loss
        pred_feat = self.features(pred)
        target_feat = self.features(target)
        
        return nn.functional.l1_loss(pred_feat, target_feat)

class TrainStarLoss(nn.Module):
    def __init__(self, vgg_weight=0.05):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='none')
        self.vgg_loss = VGGLoss()
        self.vgg_weight = vgg_weight
        
    def forward(self, pred, target):
        # Protezione valori
        pred = torch.nan_to_num(pred)
        
        # --- A. PIXEL LOSS ---
        diff = self.l1(pred, target)
        
        structure_mask = target > 0.005  
        weight_map = torch.ones_like(diff)
        weight_map[structure_mask] = 20.0  
        bright_mask = target > 0.1
        weight_map[bright_mask] = 50.0
        
        loss_pixel = torch.mean(diff * weight_map)
        
        # --- B. VGG LOSS ---
        # Il clamp qui è critico per la stabilità
        loss_perceptual = self.vgg_loss(pred, target)
        
        total_loss = loss_pixel + (self.vgg_weight * loss_perceptual)
        
        return total_loss, {'total': total_loss, 'pixel': loss_pixel, 'vgg': loss_perceptual}