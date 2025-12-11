import torch
import torch.nn as nn
import torchvision.models as models

class VGGLoss(nn.Module):
    def __init__(self, feature_layer=35, use_bn=False, device='cpu'):
        super(VGGLoss, self).__init__()
        # Carica VGG19 pre-allenata su ImageNet
        # Usiamo weights='DEFAULT' per la versione più aggiornata
        try:
            vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        except:
            # Fallback per versioni vecchie di torch
            vgg = models.vgg19(pretrained=True)
            
        if use_bn:
            vgg = models.vgg19_bn(pretrained=True)
            
        # Prendiamo le feature fino al layer indicato (35 è tipico per texture profonde)
        self.features = nn.Sequential(*list(vgg.features.children())[:feature_layer]).eval()
        
        # Congeliamo i pesi (la VGG non deve imparare, solo giudicare)
        for param in self.features.parameters():
            param.requires_grad = False
            
        # Normalizzazione standard ImageNet (necessaria per VGG)
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        # 1. Gestione input Grayscale (B, 1, H, W) -> RGB (B, 3, H, W)
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
            
        # 2. Normalizzazione
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        # 3. Estrazione Features
        pred_feat = self.features(pred)
        target_feat = self.features(target)
        
        # 4. Calcolo Loss (L1 sulle feature è più stabile di MSE)
        return nn.functional.l1_loss(pred_feat, target_feat)

class TrainStarLoss(nn.Module):
    def __init__(self, vgg_weight=0.05):
        super().__init__()
        # Loss Pixel-wise (L1)
        self.l1 = nn.L1Loss(reduction='none')
        
        # Loss Percettiva (VGG)
        # Inizializziamo la VGG. Sarà spostata su GPU automaticamente dal training loop.
        self.vgg_loss = VGGLoss()
        self.vgg_weight = vgg_weight
        
    def forward(self, pred, target):
        # --- A. PIXEL LOSS (Pesata per dettagli) ---
        diff = self.l1(pred, target)
        
        # Maschere di importanza (dal passaggio precedente)
        structure_mask = target > 0.005  
        weight_map = torch.ones_like(diff)
        weight_map[structure_mask] = 20.0  
        bright_mask = target > 0.1
        weight_map[bright_mask] = 50.0
        
        loss_pixel = torch.mean(diff * weight_map)
        
        # --- B. VGG LOSS (Percettiva) ---
        loss_perceptual = self.vgg_loss(pred, target)
        
        # --- C. TOTALE ---
        # Sommiamo le due loss. 
        # loss_pixel garantisce la correttezza dei colori/intensità.
        # loss_perceptual * 0.05 spinge per creare pattern realistici.
        total_loss = loss_pixel + (self.vgg_weight * loss_perceptual)
        
        return total_loss, {'total': total_loss, 'pixel': loss_pixel, 'vgg': loss_perceptual}