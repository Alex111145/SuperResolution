import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class VGGLoss(nn.Module):
 
    def __init__(self, feature_layer=35, use_input_norm=True, use_range_norm=False):
        super(VGGLoss, self).__init__()

        vgg19 = models.vgg19(pretrained=True)
        self.features = nn.Sequential(*list(vgg19.features.children())[:feature_layer+1])
        
 
        for k, v in self.features.named_parameters():
            v.requires_grad = False
        
    
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self.use_input_norm = use_input_norm
        self.use_range_norm = use_range_norm

    def forward(self, x, y):
      
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.shape[1] == 1:
            y = y.repeat(1, 3, 1, 1)

  
        if self.use_input_norm:
            x = (x - self.mean) / self.std
            y = (y - self.mean) / self.std
            
        x_feat = self.features(x)
        y_feat = self.features(y) 
        
        return F.l1_loss(x_feat, y_feat.detach())

class CharbonnierLoss(nn.Module):
  
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss
