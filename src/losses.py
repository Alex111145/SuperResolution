import torch
import torch.nn as nn

class TrainStarLoss(nn.Module):
    def __init__(self, l1_w=1.0):
        super().__init__()
        
    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        
        weight_map = torch.ones_like(diff)
        
        # Peso per le stelle (zone luminose)
        stars_mask = target > 0.02  
        weight_map[stars_mask] = 500.0  
        
        loss = torch.mean(diff * weight_map)
        
        return loss, {'total': loss}