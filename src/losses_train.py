import torch
import torch.nn as nn

class TrainStarLoss(nn.Module):
    def __init__(self, l1_w=1.0):
        super().__init__()
        
    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        
        weight_map = torch.ones_like(diff)
        
        # Nel training reale, manteniamo un peso alto per le stelle
        # ma riduciamo leggermente l'estremismo rispetto al sanity check
        # per permettere al modello di imparare anche il fondo cielo (rumore).
        stars_mask = target > 0.02  
        weight_map[stars_mask] = 500.0  # 500x invece di 100x per stabilità
        
        loss = torch.mean(diff * weight_map)
        
        return loss, {'total': loss}