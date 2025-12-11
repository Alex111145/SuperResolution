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
        
        # MODIFICATO: Ridotto da 500.0 a 100.0 per evitare gradienti esplosivi (NaN)
        weight_map[stars_mask] = 100.0  
        
        loss = torch.mean(diff * weight_map)
        
        return loss, {'total': loss}