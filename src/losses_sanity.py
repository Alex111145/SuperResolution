import torch
import torch.nn as nn
import torch.nn.functional as F

class SanityStarLoss(nn.Module):
    def __init__(self, l1_w=1.0, perceptual_w=0.0, astro_w=1.0):
        super().__init__()
        
    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        
        # --- MAPPA DI IMPORTANZA ---
        weight_map = torch.ones_like(diff)
        
        # Se il pixel target è luminoso (stella), l'errore pesa 500 volte di più
        # Questa logica è specifica per forzare l'overfitting sulle stelle nel Sanity Check
        stars_mask = target > 0.02  
        weight_map[stars_mask] = 100.0 
        
        loss = torch.mean(diff * weight_map)
        
        return loss, {'total': loss, 'weighted_l1': loss}