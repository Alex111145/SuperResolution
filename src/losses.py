import torch
import torch.nn as nn

class TrainStarLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        
    def forward(self, pred, target):
        # 1. Loss Strutturale (Base): Fondamentale per SSIM/FSIM alti
        loss_content = self.l1(pred, target)
        
        # 2. Loss Pesata (Stelle): Peso ridotto e soglia più alta
        diff = torch.abs(pred - target)
        
        # Soglia alzata a 0.1 per prendere solo le stelle vere (non il rumore di fondo)
        # Peso ridotto a 20.0 (invece di 500.0) per stabilità
        stars_mask = target > 0.10  
        
        weight_map = torch.ones_like(diff)
        weight_map[stars_mask] = 20.0  
        
        loss_weighted = torch.mean(diff * weight_map)
        
        # Mix: 70% Contenuto (Struttura) + 30% Stelle (Dettaglio picchi)
        total_loss = 0.7 * loss_content + 0.3 * loss_weighted
        
        return total_loss, {'total': total_loss}