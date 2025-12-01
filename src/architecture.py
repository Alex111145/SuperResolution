import torch
import torch.nn as nn
import torch.nn.functional as F
from .env_setup import import_external_archs

RRDBNet, HAT_Arch = import_external_archs()

class AntiCheckerboardLayer(nn.Module):
    def __init__(self, mode='balanced'):
        super().__init__()
        if mode == 'strong':
            k, p, s = 7, 3, 1600.0
            bk = [[1,6,15,20,15,6,1],[6,36,90,120,90,36,6],[15,90,225,300,225,90,15],
                  [20,120,300,400,300,120,20],[15,90,225,300,225,90,15],[6,36,90,120,90,36,6],[1,6,15,20,15,6,1]]
        elif mode == 'balanced':
            k, p, s = 5, 2, 256.0
            bk = [[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]]
        else:
            k, p, s = 3, 1, 16.0
            bk = [[1,2,1],[2,4,2],[1,2,1]]

        kernel = torch.tensor(bk, dtype=torch.float32) / s
        self.conv = nn.Conv2d(1, 1, k, padding=p, bias=False)
        with torch.no_grad(): self.conv.weight[0, 0] = kernel
        self.conv.weight.requires_grad = False

    def forward(self, x): return self.conv(x)

class HybridSuperResolutionModel(nn.Module):
    def __init__(self, output_size=512, smoothing='balanced', device='cuda'):
        super().__init__()
        self.output_size = output_size
        
        if RRDBNet is None: 
            raise ImportError("BasicSR mancante.")
        
        self.stage1 = RRDBNet(num_in_ch=1, num_out_ch=1, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        
        self.has_stage2 = False
        self.stage2 = nn.Identity()
        
        if HAT_Arch:
            try:
                self.stage2 = HAT_Arch(
                    img_size=64, 
                    patch_size=1, 
                    in_chans=1, 
                    embed_dim=120,
                    depths=[6, 6, 6, 6, 6, 6], 
                    num_heads=[6, 6, 6, 6, 6, 6], 
                    window_size=16,          
                    compress_ratio=3, 
                    squeeze_factor=30, 
                    conv_scale=0.01, 
                    overlap_ratio=0.5, 
                    mlp_ratio=2., 
                    qkv_bias=True, 
                    upscale=2, 
                    img_range=1., 
                    upsampler='pixelshuffle', 
                    resi_connection='1conv'
                )
                self.has_stage2 = True
                print("   ✅ HAT: Light-120 (Memory Safe)")
            except Exception as e: 
                print(f"⚠️ HAT Error: {e}. Solo Stage 1.")

        if smoothing != 'none':
            self.s1 = AntiCheckerboardLayer(smoothing)
            self.s2 = AntiCheckerboardLayer(smoothing)
            self.sf = AntiCheckerboardLayer('light')
        else:
            self.s1 = self.s2 = self.sf = nn.Identity()
            
        self.to(device)

    def forward(self, x):
        x = self.stage1(x)
        x = self.s1(x)
        if self.has_stage2: 
            x = self.stage2(x)
            x = self.s2(x)
        
        if x.shape[-1] != self.output_size:
            x = F.interpolate(x, size=(self.output_size, self.output_size), 
                              mode='bicubic', align_corners=False, antialias=True)
        return self.sf(x)