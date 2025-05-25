import torch
import torch.nn as nn
from models.vision.encoder import MambaVision
from models.text.encoder import MPNet
from models.fusion import CrossAttention, CAMF

class MVP(nn.Module):
    def __init__(self, fuse_dim, embd_dim, Nclass, use_vae=False, if_cls=True):
        super().__init__()
        self.vision_encoder = MambaVision()
        self.text_encoder = MPNet()
        self.cls = if_cls
        
        self.unifier_1 = nn.Linear(640, 768).cuda() # 1280 -> 768
        self.unifier_2 = nn.Linear(384, 768).cuda()
        
        self.use_vae = use_vae
        
        if not use_vae:
            self.fuser = CrossAttention(768, fuse_dim).cuda()
        else:
            self.fuser = CAMF(768, embd_dim, Nclass, if_cls).cuda()
            
        self.alpha = nn.Parameter(torch.randn(1)).cuda()
        self.mlp = nn.Sequential(
            nn.Linear(fuse_dim, embd_dim),
            nn.ReLU(),
            nn.Linear(embd_dim, Nclass)
        ).cuda()

    def forward(self, pixels, tokens):
        image_features = self.vision_encoder(pixels)
        text_features = self.text_encoder(tokens, None)
        
        image_features = self.unifier_1(image_features)
        text_features = self.unifier_2(text_features)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
        if not self.use_vae:
            attn_a = self.fuser(image_features, text_features)
            attn_b = self.fuser(text_features, image_features)
            
            fusion_features = self.alpha * attn_a + (1 - self.alpha) * attn_b
            if not self.cls:
                return fusion_features
        else:
            T_recon, T, P_recon, P, mu_T, logvar_T, mu_P, logvar_P, fusion_features = self.fuser(image_features, text_features)
            if not self.cls:
                return T_recon, T, P_recon, P, mu_T, logvar_T, mu_P, logvar_P, fusion_features
            
        logits = self.mlp(fusion_features)
        return image_features, text_features, logits