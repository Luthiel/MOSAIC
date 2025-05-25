import os
import torch
import torch.nn as nn
from models.graph.dgsl import DGSL
from transformers import AutoModel
from models.fusion import CAMF
from models.student import MVP
from models.linear import LinearCombine

class Mosaic(nn.Module):
    def __init__(self, in_dim, hidden_dim, state_dim, Nclass, fuse_dim, peft_ckpt, **kwargs):
        super(Mosaic, self).__init__()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # freeze all parameters
        self.use_vae = kwargs.get('if_rec', False)
        if_cls = kwargs.get('if_cls', False)
        self.mvp = MVP(fuse_dim, hidden_dim, Nclass, self.use_vae, if_cls)
        
        if peft_ckpt is not None:
            assert os.path.exists(peft_ckpt), "LoRA vision model and text model load failed, checkpoint not found!"
            self.mvp.load_state_dict(torch.load(peft_ckpt), strict=True)
            
        need_macro = kwargs.get('need_macro', True)
        use_dum = kwargs.get('use_dum', True)
        self.dgsl = DGSL(in_dim, hidden_dim, state_dim, need_macro, use_dum).to(device)
        
        self.fuser = LinearCombine(Nclass, fuse_dim).to(device)
        
    def forward(self, post, img, macro_graphs, micro_graphs):
        g_feat = self.dgsl(macro_graphs, micro_graphs)
        
        if not self.use_vae:
            vt_feat = self.mvp(img, post).squeeze(1)
            logits = self.fuser(vt_feat, g_feat)
            return logits
        
        T_recon, T, P_recon, P, mu_T, logvar_T, mu_P, logvar_P, vt_feat = self.mvp(img, post)
        logits = self.fuser(vt_feat, g_feat)
        return T_recon, T, P_recon, P, mu_T, logvar_T, mu_P, logvar_P, logits
        