import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CrossAttention(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(CrossAttention, self).__init__()
        self.q_proj = nn.Linear(input_dim, embed_dim)
        self.k_proj = nn.Linear(input_dim, embed_dim)
        self.v_proj = nn.Linear(input_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q_feat, K_feat):
        # print(Q_feat.shape, K_feat.shape)
        B, N = Q_feat.shape

        Q_feat = Q_feat.to(self.q_proj.weight.dtype)
        K_feat = K_feat.to(self.k_proj.weight.dtype)

        Q = self.q_proj(Q_feat).unsqueeze(-1)  # shape: [B, D, 1]
        K = self.k_proj(K_feat).unsqueeze(-1) 
        V = self.v_proj(K_feat).unsqueeze(-1)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(1, dtype=torch.float32))
        attn_weights = self.softmax(attn_scores)  # shape: [B, 1, 1]

        attn_feats = torch.matmul(attn_weights, V)  
        attn_feats = attn_feats.permute(0, 2, 1)

        return attn_feats

class Matcher(nn.Module):
    def __init__(self, feature_dim, seq_len=None, reduction_factor=10):
        """
        跨模态注意力匹配模块
        Args:
            feature_dim (int): 输入特征的维度
            seq_len (int, optional): 序列长度，若为None则动态适应输入
            reduction_factor (int): 降维因子，用于减少计算量
        """
        super(Matcher, self).__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.reduction_factor = reduction_factor

        # 动态投影层
        if seq_len is not None:
            reduced_dim = seq_len // reduction_factor
            self.proj_a = nn.Sequential(
                nn.Linear(seq_len, reduced_dim),
                nn.ReLU(),
                nn.Linear(reduced_dim, seq_len)
            )
            self.proj_b = nn.Sequential(
                nn.Linear(seq_len, reduced_dim),
                nn.ReLU(),
                nn.Linear(reduced_dim, seq_len)
            )
            self.fusion = nn.Sequential(
                nn.Linear(seq_len * 2, seq_len // 5),
                nn.ReLU(),
                nn.Linear(seq_len // 5, seq_len)
            )
        else:
            # 若seq_len为None，则使用自适应层
            self.proj_a = nn.Linear(feature_dim, feature_dim)
            self.proj_b = nn.Linear(feature_dim, feature_dim)
            self.fusion = nn.Linear(feature_dim * 2, feature_dim)

        self.sigmoid = nn.Sigmoid()

    def compute_similarity(self, feat_a, feat_b):
        
        feat_a = F.normalize(feat_a, p=2, dim=1)
        feat_b = F.normalize(feat_b, p=2, dim=1)
        similarity = torch.sum(feat_a * feat_b, dim=1, keepdim=True)
        return similarity

    def forward(self, feat_a, feat_b, mode='default'):
        
        # 动态调整序列长度
        if self.seq_len is None:
            seq_len = feat_a.shape[-1]
            self.proj_a = nn.Linear(seq_len, seq_len).to(feat_a.device)
            self.proj_b = nn.Linear(seq_len, seq_len).to(feat_a.device)
            self.fusion = nn.Linear(seq_len * 2, seq_len).to(feat_a.device)

        # 投影特征
        proj_a = F.relu(self.proj_a(feat_a))
        proj_b = F.relu(self.proj_b(feat_b))

        # 计算相似度
        similarity = self.sigmoid(self.compute_similarity(feat_a, feat_b))

        # 根据模式加权特征
        if mode == 'weighted':
            weighted_a = similarity * proj_a
            weighted_b = similarity * proj_b
        else:
            weighted_a = (1 - similarity) * proj_a
            weighted_b = (1 - similarity) * proj_b

        # 融合特征
        combined = torch.cat([weighted_a, weighted_b], dim=-1)
        fused = F.relu(self.fusion(combined))

        return fused


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)
    
    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, output_dim)
    
    def forward(self, z):
        return self.fc(z)

class CAMF(nn.Module):
    def __init__(self, feature_dim, latent_dim, Nclass=4, cls=False):
        super().__init__()
        self.encoder_T = Encoder(feature_dim, latent_dim)  # 图特征编码
        self.encoder_P = Encoder(feature_dim, latent_dim)  # 文本-视觉特征编码
        self.decoder_T = Decoder(latent_dim, feature_dim)  # 图特征解码
        self.decoder_P = Decoder(latent_dim, feature_dim)  # 文本-视觉特征解码
        
        self.fuser = Matcher(feature_dim)
        
        self.cls = cls
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, Nclass)
        )
        
    def forward(self, T, P):
        mu_T, logvar_T = self.encoder_T(T)
        mu_P, logvar_P = self.encoder_P(P)
        
        # 采样隐变量
        std_T, std_P = torch.exp(0.5 * logvar_T), torch.exp(0.5 * logvar_P)
        eps = torch.randn_like(std_T)
        z_T, z_P = mu_T + eps * std_T, mu_P + eps * std_P
        
        # 交叉解码
        T_recon = self.decoder_P(z_P)
        P_recon = self.decoder_T(z_T)
        
        fuse_feat = self.fuser(T_recon, P_recon, 'default')
        if not self.cls:
            return T_recon, T, P_recon, P, mu_T, logvar_T, mu_P, logvar_P, fuse_feat
        
        logits = self.mlp(fuse_feat)
        return T_recon, T, P_recon, P, mu_T, logvar_T, mu_P, logvar_P, logits

# 示例数据
# feature_dim, latent_dim = 128, 32
# model = CAMF(feature_dim, latent_dim)
# T = torch.randn(16, feature_dim)  # 图特征
# P = torch.randn(16, feature_dim)  # 文本-视觉特征
# loss = model(T, P)
# print("Loss:", loss.item())
    
