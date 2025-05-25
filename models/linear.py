import torch
import torch.nn as nn
from models.fusion import CrossAttention

class Proj(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim=4):
        super(Proj, self).__init__()
        self.CA_1 = CrossAttention(input_dim, embed_dim)
        self.CA_2 = CrossAttention(input_dim, embed_dim)
        self.fc = nn.Linear(embed_dim * 2, output_dim)
    
    def forward(self, feat_A, feat_B):
        
        a2b = self.CA_1(feat_A, feat_B).squeeze(1)
        b2a = self.CA_2(feat_B, feat_A).squeeze(1)
        x = torch.cat([a2b, b2a], dim=1)
        x = self.fc(x)
        return x

class LinearCombine(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, dropout=0.2):
        """
        混合模态融合模型
        Args:
            main_dim (int): 主模态特征维度
            aux_dim (int): 辅助模态特征维度
            num_classes (int): 分类类别数
            hidden_dim (int): 隐藏层维度
            dropout (float): Dropout概率
        """
        super(LinearCombine, self).__init__()
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # 融合特征的分类器
        self.fused_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # 主模态的独立分类器
        self.main_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # 融合权重的可学习参数
        self.gamma = nn.Parameter(torch.tensor(0.3))  # 初始偏向主模态
        
    def forward(self, main_features, aux_features):
        """
        前向传播
        Args:
            main_features (torch.Tensor): 主模态特征，形状 (batch_size, main_dim)
            aux_features (torch.Tensor): 辅助模态特征，形状 (batch_size, aux_dim)
        Returns:
            torch.Tensor: 分类logit，形状 (batch_size, num_classes)
        """
        
        # 注意力机制：基于主模态生成注意力权重
        attention_weights = self.attention(main_features)
        weighted_aux = attention_weights * aux_features
        
        # 特征级融合
        fused_features = main_features + weighted_aux
        
        # 融合特征 logit
        fused_logit = self.fused_classifier(fused_features)
        
        # 主模态 logit
        main_logit = self.main_classifier(main_features)
        
        # 决策级融合
        gamma = torch.sigmoid(self.gamma)  
        final_logit = gamma * fused_logit + (1 - gamma) * main_logit
        
        return final_logit
        