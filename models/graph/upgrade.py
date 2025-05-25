import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv

class AssociationScore(nn.Module):
    """计算节点的全局关联性评分"""
    def __init__(self, in_dim, hidden_dim):
        super(AssociationScore, self).__init__()
        self.gcn = GCNConv(in_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index):
        h = self.gcn(x, edge_index)  # GNN 计算邻域特征
        score = self.mlp(h)  # 计算关联性，正负关联
        return score.squeeze()  

class GatingDenoising(nn.Module):
    """门控机制降低边权重"""
    def __init__(self):
        super(GatingDenoising, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, edge_weights, scores, edge_index):
        i, j = edge_index
        gate = torch.sigmoid(self.alpha * (scores[i] + scores[j]) + self.beta)  
        return edge_weights * gate  # 降噪后的边权重

class SVDFilter(nn.Module):
    """SVD 低秩近似降噪"""
    def __init__(self, use_energy=False, threshold=0.9):
        super(SVDFilter, self).__init__()
        self.use_energy = use_energy
        self.threshold = threshold

    def forward(self, x):
        U, Sigma, V = torch.svd(x) 

        if self.use_energy:
            total = torch.sum(Sigma)
            cumulative = torch.cumsum(Sigma, dim=0)
            k = torch.argmax(cumulative >= self.threshold * total) + 1
        else:
            diffs = torch.diff(Sigma)
            k = (diffs > (1 - self.threshold)).sum() + 1

        # 保留前 k 个奇异值和对应的向量
        U_k = U[:, :k]
        Sigma_k = torch.diag(Sigma[:k])
        V_k = V[:, :k]  # 注意 V 是列正交矩阵

        # 重构低秩矩阵
        x_denoised = U_k @ Sigma_k @ V_k.T
        return x_denoised

class FeatureEnhancer(nn.Module):
    """MLP 进行特征映射增强"""
    def __init__(self, in_dim, out_dim):
        super(FeatureEnhancer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, scores):
        enhanced_x = self.mlp(x)
        return enhanced_x * scores.unsqueeze(-1) + x  # 增强特征

class DUM(nn.Module):
    """完整的降噪提升模块"""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(DUM, self).__init__()
        self.assoc = AssociationScore(in_dim, hidden_dim)
        self.gating = GatingDenoising()
        self.svd = SVDFilter()
        self.enhancer = FeatureEnhancer(in_dim, out_dim)

    def forward(self, x, edge_index, edge_weights):
        x = x.cuda()
        edge_index = edge_index.cuda()
        edge_weights = edge_weights.cuda()
        scores = self.assoc(x, edge_index)  # 计算关联性
        edge_weights = self.gating(edge_weights, scores, edge_index)  # 降噪
        
        torch.cuda.empty_cache()
        del edge_index
        
        x_svd = self.svd(x)  # SVD 低噪声特征
        x_fused = 0.5 * x + 0.5 * x_svd  # 结合原始和降噪特征
        
        torch.cuda.empty_cache()
        del x_svd
        
        x_enhanced = self.enhancer(x_fused, scores)  # 进一步增强
        
        torch.cuda.empty_cache()
        del x_fused, scores
        
        return x_enhanced, edge_weights  # 返回增强的特征和降噪后的边权重

'''
# 示例数据
num_nodes = 5
in_dim, hidden_dim, out_dim = 16, 32, 16
x = torch.randn(num_nodes, in_dim).cuda()  # 节点特征
edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]).cuda() # 邻接边
edge_weights = torch.rand(edge_index.shape[1], 1).cuda()  # 初始边权重

# 运行模型
dum = DUM(in_dim, hidden_dim, out_dim).cuda()
x_enhanced, edge_weights_new = dum(x, edge_index, edge_weights)

print("Original Node Features:", x)
print("Enhanced Node Features:", x_enhanced.shape)
print("Updated Edge Weights:", edge_weights_new.shape)
'''