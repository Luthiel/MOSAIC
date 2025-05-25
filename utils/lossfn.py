import torch
import torch.nn as nn
import torch.nn.functional as F

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.omega = nn.Parameter(torch.tensor(0.5), requires_grad=True)
    
    def forward(self, s_v_feat, t_v_feat, s_t_feat, t_t_feat):
        vision_loss = F.mse_loss(s_v_feat, t_v_feat)
        text_loss = F.mse_loss(s_t_feat, t_t_feat)
        return self.omega * vision_loss + (1 - self.omega) * text_loss
        
        
class KLDistillationLoss(torch.nn.Module):
    def __init__(self, temperature=2.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits):
        """
        student_logits: 学生模型的原始 logit (batch_size, num_classes)
        teacher_logits: 教师模型的原始 logit (batch_size, num_classes)
        """
        tau2 = self.temperature ** 2  

        # 计算 softmax 概率分布
        p_s = F.softmax(student_logits / self.temperature, dim=-1)
        p_t = F.softmax(teacher_logits / self.temperature, dim=-1)

        # 计算 KL 散度（教师分布作为目标）
        kl_loss = F.kl_div(p_s.log(), p_t, reduction="batchmean") * tau2

        return kl_loss

        
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, h_v, h_t):
        # 计算相似度矩阵
        sim_matrix = torch.matmul(h_v, h_t.T) / self.temperature  # [B, B]

        # 计算对角线上的正样本相似度（正对比）
        pos_sim = torch.diagonal(sim_matrix)

        # 计算 log_softmax 归一化损失
        loss = -torch.log(torch.exp(pos_sim) / torch.exp(sim_matrix).sum(dim=1))

        return loss.mean()


class KDLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2, kl_tmp=1.0, ct_tmp=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse_loss = MSELoss()
        self.kld_loss = KLDistillationLoss(kl_tmp)
        self.contrastive_loss = ContrastiveLoss(ct_tmp)
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, s_v_feat, t_v_feat, s_t_feat, t_t_feat, s_logits, t_logits):
        mse = self.mse_loss(s_v_feat, t_v_feat, s_t_feat, t_t_feat)
        kld = self.kld_loss(s_logits, t_logits)
        contrastive = self.contrastive_loss(s_v_feat, s_t_feat)

        total_loss = self.alpha * mse + self.beta * kld + self.gamma * contrastive
        return total_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=3.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)

        logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)  
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)       

        if self.alpha is not None:
            alpha_t = self.alpha[targets]                       
            loss = -alpha_t * ((1 - pt) ** self.gamma) * logpt
        else:
            loss = -((1 - pt) ** self.gamma) * logpt

        return loss.mean() if self.reduction == 'mean' else loss.sum()


class FinalLoss(nn.Module):
    def __init__(self, alpha=None, beta=0.1, gamma=3.0, reduction='mean'):
        super().__init__()
        self.focal_loss = FocalLoss(alpha, gamma, reduction)
        self.beta = beta
    
    def kl_divergence(self, mu1, logvar1, mu2, logvar2):
        """计算两个高斯分布的 KL 散度"""
        return 0.5 * torch.sum(
            logvar2 - logvar1 + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / torch.exp(logvar2) - 1
        )

    def forward(self, T_recon, T, P_recon, P, mu_T, logvar_T, mu_P, logvar_P, logits, labels):
        recon_loss = F.mse_loss(T_recon, T) + F.mse_loss(P_recon, P)
        kl_loss = self.kl_divergence(mu_T, logvar_T, mu_P, logvar_P)
        focal_loss = self.focal_loss(logits, labels)
        
        # print("-----------------")
        # print(recon_loss, kl_loss, focal_loss)
        return 10 * recon_loss + self.beta * kl_loss + focal_loss
        # return 10 * recon_loss + focal_loss


'''
def fusion_loss(model, T, E, edge_index, lambda_ca, lambda_da):
    recon_T, recon_E, mu_T, logvar_T, mu_E, logvar_E, z_T, z_E = model(T, E, edge_index)
    
    # VAE损失（KL散度 + 重构损失）
    kl_T = -0.5 * torch.sum(1 + logvar_T - mu_T.pow(2) - logvar_T.exp())
    kl_E = -0.5 * torch.sum(1 + logvar_E - mu_E.pow(2) - logvar_E.exp())
    recon_loss_T = F.mse_loss(recon_T, T, reduction='sum')  # 图特征的重构损失
    recon_loss_E = F.mse_loss(recon_E, E, reduction='sum')  # 融合特征的重构损失
    vae_loss = kl_T + kl_E + recon_loss_T + recon_loss_E
    
    # 对抗损失（DA）
    real_labels = torch.ones(z_T.size(0), 1)
    fake_labels = torch.zeros(z_E.size(0), 1)
    d_real = model.discriminator(z_T)
    d_fake = model.discriminator(z_E)
    da_loss = F.binary_cross_entropy(d_real, real_labels) + F.binary_cross_entropy(d_fake, fake_labels)
    
    # CA损失（z_T 和 z_E 之间的L2距离）
    ca_loss = F.mse_loss(z_T, z_E)
    
    # 总损失
    total_loss = vae_loss + lambda_ca * ca_loss + lambda_da * da_loss
    return total_loss
'''