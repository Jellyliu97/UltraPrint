import torch
import torch.nn as nn
import torch.nn.functional as F
import math



#批次内的其他样本作为负样本
class InfoNCELoss2(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss2, self).__init__()

        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))

    def forward(self, proj1, proj2):
        """计算对比学习损失"""
        batch_size = proj1.shape[0]
        
        # 计算相似度矩阵
        proj1 = F.normalize(proj1, dim=-1)
        proj2 = F.normalize(proj2, dim=-1)
        
        logit_scale = self.logit_scale.exp()
        similarities = logit_scale * proj1 @ proj2.t()  # [B, B]
        
        # 创建标签 - 对角线是正样本对
        labels = torch.arange(batch_size).to(proj1.device)
        
        # 计算对称的对比损失
        loss_i = F.cross_entropy(similarities, labels)
        loss_j = F.cross_entropy(similarities.t(), labels)
        
        loss = (loss_i + loss_j) / 2
        return loss


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, image_features, text_features):
        # 归一化特征
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # 计算相似度矩阵
        logits_per_image = image_features @ text_features.T / self.temperature
        logits_per_text = logits_per_image.T
        
        # 创建标签（对角线为正样本）
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)
        
        # 计算两个方向的损失
        loss_i = self.criterion(logits_per_image, labels)
        loss_t = self.criterion(logits_per_text, labels)
        
        return (loss_i + loss_t) / 2
    

#利用同一个批次内部的数据做正负样本的对比学习，利用正负样本的标签进行区分
class BatchInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(BatchInfoNCELoss, self).__init__()
        self.temperature = temperature
        # 更安全的初始化方式
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / max(temperature, 1e-8)))

    def forward(self, proj, labels):
        """计算对比学习损失（修正稳定版本）"""
        batch_size = proj.shape[0]
        device = proj.device
        
        # 归一化特征向量
        proj = nn.functional.normalize(proj, p=2, dim=1)
        
        # 计算相似度矩阵
        logit_scale = self.logit_scale.exp()
        similarities = logit_scale * torch.mm(proj, proj.t())  # [B, B]
        
        # 创建正样本掩码（相同标签的样本对）
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # [B, B]
        
        # 排除自身作为正样本
        self_mask = torch.eye(batch_size, device=device)
        pos_mask = mask * (1 - self_mask)
        
        # 数值稳定的InfoNCE损失计算
        # 将对角线设置为负无穷，避免自身作为正样本
        similarities_adjusted = similarities - self_mask * 1e9
        
        # 计算log softmax（数值稳定）
        logits = similarities_adjusted / self.temperature
        log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        
        # 只计算正样本对的损失
        pos_per_sample = pos_mask.sum(1)  # 每个样本的正样本数量
        loss = - (pos_mask * log_probs).sum(dim=1) / (pos_per_sample + 1e-8)
        
        # 过滤没有正样本的情况
        valid_samples = pos_per_sample > 0
        if valid_samples.sum() > 0:
            loss = loss[valid_samples].mean()
        else:
            # 如果没有正样本，返回0损失避免NaN
            loss = torch.tensor(0.0, device=device)
        
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, reduction='mean'):  #alpha = poistive numbers / total numbers  正样本越少，α值应设置得越大 /   γ越大，模型越关注难样本, 0-5。
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)
        pt = torch.exp(-bce_loss)  # 预测正确的概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        

if __name__ == "__main__":  
    # 测试代码
    batch_size = 128
    feature_dim = 1
    temperature = 0.07

    proj1 = torch.randn(batch_size, feature_dim)
    # proj2 = torch.randn(batch_size, feature_dim)

    # criterion = InfoNCELoss2(temperature=temperature)
    # loss = criterion(proj1, proj2)
    # print(f"InfoNCELoss2 Loss: {loss.item()}")




    # labels = torch.randint(0, 2, (batch_size,))  # 假设有两个类别0和1
    # print(f"Labels: {labels}")
    # criterion_batch = BatchInfoNCELoss(temperature=temperature)
    # loss_batch = criterion_batch(proj1, labels)
    # print(f"BatchInfoNCELoss Loss: {loss_batch.item()}")

    focalloss = FocalLoss(alpha=0.5, gamma=2, reduction='mean')

    targets = torch.randint(0, 2, (batch_size,)).float()
    print(f"Targets: {targets.shape}")
    print(f"Proj1: {proj1.shape}")

    focal_loss_value = focalloss(proj1.squeeze(), targets)
    print(f"Focal Loss: {focal_loss_value.item()}") 

    mse_loss = nn.MSELoss()
    mse_loss_value = mse_loss(proj1.squeeze(), targets)
    print(f"MSE Loss: {mse_loss_value.item()}")