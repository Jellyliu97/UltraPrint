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
    
#计算
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

#利用同一个批次内部的数据做不同类别样本的对比学习，利用样本的标签进行区分
class MultiPositiveInfoNCELoss(nn.Module):
    """
    处理多正样本的InfoNCE损失
    支持不同的温度策略
    """
    
    def __init__(self, temperature=0.07, base_temperature=0.07, 
                 include_self=True, normalize=True):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.include_self = include_self
        self.normalize = normalize
    
    def forward(self, features, labels):
        device = features.device
        batch_size = features.size(0)
        
        if self.normalize:
            features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T)  # [batch, batch]
        
        # 创建正样本掩码
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # 处理是否包含自身
        if not self.include_self:
            self_mask = torch.eye(batch_size, device=device)
            mask = mask * (1 - self_mask)
        
        # 计算每个样本的正样本数
        pos_count = mask.sum(dim=1)  # [batch]
        
        # 掩码的对角线设为0（避免自身在分母中重复计算）
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        
        # 计算logits
        logits = similarity_matrix / self.temperature
        
        # 为了数值稳定性，减去最大值
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        # 计算exp
        exp_logits = torch.exp(logits) * logits_mask
        
        # 计算log概率
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
        
        # 计算每个正样本的平均log概率
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / pos_count.clamp(min=1)
        
        # 损失
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos.mean()
        
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
        
# ArcFace 的核心是学习一个"角度空间"
# 1. 为每个已知类别学习一个权重向量（类中心）
# 2. 将特征向量和类中心都归一化到单位球面
# 3. 在角度空间施加间隔 margin
class ArcFaceLoss(nn.Module):
    """
    简化但稳定的ArcFace实现
    更容易训练，数值稳定性更好
    """
    
    def __init__(self, num_classes, feat_dim, margin=0.5, scale=32):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.margin = margin
        self.scale = scale
        
        # 权重参数
        self.W = nn.Parameter(torch.Tensor(num_classes, feat_dim))
        nn.init.xavier_normal_(self.W)
        
    def forward(self, features, labels):
        """
        features: [batch_size, feat_dim]
        labels: [batch_size]
        """
        # 归一化特征和权重
        features_norm = F.normalize(features, dim=1)
        weights_norm = F.normalize(self.W, dim=1)
        
        # 计算余弦相似度
        cosine = torch.mm(features_norm, weights_norm.t())  # [batch, num_classes]
        
        # 计算角度
        with torch.no_grad():
            # 只对真实类别计算角度
            cosine_of_target = cosine[torch.arange(len(cosine)), labels]
            # 确保数值稳定性
            cosine_of_target = torch.clamp(cosine_of_target, -1+1e-7, 1-1e-7)
            theta = torch.acos(cosine_of_target)
        
        # 添加margin
        cosine_m = torch.cos(theta + self.margin)
        
        # 创建输出
        output = cosine.clone()
        output[torch.arange(len(cosine)), labels] = cosine_m
        
        # 应用缩放
        output *= self.scale
        
        return output


class ArcFaceWithCE(nn.Module):
    """
    封装ArcFace和CrossEntropy，使用更方便
    """
    
    def __init__(self, num_classes, feat_dim, margin=0.5, scale=32):
        super().__init__()
        self.arcface = ArcFaceLoss(num_classes, feat_dim, margin, scale)
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, features, labels, return_logits=False):
        """
        返回损失值，也可以返回logits用于计算准确率
        """
        # 计算ArcFace的logits
        logits = self.arcface(features, labels)
        
        # 计算交叉熵损失
        loss = self.ce_loss(logits, labels)
        
        if return_logits:
            return loss, logits
        return loss

class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
        """
        n = inputs.size(0)
        
        # Normalize features
        inputs = torch.nn.functional.normalize(inputs, p=2, dim=1)
        
        # Compute pairwise distance, (batch_size, batch_size)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        # Ensure targets is (N, 1) for broadcasting
        targets = targets.view(-1, 1)
        mask = targets.eq(targets.t())
        
        dist_ap, _ = torch.max(dist * mask.float(), dim=1)
        dist_an, _ = torch.min(dist + 1e6 * mask.float(), dim=1) # Add large val to mask out positives
        
        # Compute triplet loss
        loss = torch.clamp(dist_ap - dist_an + self.margin, min=0.0).mean()
        
        return loss
    
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