import torch
import torch.nn as nn
from .build import LOSS


@LOSS.register_module()
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf """
    def __init__(self, temperature=0.07,
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels, mask=None):
        """Compute loss for model. 
        Args:
            features: hidden vector of size [npoints, ...].
            labels: ground truth of shape [npoints].
            mask: contrastive mask of shape [npoints, npoints], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(labels.device)
        
        contrast_feature = features
        anchor_feature = features

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(features.size(0)).view(-1, 1).to(mask.device), 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+ 1e-9)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


@LOSS.register_module()
class SelfInfoNCE(nn.Module):
    def __init__(self, temperature=4, **kwargs):
       super(SelfInfoNCE, self).__init__() 
       self.temperature = temperature
       self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, fea_class):
        label = torch.arange(fea_class.size(0), device=fea_class.device)
        fea_matrix = torch.matmul(fea_class, fea_class.transpose(1,0))
        loss = self.cross_entropy(fea_matrix/self.temperature, label)
        return loss


@LOSS.register_module()
class PointInfoNCE(nn.Module):
    def __init__(self, temperature=4, **kwargs):
       super(PointInfoNCE, self).__init__() 
       self.temperature = temperature
       self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, feas1, feas2):
        # feas1, feas2: (npoints, feature_dims)
        labels = torch.arange(feas1.size(0), device=feas1.device)
        feas_matrix = torch.matmul(feas1, feas2.transpose(1,0))
        loss = self.cross_entropy(feas_matrix/self.temperature, labels)
        return loss
