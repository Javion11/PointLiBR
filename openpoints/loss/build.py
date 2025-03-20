import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, KLDivLoss
from openpoints.utils import registry

LOSS = registry.Registry('loss')
LOSS.register_module(name='CrossEntropy', module=CrossEntropyLoss)
LOSS.register_module(name='CrossEntropyLoss', module=CrossEntropyLoss)
LOSS.register_module(name='KLDivLoss', module=KLDivLoss)


@LOSS.register_module()
class CriterionKD(nn.Module):
    '''
    Knowledge Distillation Loss
    '''
    def __init__(self, temperature=4, **kwargs):
        super(CriterionKD, self).__init__()
        self.temperature = temperature
        self.criterion_kd = nn.KLDivLoss(reduction='mean')

    def forward(self, pred, soft):
        soft = soft.detach()
        loss = self.criterion_kd(F.log_softmax(pred / self.temperature, dim=1), \
            F.softmax(soft / self.temperature, dim=1)) * self.temperature ** 2
        return loss


@LOSS.register_module()
class CriterionKD_CrossEntropy(nn.Module):
    '''
    Knowledge Distillation Loss + CrossEntropy
    '''
    def __init__(self, alpha=0.5, temperature=4, **kwargs):
        super(CriterionKD_CrossEntropy, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.criterion_kd = nn.KLDivLoss(reduction='mean')
        self.criterion_hard = CrossEntropyLoss(label_smoothing=0.2)

    def forward(self, pred, soft, label):
        soft = soft.detach()
        loss = self.alpha * self.criterion_kd(F.log_softmax(pred / self.temperature, dim=1), \
            F.softmax(soft / self.temperature, dim=1)) * self.temperature ** 2 + (1-self.alpha) * self.criterion_hard(pred, label)
        return loss


@LOSS.register_module()
class Focal_Loss(nn.Module):
    """
    Focal loss implemented in multiple classifications
    Formula: - alpha*(1-pt)**gamma*cross_entropy (pt is softmax(pred); label_smoothing param control cross_entropy)
    """
    def __init__(self, alpha=0.25, gamma=2, label_smoothing=0, weight=None, num_classes=13): 
        super(Focal_Loss,self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.weight = weight
        self.num_classes = num_classes
    
    def forward(self, pred, label):
        """
        pred: prediction output (N, num_classes)
        label: gt (N)
        """
        softmax = F.softmax(pred, dim=1)
        log_softmax = F.log_softmax(pred, dim=1)
        one_hot = torch.zeros((label.size(0), self.num_classes), device=label.device).scatter(1, label.unsqueeze(1), 1)
        if self.label_smoothing != 0:
            one_hot = one_hot * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
        
        ce_loss = -(one_hot * log_softmax).sum(dim=1)

        pt = (softmax * one_hot).sum(dim=1)
        focal_weight = (1 - pt).pow(self.gamma)
        if self.alpha is not None:
            alpha_weight = (self.alpha * one_hot).sum(dim=1)
            focal_weight = focal_weight * alpha_weight
        floss = (focal_weight * ce_loss).mean()
        return floss


@LOSS.register_module()
class SmoothCrossEntropy(nn.Module):
    def __init__(self, label_smoothing=0.2, 
                 ignore_index=None, 
                 num_classes=None, 
                 weight=None, 
                 return_valid=False
                 ):
        super(SmoothCrossEntropy, self).__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.return_valid = return_valid
        # Reduce label values in the range of logit shape
        if ignore_index is not None:
            reducing_list = torch.range(0, num_classes).long().cuda(non_blocking=True)
            inserted_value = torch.zeros((1, )).long().cuda(non_blocking=True)
            self.reducing_list = torch.cat([
                reducing_list[:ignore_index], inserted_value,
                reducing_list[ignore_index:]
            ], 0)
        if weight is not None:
            self.weight = torch.from_numpy(weight).float().cuda(
                non_blocking=True).squeeze()
        else:
            self.weight = None
            
    def forward(self, pred, gt):
        if len(pred.shape)>2:
            pred = pred.transpose(1, 2).reshape(-1, pred.shape[1])
        gt = gt.contiguous().view(-1)
        
        if self.ignore_index is not None: 
            valid_idx = gt != self.ignore_index
            pred = pred[valid_idx, :]
            gt = gt[valid_idx]        
            gt = torch.gather(self.reducing_list, 0, gt)
            
        if self.label_smoothing > 0:
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - self.label_smoothing) + (1 - one_hot) * self.label_smoothing / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)
            if self.weight is not None:
                loss = -(one_hot * log_prb * self.weight).sum(dim=1).mean()
            else:
                loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gt, weight=self.weight)
        
        if self.return_valid:
            return loss, pred, gt
        else:
            return loss


@LOSS.register_module()
class MaskedCrossEntropy(nn.Module):
    def __init__(self, label_smoothing=0.2, weight=None, ignore_index=None, **kwargs):
        super(MaskedCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        if weight is not None: 
            weight = torch.tensor(weight)
        self.creterion = CrossEntropyLoss(label_smoothing=label_smoothing, weight=weight)
        
    def forward(self, logit, target, mask):
        if len(logit.size())==3: # (b,n,c); not suitable for (b*n,c)
            logit = logit.transpose(1, 2).reshape(-1, logit.shape[1])
            target = target.flatten()
            mask = mask.flatten()
        
        if self.ignore_index == 0: # NOTE: map [1,8] to [0,7] in Semantic3D (ignore_index=0), fit torch crossentropy function 
            loss = self.creterion(logit[mask], target[mask]-1) 
        else: # scannetv2 ignore index==255
            loss = self.creterion(logit[mask], target[mask]) 
        return loss


def build_criterion_from_cfg(cfg, **kwargs):
    """
    Build a criterion (loss function), defined by cfg.NAME.
    Args:
        cfg (eDICT): 
    Returns:
        criterion: a constructed loss function specified by cfg.NAME
    """
    return LOSS.build(cfg, **kwargs)