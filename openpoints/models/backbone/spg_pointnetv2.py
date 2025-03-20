import copy
import torch
import torch.nn as nn
from torch import distributed as dist
from ..build import MODELS, build_model_from_cfg
from ..layers import concat_all_gather_diff


@MODELS.register_module()
class BaseSeg_Balance_Prior(nn.Module):
    def __init__(self,
                 beta=0.999,
                 encoder_args=None,
                 cls_args=None,
                 **kwargs):
        super().__init__()
        self.beta = beta
        self.num_classes = cls_args.num_classes
        self.encoder = build_model_from_cfg(encoder_args)
        self.projection = nn.Sequential(
            nn.Linear(encoder_args.mlps[-1][-1][-1], encoder_args.mlps[-2][-1][-1]), nn.ReLU(inplace=True),
            nn.Linear(encoder_args.mlps[-2][-1][-1], encoder_args.mlps[-3][-1][-1]), nn.ReLU(inplace=True)
            )
        # ema
        self.register_buffer('prior_ema', torch.rand(self.num_classes, encoder_args.mlps[-3][-1][-1]))
        self.prior_ema = nn.functional.normalize(self.prior_ema, dim=1)

    @torch.no_grad()
    def _ema(self, prior):
        """prior: n*(dim+1), feature dim + label"""
        # NOTE: turn on broadcast_buffers=True could avoid manual sync
        if dist.is_initialized():
            # gather prior before updating self.prior_ema
            prior = concat_all_gather_diff(prior)
        cur_status = self.prior_ema.clone()
        for label in range(self.num_classes):
            mask_c = prior[:, -1] == label
            if mask_c.nonzero().numel() > 0:
                cur_status[label, :] = prior[mask_c, :-1].mean(0)
        self.prior_ema = self.beta * self.prior_ema + (1 - self.beta) * cur_status
    
    def forward(self, data, is_train=False, mask=None, ignore_index=None):
        p0, f0 = data['pos'], data['x']
        if is_train: labels = data['y']
        
        # NOTE: prior model data prepare         
        # aggregate the same class feature in a batch 
        feat, coord, class_index = [], [], []
        p0 = p0.reshape(-1, p0.size(-1)).contiguous()
        f0 = f0.transpose(1,2).reshape(-1, f0.size(1))
        labels = labels.flatten()
        for c in range(self.num_classes):
            index = (labels==c).nonzero().squeeze(1)
            if index.numel() != 0:
                if index.numel() >= 256:
                    feat.append(f0[index, :])
                    coord.append(p0[index, :])
                    class_index.append(c)        
        
        feat_out, offset = [], []
        for i in range(len(feat)):
            p, f = self.encoder.forward_all_features(coord[i].unsqueeze(0), 
                                                        feat[i].transpose(0,1).unsqueeze(0).contiguous())
            feat_out.append(f[-1].squeeze(0))
            offset.append(f[-1].size(-1))
        feat = torch.cat(feat_out, dim=-1).transpose(0,1).contiguous()
        offset = torch.cumsum(torch.IntTensor(offset), dim=0, dtype=torch.int32).cuda(device=p0.device)
        feat = self.projection(feat)

        # aggregate the same class feature in a example of a batch
        feat = nn.functional.normalize(feat, dim=1)
        current_prior = torch.cat([feat, torch.zeros([feat.size(0),1], device=feat.device)], dim=1)
        for i, c_index in enumerate(class_index):
            begin = offset[i-1] if i > 0 else 0
            end = offset[i]
            current_prior[begin:end, -1] = c_index
        self._ema(current_prior.detach())
        self.prior_ema = nn.functional.normalize(self.prior_ema, dim=1)
        return current_prior, self.prior_ema


@MODELS.register_module()
class BaseSeg_Balance_Main(nn.Module):
    def __init__(self,
                 beta=0.999,
                 encoder_args=None,
                 decoder_args=None,
                 cls_args=None,
                 **kwargs):
        super().__init__()
        self.beta = beta
        self.num_classes = cls_args.num_classes
        self.encoder = build_model_from_cfg(encoder_args)
        if decoder_args is not None:
            decoder_args_merged_with_encoder = copy.deepcopy(encoder_args)
            decoder_args_merged_with_encoder.update(decoder_args)
            decoder_args_merged_with_encoder.encoder_channel_list = self.encoder.channel_list if hasattr(self.encoder, 'channel_list') else None
            self.decoder = build_model_from_cfg(decoder_args_merged_with_encoder) 
        else:
            self.decoder = None

        if cls_args is not None:
            if hasattr(self.decoder, 'out_channels'):
                in_channels = self.decoder.out_channels
            elif hasattr(self.encoder, 'out_channels'):
                in_channels = self.encoder.out_channels
            else:
                in_channels = cls_args.get('in_channels', None)
            cls_args.in_channels = in_channels
            self.head = build_model_from_cfg(cls_args)
        else:
            self.head = None
        # ema
        self.register_buffer('prior_ema', torch.rand(self.num_classes, encoder_args.mlps[-3][-1][-1]))
        self.prior_ema = nn.functional.normalize(self.prior_ema, dim=1)

    @torch.no_grad()
    def _ema(self, prior):
        """prior: n*(dim+1), feature dim + label"""
        # NOTE: turn on broadcast_buffers=True could avoid manual sync
        if dist.is_initialized():
            # gather prior before updating self.prior_ema
            prior = concat_all_gather_diff(prior)
        cur_status = self.prior_ema.clone()
        for label in range(self.num_classes):
            mask_c = prior[:, -1] == label
            if mask_c.nonzero().numel() > 0:
                cur_status[label, :] = prior[mask_c, :-1].mean(0)
        self.prior_ema = self.beta * self.prior_ema + (1 - self.beta) * cur_status
    
    def forward(self, data, is_train=False, mask=None, ignore_index=None):
        p0, f0 = data['pos'], data['x']
        if is_train: labels = data['y']
        p, f = self.encoder.forward_all_features(p0, f0)
        f = self.decoder(p, f).squeeze(-1)
        logits = self.head(f)
        if not is_train: return logits.squeeze(0).transpose(0,1).contiguous()
        
        labels = labels.reshape(-1).contiguous()
        f = f.transpose(1, 2).reshape(-1, f.size(1)).contiguous()
        logits = logits.transpose(1, 2).reshape(-1, logits.size(1)).contiguous()
        feas_norm = nn.functional.normalize(f, dim=1)
        logits_softmax = nn.functional.softmax(logits, dim=1)
        preds = logits_softmax.argmax(dim=1)
        mask_true = (preds==labels)
        self._ema(torch.cat([feas_norm.detach()[mask_true, :], labels[mask_true].unsqueeze(1)], dim=1))
        self.prior_ema = nn.functional.normalize(self.prior_ema, dim=1)
        return logits, feas_norm, self.prior_ema
