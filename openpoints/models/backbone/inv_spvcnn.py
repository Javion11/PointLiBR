import torch
import torch.nn as nn
from torch import distributed as dist
try:
    import torchsparse
    import torchsparse.nn as spnn
    import torchsparse.nn.functional as F
    from torchsparse.nn.utils import get_kernel_offsets
    from torchsparse import PointTensor, SparseTensor
except ImportError:
    torchsparse = None
from ..build import MODELS
from .spvcnn import SPVCNN, offset2batch, initial_voxelize, voxel_to_point, point_to_voxel
from ..layers import concat_all_gather_diff


@MODELS.register_module()
class SPVCNN_Contrast(SPVCNN):
    def __init__(
        self,
        in_channels,
        out_channels,
        base_channels=32,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 2, 2, 2, 2, 2, 2, 2),
        beta=0.999,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channels=channels,
            layers=layers,)
        # ema
        self.beta = beta
        self.num_classes = out_channels
        self.register_buffer('prior_ema', torch.rand(out_channels, channels[-1]))
        self.prior_ema = nn.functional.normalize(self.prior_ema, dim=1)

    @torch.no_grad()
    def _ema(self, prior):
        """prior: n*(dim+1), feature dim + label"""
        if dist.is_initialized():
            # gather prior before updating self.prior_ema
            prior = concat_all_gather_diff(prior)
        cur_status = self.prior_ema.clone()
        for label in range(self.num_classes):
            mask_c = prior[:, -1] == label
            if mask_c.nonzero().numel() > 0:
                cur_status[label, :] = prior[mask_c, :-1].mean(0)
        self.prior_ema = self.beta * self.prior_ema + (1 - self.beta) * cur_status

    def forward(self, data_dict, is_train=False, minor_mask=None):
        grid_coord = data_dict["grid_coord"]
        feat = data_dict["x"]
        offset = data_dict["offset"]
        labels = data_dict["y"]
        batch = offset2batch(offset)
        
        # x: SparseTensor z: PointTensor
        z = PointTensor(
            feat,
            torch.cat(
                [grid_coord.float(), batch.unsqueeze(-1).float()], dim=1
            ).contiguous(),
        )
        x0 = initial_voxelize(z)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.stage1(x1)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        z1 = voxel_to_point(x4, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)

        y1 = point_to_voxel(x4, z1)
        y1.F = self.dropout(y1.F)
        y1 = self.up1[0](y1)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)
        z2 = voxel_to_point(y2, z1)
        z2.F = z2.F + self.point_transforms[1](z1.F)

        y3 = point_to_voxel(y2, z2)
        y3.F = self.dropout(y3.F)
        y3 = self.up3[0](y3)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)
        z3 = voxel_to_point(y4, z2)
        z3.F = z3.F + self.point_transforms[2](z2.F)

        seg_logits_src = self.classifier(z3.F)
        feat = z3.F
        if is_train:
            feat_norm_src = nn.functional.normalize(feat, dim=1)
            if minor_mask is not None:
                feat_norm = feat_norm_src[minor_mask, :]
                labels = labels[minor_mask]
                seg_logits = seg_logits_src[minor_mask, :]    
            else: 
                feat_norm = feat_norm_src
                seg_logits = seg_logits_src
            logits_softmax = nn.functional.softmax(seg_logits, dim=1)
            preds = logits_softmax.argmax(dim=1)
            mask_true = (preds==labels)
            self._ema(torch.cat([feat_norm.detach()[mask_true, :], labels[mask_true].unsqueeze(1)], dim=1))
            return seg_logits_src, feat_norm, self.prior_ema
        return seg_logits_src
