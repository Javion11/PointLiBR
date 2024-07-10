import torch
import torch.nn as nn
from torch import distributed as dist
import torch_scatter
from openpoints.cpp.pointops.functions import pointops
from ..build import MODELS
from .ptnet import TransitionDown, TransitionUp, PointTransformerBlock
from .ptnetv2 import GVAPatchEmbed, Encoder, Decoder, PointBatchNorm 
from ..layers import concat_all_gather_diff


@MODELS.register_module()
class PTSeg_Balance_Prior(nn.Module):
    def __init__(self,
                 block=PointTransformerBlock,
                 blocks=[2, 3, 4, 6, 3],    # depth, default: blocks=[2, 3, 4, 6, 3]
                 width=32,
                 nsample=[8, 16, 16, 16, 16],
                 in_channels=6,
                 num_classes=13,
                 mid_res=False,
                 beta=0.999,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.beta = beta
        if in_channels<6:
            in_channels = in_channels + 3 # modify in_channels from rgb(z) to xyz+rgb(z)
        self.in_planes, planes = in_channels, [width * 2**i for i in range(len(blocks))]
        share_planes = 8
        stride, nsample = [1, 4, 4, 4, 4], nsample

        if isinstance(block, str):
            block = eval(block)
        self.mid_res = mid_res

        # prior model
        self.enc1_prior = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0],
                                   nsample=nsample[0])  # N/1
        self.enc2_prior = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1],
                                   nsample=nsample[1])  # N/4
        self.enc3_prior = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2],
                                   nsample=nsample[2])  # N/16
        self.enc4_prior = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3],
                                   nsample=nsample[3])  # N/64
        self.enc5_prior = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4],
                                   nsample=nsample[4])  # N/256
        self.projection = nn.Sequential(nn.Linear(planes[4], planes[3]), nn.BatchNorm1d(planes[3]), nn.ReLU(inplace=True),
                                        nn.Linear(planes[3], planes[2]), nn.BatchNorm1d(planes[2]), nn.ReLU(inplace=True),
                                        nn.Linear(planes[2], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True))
        
        self.register_buffer('prior_ema', torch.rand(num_classes, planes[0]))
        self.prior_ema = nn.functional.normalize(self.prior_ema, dim=1)

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample, mid_res=self.mid_res))
        return nn.Sequential(*layers)

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

    def forward(self, p0, x0=None, o0=None):
        # p, x, o: points, features, batches 
        # The dataloader input here is different from PointTransformer source code; 
        # it's need to modify input data{'pos':, 'x':, 'label':, } to the following form.
        if x0 is None:  # this means p0 is a dict.
            p0, x0, o0, labels = p0['pos'], p0.get('x', None), p0.get('offset', None), p0.get('y', None)
            if x0 is None:
                x0 = p0
            if o0 == None:
                o0, count = [], 0
                for _ in range(p0.size()[0]):
                    count += p0.size()[1]
                    o0.append(count)
                o0 = torch.IntTensor(o0).cuda(device=p0.device)
            if len(x0.size())>2: # means x0:(b, c ,n), need to be catted to (b*n, c)
                x0 = x0.transpose(1,2).contiguous() # po(b, n, 3), x0(b, n, c=3)
                p0 = torch.cat([p0_split.squeeze() for p0_split in p0.split(1,0)])
                x0 = torch.cat([x0_split.squeeze() for x0_split in x0.split(1,0)]) 
        if x0.size(1)<6:
            x0 = torch.cat((p0,x0),1) # x0(n, c=in_channels+3)

        # NOTE: prior model data prepare
        # # method 1: aggregate the same class feature in a batch 
        # feat, coord, offset = [], [], []
        # point_set = [] # record the number of point set for each class in a batch
        # batch = o0.size(0)
        # for c in range(self.num_classes):
        #     point_set_num = 0
        #     index = (label==c).nonzero().squeeze(1)
        #     if index.numel() != 0:
        #         for b in range(batch):
        #             if b == 0:
        #                 point_mask = index<o0[b]
        #             else:
        #                 point_mask = (o0[b-1]<=index) & (index<o0[b])
        #             point_num = point_mask.sum()
        #             # NOTE: When the number of points in the point cloud set is less than the sampling rate, 
        #             # fill it up to the sampling rate (256)
        #             if point_num >= 256:
        #                 feat.append(x0[index[point_mask], :])
        #                 coord.append(p0[index[point_mask], :])
        #                 offset.append(point_num)
        #                 point_set_num += 1
        #             elif (point_num > 0) and (point_num < 256):
        #                 select_index = torch.randint(0, point_num, (256,))
        #                 select_index = index[point_mask][select_index]
        #                 feat.append(x0[select_index, :])
        #                 coord.append(p0[select_index, :])
        #                 offset.append(torch.tensor(256))
        #                 point_set_num += 1
        #     point_set.append(point_set_num)            
        # feat = torch.cat(feat, dim=0)
        # coord = torch.cat(coord, dim=0)
        # offset = torch.cumsum(torch.IntTensor(offset), dim=0, dtype=torch.int32).cuda(device=p0.device)
        # point_set = torch.tensor(point_set)

        # method 2: aggregate the same class feature in a example of a batch
        feat, coord, offset = [], [], []
        class_index_batch = [] # record the class index in a example of a batch
        for b in range(o0.size(0)):
            class_index = [] # record the class index in a example
            for c in range(self.num_classes):
                index = (labels==c).nonzero().squeeze(1)
                if b == 0:
                    point_mask = index<o0[b]
                else:
                    point_mask = (o0[b-1]<=index) & (index<o0[b])
                point_num = point_mask.sum().item()
                if point_num != 0:
                    # NOTE: When the number of points in the point cloud set is less than the sampling rate, 
                    # fill it up to the sampling rate (256)
                    if point_num >= 256:
                        feat.append(x0[index[point_mask], :])
                        coord.append(p0[index[point_mask], :])
                        offset.append(point_num)
                    elif (point_num > 0) and (point_num < 256):
                        select_index = torch.randint(0, point_num, (256,))
                        select_index = index[point_mask][select_index]
                        feat.append(x0[select_index, :])
                        coord.append(p0[select_index, :])
                        offset.append(torch.tensor(256))
                    class_index.append(c)     
            class_index_batch.append(class_index)       
        feat = torch.cat(feat, dim=0)
        coord = torch.cat(coord, dim=0)
        offset = torch.cumsum(torch.IntTensor(offset), dim=0, dtype=torch.int32).cuda(device=p0.device)
        
        # prior information process
        p1_prior, x1_prior, o1_prior = self.enc1_prior([coord, feat, offset])
        p2_prior, x2_prior, o2_prior = self.enc2_prior([p1_prior, x1_prior, o1_prior])
        p3_prior, x3_prior, o3_prior = self.enc3_prior([p2_prior, x2_prior, o2_prior])
        p4_prior, x4_prior, o4_prior = self.enc4_prior([p3_prior, x3_prior, o3_prior])
        p5_prior, feat, offset = self.enc5_prior([p4_prior, x4_prior, o4_prior])
        feat = self.projection(feat)
        # # method 1: aggregate the same class feature in a batch 
        # prior = []
        # for i, num in enumerate(point_set):
        #     if num == 0:
        #         prior.append(memory_prior[i, :].unsqueeze(0))
        #     else:
        #         if i == 0:
        #             begin = 0
        #         elif point_set[:i].sum() == 0: 
        #             begin = 0
        #         else:
        #             begin = offset[point_set[:i].sum()]
        #         end = offset[point_set[:i+1].sum()]
        #         prior.append(feat[begin:end, :].mean(dim=0, keepdim=True))
        # prior = torch.cat(prior, dim=0)
        # prior = torch.div(prior, torch.norm(prior,dim=1, keepdim=True) + 1e-9)

        # method 2: aggregate the same class feature in a example of a batch
        feat = nn.functional.normalize(feat, dim=1)
        current_prior = torch.cat([feat, torch.zeros([feat.size(0),1], device=feat.device)], dim=1)
        class_index_len = 0
        for i in range(o0.size(0)):
            for j, c_index in enumerate(class_index_batch[i]):
                begin = offset[class_index_len-1] if class_index_len-1 >= 0 else 0
                end = offset[class_index_len]
                current_prior[begin:end, -1] = c_index
                class_index_len += 1
        self._ema(current_prior.detach())
        self.prior_ema = nn.functional.normalize(self.prior_ema, dim=1)
        return current_prior, self.prior_ema


@MODELS.register_module()
class PTSeg_Balance_Main(nn.Module):
    def __init__(self,
                 block=PointTransformerBlock,
                 blocks=[2, 3, 4, 6, 3],    # depth
                 width=32,
                 nsample=[8, 16, 16, 16, 16],
                 in_channels=6,
                 num_classes=13,
                 dec_local_aggr=True,
                 mid_res=False,
                 beta=0.999,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        if in_channels<6:
            in_channels = in_channels + 3 # modify in_channels from rgb(z) to xyz+rgb(z)
        self.in_planes, planes = in_channels, [width * 2**i for i in range(len(blocks))]
        share_planes = 8
        stride, nsample = [1, 4, 4, 4, 4], nsample
        self.beta = beta

        if isinstance(block, str):
            block = eval(block)
        self.mid_res = mid_res
        self.dec_local_aggr = dec_local_aggr

        # main model enc 5
        self.in_planes = in_channels
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0],
                                   nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1],
                                   nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2],
                                   nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3],
                                   nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4],
                                   nsample=nsample[4])  # N/256
        # main model dec 5, no interpolation
        self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample[4], True)  # transform p5
        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample[0])  # fusion p2 and p1
        self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True),
                                nn.Linear(planes[0], self.num_classes))

        self.register_buffer('prior_ema', torch.rand(num_classes, planes[0]))
        self.prior_ema = nn.functional.normalize(self.prior_ema, dim=1)

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample, mid_res=self.mid_res))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion

        if self.dec_local_aggr:
            for _ in range(1, blocks):
                layers.append(block(self.in_planes, self.in_planes, share_planes,
                              nsample=nsample, mid_res=self.mid_res))
        return nn.Sequential(*layers)

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

    def forward(self, p0, x0=None, o0=None, is_train=False):
        # p, x, o: points, features, batches 
        # The dataloader input here is different from PointTransformer source code; 
        # it's need to modify input data{'pos':, 'x':, 'label':, } to the following form.
        if x0 is None:  # this means p0 is a dict.
            p0, x0, o0, labels = p0['pos'], p0.get('x', None), p0.get('offset', None), p0.get('y', None)
            if x0 is None:
                x0 = p0
            if o0 == None:
                o0, count = [], 0
                for _ in range(p0.size()[0]):
                    count += p0.size()[1]
                    o0.append(count)
                o0 = torch.IntTensor(o0).cuda(device=p0.device)
            if len(x0.size())>2: # means x0:(b, c ,n), need to be catted to (b*n, c)
                x0 = x0.transpose(1,2).contiguous() # po(b, n, 3), x0(b, n, c=3)
                p0 = torch.cat([p0_split.squeeze() for p0_split in p0.split(1,0)])
                x0 = torch.cat([x0_split.squeeze() for x0_split in x0.split(1,0)]) 
        if x0.size(1)<6:
            x0 = torch.cat((p0,x0),1) # x0(n, c=in_channels+3)       
        
        # main model encoder and decoder
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        logits = self.cls(x1)
        if not is_train: return logits
        feas_norm = nn.functional.normalize(x1, dim=1)
        logits_softmax = nn.functional.softmax(logits, dim=1)
        preds = logits_softmax.argmax(dim=1)
        mask_true = (preds==labels)
        # # method1: current scene prototype
        # feas_true_mean = []
        # for c in range(self.num_classes):
        #     mask_true_c = mask_true & (labels==c)
        #     if mask_true_c.sum() > 0:
        #         feas_true_mean.append(torch.cat([feas_norm[mask_true_c, :].mean(0), \
        #             torch.tensor([c], device=feas_norm.device)]).unsqueeze(0))
        # feas_true_mean = torch.cat(feas_true_mean, dim=0)
        # method2: all scenes prototype
        self._ema(torch.cat([feas_norm.detach()[mask_true, :], labels[mask_true].unsqueeze(1)], dim=1))
        self.prior_ema = nn.functional.normalize(self.prior_ema, dim=1)
        return logits, feas_norm, self.prior_ema


@MODELS.register_module()
class PTSegV2_Balance_Prior(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes=13,
                 patch_embed_depth=2,
                 patch_embed_channels=48,
                 patch_embed_groups=6,
                 patch_embed_neighbours=16,
                 enc_depths=(2, 6, 2),
                 enc_channels=(96, 192, 384),
                 enc_groups=(12, 24, 48),
                 enc_neighbours=(16, 16, 16),
                 grid_sizes=(0.1, 0.2, 0.4),
                 attn_qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.3,
                 enable_checkpoint=False,
                 beta=0.999,
                 **kwargs):
        super(PTSegV2_Balance_Prior, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = len(enc_depths)
        self.beta = beta
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_groups)
        assert self.num_stages == len(enc_neighbours)
        assert self.num_stages == len(grid_sizes)
        self.patch_embed = GVAPatchEmbed(
            in_channels=in_channels,
            embed_channels=patch_embed_channels,
            groups=patch_embed_groups,
            depth=patch_embed_depth,
            neighbours=patch_embed_neighbours,
            qkv_bias=attn_qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            enable_checkpoint=enable_checkpoint
        )

        enc_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))]
        enc_channels = [patch_embed_channels] + list(enc_channels)
        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(self.num_stages):
            enc = Encoder(
                depth=enc_depths[i],
                in_channels=enc_channels[i],
                embed_channels=enc_channels[i + 1],
                groups=enc_groups[i],
                grid_size=grid_sizes[i],
                neighbours=enc_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=enc_dp_rates[sum(enc_depths[:i]):sum(enc_depths[:i + 1])],
                enable_checkpoint=enable_checkpoint
            )
            self.enc_stages.append(enc)
        self.projection = nn.Sequential(nn.Linear(enc_channels[-1], enc_channels[-2]), nn.BatchNorm1d(enc_channels[-2]), nn.ReLU(inplace=True),
                                        nn.Linear(enc_channels[-2], 48), nn.BatchNorm1d(48), nn.ReLU(inplace=True))
        # ema
        self.register_buffer('prior_ema', torch.rand(num_classes, 48))
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

    def forward(self, p0, x0=None, o0=None, is_train=False, mask=None, ignore_index=None):
        # p, x, o: points, features, batches 
        # The dataloader input here is different from PointTransformer source code; 
        # it's need to modify input data{'pos':, 'x':, 'label':, } to the following form.
        if x0 is None:  # this means p0 is a dict.
            p0, x0, o0, labels = p0['pos'], p0.get('x', None), p0.get('offset', None), p0.get('y', None)
            if x0 is None:
                x0 = p0
            if o0 == None:
                o0, count = [], 0
                for _ in range(p0.size()[0]):
                    count += p0.size()[1]
                    o0.append(count)
                o0 = torch.IntTensor(o0).cuda(device=p0.device)
            if len(x0.size())>2: # means x0:(b, c ,n), need to be catted to (b*n, c)
                x0 = x0.transpose(1,2).contiguous() # po(b, n, 3), x0(b, n, c=3)
                p0 = torch.cat([p0_split.squeeze() for p0_split in p0.split(1,0)])
                x0 = torch.cat([x0_split.squeeze() for x0_split in x0.split(1,0)]) 
                if is_train:
                    labels = torch.cat([labels.squeeze() for labels in labels.split(1,0)]) 
        if x0.size(1)<6:
            x0 = torch.cat((p0,x0),1) # x0(n, c=in_channels+3)
                
        if mask is not None:
            p0, x0, labels = p0[mask], x0[mask], labels[mask]
            new_o0 = []
            for i in range(o0.size(0)):
                new_o0.append(mask[:(o0[i])].sum())
            o0 = torch.tensor(new_o0, dtype=o0.dtype, device=o0.device)
            if ignore_index == 0:
                labels = labels - 1
        
        # aggregate the same class feature in a example of a batch
        feat, coord, offset = [], [], []
        class_index_batch = [] # record the class index in a example of a batch
        for b in range(o0.size(0)):
            class_index = [] # record the class index in a example
            for c in range(self.num_classes):
                index = (labels==c).nonzero().squeeze(1)
                if b == 0:
                    point_mask = index<o0[b]
                else:
                    point_mask = (o0[b-1]<=index) & (index<o0[b])
                point_num = point_mask.sum().item()
                if point_num != 0:
                    # NOTE: When the number of points in the point cloud set is less than the sampling rate, 
                    # fill it up to the sampling rate (256)
                    if point_num >= 256:
                        feat.append(x0[index[point_mask], :])
                        coord.append(p0[index[point_mask], :])
                        offset.append(point_num)
                    elif (point_num > 0) and (point_num < 256):
                        select_index = torch.randint(0, point_num, (256,))
                        select_index = index[point_mask][select_index]
                        feat.append(x0[select_index, :])
                        coord.append(p0[select_index, :])
                        offset.append(torch.tensor(256))
                    class_index.append(c)     
            class_index_batch.append(class_index)       
        feat = torch.cat(feat, dim=0)
        coord = torch.cat(coord, dim=0)
        offset = torch.cumsum(torch.IntTensor(offset), dim=0, dtype=torch.int32).cuda(device=p0.device)

        # a batch of point cloud is a list of coord, feat and offset
        points = [coord, feat, offset]
        points = self.patch_embed(points)
        skips = [[points]]
        for i in range(self.num_stages):
            points, cluster = self.enc_stages[i](points)
            skips[-1].append(cluster)  # record grid cluster of pooling
            skips.append([points])  # record points info of current stage
        coord, feat, offset = skips[-1][0][0], skips[-1][0][1], skips[-1][0][2]  # unpooling feature info in the last enc stage
        feat = self.projection(feat)

        feat = nn.functional.normalize(feat, dim=1)
        current_prior = torch.cat([feat, torch.ones([feat.size(0),1], device=feat.device)*255], dim=1)
        class_index_len = 0
        for i in range(o0.size(0)):
            for j, c_index in enumerate(class_index_batch[i]):
                begin = offset[class_index_len-1] if class_index_len-1 >= 0 else 0
                end = offset[class_index_len]
                current_prior[begin:end, -1] = c_index
                class_index_len += 1
        self._ema(current_prior.detach())
        self.prior_ema = nn.functional.normalize(self.prior_ema, dim=1)
        return current_prior, self.prior_ema


@MODELS.register_module()
class PTSegV2_Balance_Main(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes=13,
                 patch_embed_depth=2,
                 patch_embed_channels=48,
                 patch_embed_groups=6,
                 patch_embed_neighbours=16,
                 enc_depths=(2, 6, 2),
                 enc_channels=(96, 192, 384),
                 enc_groups=(12, 24, 48),
                 enc_neighbours=(16, 16, 16),
                 dec_depths=(1, 1, 1),
                 dec_channels=(48, 96, 192),
                 dec_groups=(6, 12, 24),
                 dec_neighbours=(16, 16, 16),
                 grid_sizes=(0.1, 0.2, 0.4),
                 attn_qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.3,
                 enable_checkpoint=False,
                 unpool_backend="interp",
                 beta=0.999,
                 **kwargs):
        super(PTSegV2_Balance_Main, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = len(enc_depths)
        self.beta = beta
        assert self.num_stages == len(dec_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(dec_channels)
        assert self.num_stages == len(enc_groups)
        assert self.num_stages == len(dec_groups)
        assert self.num_stages == len(enc_neighbours)
        assert self.num_stages == len(dec_neighbours)
        assert self.num_stages == len(grid_sizes)
        self.patch_embed = GVAPatchEmbed(
            in_channels=in_channels,
            embed_channels=patch_embed_channels,
            groups=patch_embed_groups,
            depth=patch_embed_depth,
            neighbours=patch_embed_neighbours,
            qkv_bias=attn_qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            enable_checkpoint=enable_checkpoint
        )

        enc_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))]
        dec_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(dec_depths))]
        enc_channels = [patch_embed_channels] + list(enc_channels)
        dec_channels = list(dec_channels) + [enc_channels[-1]]
        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(self.num_stages):
            enc = Encoder(
                depth=enc_depths[i],
                in_channels=enc_channels[i],
                embed_channels=enc_channels[i + 1],
                groups=enc_groups[i],
                grid_size=grid_sizes[i],
                neighbours=enc_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=enc_dp_rates[sum(enc_depths[:i]):sum(enc_depths[:i + 1])],
                enable_checkpoint=enable_checkpoint
            )
            dec = Decoder(
                depth=dec_depths[i],
                in_channels=dec_channels[i + 1],
                skip_channels=enc_channels[i],
                embed_channels=dec_channels[i],
                groups=dec_groups[i],
                neighbours=dec_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dec_dp_rates[sum(dec_depths[:i]):sum(dec_depths[:i + 1])],
                enable_checkpoint=enable_checkpoint,
                unpool_backend=unpool_backend
            )
            self.enc_stages.append(enc)
            self.dec_stages.append(dec)
        self.seg_head = nn.Sequential(
            nn.Linear(dec_channels[0], dec_channels[0]),
            PointBatchNorm(dec_channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(dec_channels[0], num_classes)
        ) if num_classes > 0 else nn.Identity()

        # ema
        self.register_buffer('prior_ema', torch.rand(num_classes, dec_channels[0]))
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

    def forward(self, p0, x0=None, o0=None, is_train=False, mask=None, ignore_index=None):
        # p, x, o: points, features, batches 
        # The dataloader input here is different from PointTransformer source code; 
        # it's need to modify input data{'pos':, 'x':, 'label':, } to the following form.
        if x0 is None:  # this means p0 is a dict.
            p0, x0, o0, labels = p0['pos'], p0.get('x', None), p0.get('offset', None), p0.get('y', None)
            if x0 is None: x0 = p0
            if o0 == None:
                o0, count = [], 0
                for _ in range(p0.size()[0]):
                    count += p0.size()[1]
                    o0.append(count)
                o0 = torch.IntTensor(o0).cuda(device=p0.device)
            if len(x0.size())>2: # means x0:(b, c ,n), need to be catted to (b*n, c)
                x0 = x0.transpose(1,2).contiguous() # po(b, n, 3), x0(b, n, c=3)
                p0 = torch.cat([p0_split.squeeze() for p0_split in p0.split(1,0)])
                x0 = torch.cat([x0_split.squeeze() for x0_split in x0.split(1,0)]) 
                if is_train:
                    labels = torch.cat([labels.squeeze() for labels in labels.split(1,0)]) 
        if x0.size(1)<6:
            x0 = torch.cat((p0,x0),1) # x0(n, c=in_channels+3)
        coord, feat, offset = p0, x0, o0

        # a batch of point cloud is a list of coord, feat and offset
        points = [coord, feat, offset]
        points = self.patch_embed(points)
        skips = [[points]]
        for i in range(self.num_stages):
            points, cluster = self.enc_stages[i](points)
            skips[-1].append(cluster)  # record grid cluster of pooling
            skips.append([points])  # record points info of current stage

        points = skips.pop(-1)[0]  # unpooling points info in the last enc stage
        for i in reversed(range(self.num_stages)):
            skip_points, cluster = skips.pop(-1)
            points = self.dec_stages[i](points, skip_points, cluster)
        coord, feat, offset = points
        seg_logits_src = self.seg_head(feat)
        if not is_train: return seg_logits_src

        feas_norm = nn.functional.normalize(feat, dim=1)
        if mask is not None:
            labels = labels[mask]
            if ignore_index == 0:
                labels = labels - 1
            seg_logits, feas_norm = seg_logits_src[mask, :], feas_norm[mask, :]
        logits_softmax = nn.functional.softmax(seg_logits, dim=1)
        preds = logits_softmax.argmax(dim=1)
        mask_true = (preds==labels)
        # # method1: current scene prototype
        # feas_true_mean = []
        # for c in range(self.num_classes):
        #     mask_true_c = mask_true & (labels==c)
        #     if mask_true_c.sum() > 0:
        #         feas_true_mean.append(torch.cat([feas_norm[mask_true_c, :].mean(0), \
        #             torch.tensor([c], device=feas_norm.device)]).unsqueeze(0))
        # feas_true_mean = torch.cat(feas_true_mean, dim=0)
        # method2: all scenes prototype
        self._ema(torch.cat([feas_norm.detach()[mask_true, :], labels[mask_true].unsqueeze(1)], dim=1))
        self.prior_ema = nn.functional.normalize(self.prior_ema, dim=1)
        return seg_logits_src, feas_norm, self.prior_ema
