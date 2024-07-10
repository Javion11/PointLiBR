"""Point transformer
Reference: https://github.com/POSTECH-CVLab/point-transformer
Their result: 70.0 mIoU on S3DIS Area 5. 
"""
from functools import partial
from xml.etree.ElementInclude import include
import torch
import torch.nn as nn
import torch_points_kernels as tp

from openpoints.cpp.pointops.functions import pointops
from ..build import MODELS
from ..layers import KPConvLayer, FastBatchNorm1d
from ..layers import furthest_point_sample, ball_query, grouping_operation


class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True),
                                      nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                      nn.Linear(mid_planes, mid_planes // share_planes),
                                      nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                      nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
        x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
        x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)  # (n, nsample, c)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        for i, layer in enumerate(self.linear_p): 
            p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)    # (n, nsample, c)
        w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes,
                                              self.mid_planes).sum(2)  # (n, nsample, c)
        for i, layer in enumerate(self.linear_w): 
            w = layer(w.transpose(1, 2).contiguous()).transpose(1,2).contiguous() if i % 3 == 0 else layer(w)
        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = x_v.shape
        s = self.share_planes
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        return x


class EdgeConvLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.conv = nn.Sequential(nn.Conv2d(in_planes * 2, out_planes, kernel_size=1, bias=False),
                                  nn.BatchNorm2d(out_planes),
                                  nn.LeakyReLU(negative_slope=0.2))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_k = pointops.queryandgroup(self.nsample, p, p, x, None, o, o, use_xyz=True)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        x = x.unsqueeze(1).repeat(1, self.nsample, 1)
        feature = torch.cat((x_k - x, x), dim=2).permute(2, 0, 1).contiguous()
        feature = feature.unsqueeze(0)
        feature = self.conv(feature)
        feature = feature.max(dim=-1, keepdim=False)[0]
        feature = feature.squeeze(0).permute(1, 0).contiguous()
        return feature


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            # print(n_o.device, p.device)
            idx = pointops.furthestsampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)  # (m, nsample, 3+c)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, x, o]


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2 * in_planes, in_planes), nn.BatchNorm1d(in_planes),
                                         nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes),
                                         nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes),
                                         nn.ReLU(inplace=True))

    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
                x_b = x[s_i:e_i, :]
                # cat avg pooling
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1
            p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x


class PointTransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16, res=True, **kwargs):
        super(PointTransformerBlock, self).__init__()
        self.res = res
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        if self.res:
            x += identity
        x = self.relu(x)
        return [p, x, o]


class EdgeConvBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16, mid_res=False):
        super(EdgeConvBlock, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.local_aggr = PointNet2EdgeConvLayer(planes, planes, share_planes, nsample)
        # self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.mid_res = mid_res

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if not self.mid_res:
            identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        if self.mid_res:
            identity = x
        x = self.local_aggr([p, x, o])
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]


class PointNet2EdgeConvLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.nsample = nsample
        # self.linear_q = nn.Linear(in_planes, mid_planes)
        self.conv = nn.Sequential(nn.Conv1d(in_planes + 3, out_planes, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(out_planes),
                                  nn.ReLU(inplace=True))

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_k = pointops.queryandgroup(self.nsample, p, p, x, None, o, o, use_xyz=True).transpose(1, 2).contiguous()
        feature = self.conv(x_k)
        feature = feature.max(dim=-1, keepdim=False)[0]
        return feature


@MODELS.register_module()
class PTSeg(nn.Module):
    def __init__(self,
                 block=PointTransformerBlock,
                 blocks=[2, 3, 4, 6, 3],    # depth
                 width=32,
                 nsample=[8, 16, 16, 16, 16],
                 in_channels=6,
                 num_classes=13,
                 dec_local_aggr=True,
                 mid_res=False,
                 stacked = False,
                 **kwargs):
        super().__init__()
        self.stacked = stacked
        if in_channels<6:
            in_channels = in_channels + 3 # modify in_channels from rgb(z) to xyz+rgb(z)
        self.in_planes, planes = in_channels, [width * 2**i for i in range(len(blocks))]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], nsample

        if isinstance(block, str):
            block = eval(block)
        self.mid_res = mid_res
        self.dec_local_aggr = dec_local_aggr

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

        # dec 5, no interpolation
        self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample[4], True)  # transform p5
        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample[0])  # fusion p2 and p1
        if not self.stacked:
            self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True),
                                    nn.Linear(planes[0], num_classes))

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

    def forward(self, p0, x0=None, o0=None, return_feats=False):
        # p, x, o: points, features, batches 
        # The dataloader input here is different from PointTransformer source code; 
        # it's need to modify input data{'pos':, 'x':, 'label':, } to the following form.
        #NOTE: if PTSeg is used as PTSegStacked Module, the pre-processing for p0,x0,o0 will be setted in PTSegStacked Class
        if not self.stacked:
            if x0 is None:  # this means p0 is a dict.
                p0, x0, o0= p0['pos'], p0.get('x', None), p0.get('offset', None)
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
        if self.stacked:
            x = x1
        else:
            x = self.cls(x1)
        if return_feats:
            return x, x1
        return x


class KPConvSimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, prev_grid_size, sigma=1.0, negative_slope=0.2, bn_momentum=0.02):
        super().__init__()
        self.kpconv = KPConvLayer(in_channels, out_channels, point_influence=prev_grid_size * sigma, add_one=False)
        self.bn = FastBatchNorm1d(out_channels, momentum=bn_momentum)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, xyz, feats, offset):
        # feats: [N, C]
        # xyz: [N, 3]
        # batch: [N,]
        # neighbor_idx: [N, M]
        offset =[offset[0].item()] + [offset[i].item()-offset[i-1].item() for i in range(1, len(offset))]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset)], 0).long().cuda(device=xyz.device)
        sigma = 1.0
        grid_size = 0.04
        max_num_neighbors = 34
        radius = 2.5 * grid_size * sigma
        neighbor_idx = tp.ball_query(radius, max_num_neighbors, xyz, xyz, batch_x=batch, batch_y=batch, mode="partial_dense")[0]
        feats = self.kpconv(xyz, xyz, neighbor_idx, feats)
        feats = self.activation(self.bn(feats))
        return feats

def create_affinity_matrix(support_xyz: torch.Tensor, support_features: torch.Tensor, o0: torch.Tensor,\
    query_xyz: torch.tensor=None, sample_agg: bool=True):
    """
    support_xyz: (b*n, 3)->(b, npoints, 3)
    support_features: (b*n, c)->(b, c, npoints)
    sample_agg: whether need to downsample the point cloud to compute affinity matrix
    """
    support_xyz_list = list(torch.split(support_xyz, o0[0], dim=0))
    for i in range(len(support_xyz_list)):
        support_xyz_list[i] = support_xyz_list[i].unsqueeze(0)
    support_xyz = torch.cat(support_xyz_list, dim=0).contiguous()
    support_features_list = list(torch.split(support_features, o0[0], dim=0))
    for i in range(len(support_features_list)):
        support_features_list[i] = support_features_list[i].unsqueeze(0)
    support_features = torch.cat(support_features_list, dim=0).transpose(1,2).contiguous()
    
    if query_xyz is not None:
        if support_xyz.size() == query_xyz.size():
            sample_agg = False
    if sample_agg:
        # sample stride set to 16; aggregate radius=0.2; local aggregate nsample=64
        stride = 16
        radius = 0.2
        nsample = 64 # there are about 80 samples in a sphere of radius 0.2
        # downsample to fetch the query_xyz as the center point to aggregate local points
        if query_xyz == None:
            idx = furthest_point_sample(support_xyz, support_xyz.shape[1] // stride).long()
            query_xyz = torch.gather(support_xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        # aggregate local feature according to query_xyz
        neighbour_idx = ball_query(radius, nsample, support_xyz, query_xyz)
        # observe the number of points in each neighbour]
        # print(torch.tensor([neighbour_idx.view(-1,neighbour_idx.size(2))[i,:].unique().size() for i in range(neighbour_idx.size(1))], dtype=torch.float).mean())
        grouped_features = grouping_operation(support_features, neighbour_idx) # (B, C, npoint after downsample, nsample)
        
        new_features = torch.mean(grouped_features, dim=-1, keepdim=False) #（b(0), c, npoints）
        new_features = torch.div(new_features, torch.norm(new_features, dim=1, keepdim=True)) # method2: l2 normalization
        affinity_matrix = torch.einsum("...cn,...cm->...nm", [new_features, new_features]) 
        # affinity_matrix = affinity_matrix / new_features.size(0) # method1: div feature dim
        # affinity_matrix = torch.div(affinity_matrix, torch.norm(affinity_matrix, dim=1, keepdim=True)) # method3: affinity_matrix normalization 
    else:
        support_features = torch.div(support_features,torch.norm(support_features, dim=1, keepdim=True)) # method2: l2 normalization
        affinity_matrix = torch.einsum("...cn,...cm->...nm",[support_features, support_features])
        # affinity_matrix = affinity_matrix[0] / support_features.size(1) # method1: div feature dim
        # affinity_matrix = torch.div(affinity_matrix, torch.norm(affinity_matrix, dim=1, keepdim=True)) # method3: affinity_matrix normalization 
    return affinity_matrix, query_xyz

def create_at(features: torch.Tensor, o0: torch.Tensor):
    """
    create attention map
    features: (b*npoints, c) -> (b, c, npoints)
    o0: (b), e.g. tensor(24000, 48000)
    return: (b, npoints)
    """
    features_list = list(torch.split(features, o0[0], dim=0))
    for i in range(len(features_list)):
        features_list[i] = features_list[i].unsqueeze(0)
    features = torch.cat(features_list, dim=0).transpose(1,2)
    # # method 1: div max(point feature mean of squares) 
    # batch_mean = features.pow(2).mean(1) # (b, npoints)
    # batch_max = batch_mean.max(1)[0].unsqueeze(1) # (b, 1)
    # attention = torch.div(batch_mean, batch_max)
    # method 2: l2 normalization
    features = torch.div(features, torch.norm(features,dim=1, keepdim=True))
    attention = torch.mean(features, dim=1)
    # # method3: according to the original paper, NOTE: cause the loss too small
    # batch_sum = features.pow(2).sum(1) 
    # attention = torch.div(batch_sum, torch.norm(batch_sum, dim=1, keepdim=True)) 
    return attention

@MODELS.register_module()
class PTSegStacked(nn.Module):
    def __init__(self, stacked_num, in_channels, width, inter_channels, num_classes=13, stem_layer=False, stem_channels=32, fore=True, **kwargs):
        super(PTSegStacked, self).__init__()
        self.stacked_num = stacked_num
        self.width = width
        self.stem_layer_mode = stem_layer
        self.fore = fore # select the structure of inter base module, True: type1; False: type4(save memory)
        # add a stem_mlp: ReLU(BN(Linear(in_features=7, out_features=32, bias=False)))
        if in_channels<6:
            in_channels = in_channels + 3 # modify in_channels from rgb(z) to xyz+rgb(z)
        self.in_channels = in_channels
        if stem_layer == 'mlp':
            self.stem_layer = nn.Sequential(nn.Linear(in_channels, stem_channels, bias=True))
            self.in_channels = stem_channels
        elif stem_layer == 'kpconv':
            self.stem_layer = KPConvSimpleBlock(self.in_channels, stem_channels, prev_grid_size=0.04, sigma=1)
            self.in_channels = stem_channels
        PTSeg_List, fore_list, intermediate_list, pred_list, feature_merge_list, pred_merge_list = [], [], [], [], [], []
        res_list = [] # residual connection between hourglass modules
        for _ in range(stacked_num):
            PTSeg_List.append(PTSeg(in_channels=self.in_channels, width=self.width, stacked=True))
        for _ in range(stacked_num-1):
            if self.fore:
                fore_list.append(nn.Sequential(nn.Linear(self.width, inter_channels), nn.BatchNorm1d(inter_channels), nn.ReLU(inplace=True)))
            else:
                inter_channels = width
            intermediate_list.append(self._make_inter(PointTransformerBlock, inter_channels))
            pred_list.append(nn.Sequential(nn.Linear(width, width), nn.BatchNorm1d(width), nn.ReLU(inplace=True), nn.Linear(width, num_classes)))
            feature_merge_list.append(nn.Linear(width, self.in_channels))
            pred_merge_list.append(nn.Linear(num_classes, self.in_channels))
            res_list.append(nn.Linear(self.in_channels, self.in_channels))
        self.PTSeg_List = nn.ModuleList(PTSeg_List) # main hourglass module
        if self.fore:
            self.fore_list = nn.ModuleList(fore_list)
        self.intermediate_list = nn.ModuleList(intermediate_list) 
        self.pred_list = nn.ModuleList(pred_list) # intermediate output module
        self.feature_merge_list = nn.ModuleList(feature_merge_list)
        self.pred_merge_list = nn.ModuleList(pred_merge_list)
        self.cls = nn.Sequential(nn.Linear(self.width, self.width), nn.BatchNorm1d(self.width), 
                                nn.ReLU(inplace=True), nn.Linear(self.width, num_classes)) # the final output module
        self.res_list = nn.ModuleList(res_list)
        
    def _make_inter(self, block, planes, share_planes=8, nsample=16):
        layers = []
        if self.fore:
            layers.append(block(planes, planes//2, share_planes, nsample=nsample, res=False))
            layers.append(block(planes//2, planes//4, share_planes, nsample=nsample, res=False))
        else:
            layers.append(block(planes, planes, share_planes, nsample=nsample, res=True))
            layers.append(TransitionDown(planes, planes))
        return nn.Sequential(*layers)


    def forward(self, p0, x0=None, o0=None, query_xyz=None, kd_struct=None, at=None):
        # p, x, o: points, features, batches 
        # The dataloader input here is different from PointTransformer source code; 
        # it's need to modify input data{'pos':, 'x':, 'label':, } to the following form.
        if x0 is None:  # this means p0 is a dict.
            p0, x0, o0= p0['pos'], p0.get('x', None), p0.get('offset', None)
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

        pred_list, affinity_list, query_xyz_list, at_list = [], [], [], []
        if self.stem_layer_mode == 'mlp':
            x = self.stem_layer(x0)
        elif self.stem_layer_mode == 'kpconv': 
            x = self.stem_layer(p0, x0, o0)
        for i in range(self.stacked_num-1):
            # NOTE: insert attention map before each module
            if at:
                if i in at:
                    at_list.append(create_at(x, o0))

            hg = self.PTSeg_List[i](p0, x, o0)
            if self.fore:
                hg_identity = hg
                hg = self.fore_list[i](hg)

            # save the inter-point affinity matrix computed by each module output feature, which will be used for structure kd
            if kd_struct:
                if i in kd_struct:
                    if isinstance(query_xyz,list):
                        affinity_matrix, query_xyz_temp = create_affinity_matrix(p0, hg, o0, query_xyz[kd_struct.index(i)])
                    else:
                        affinity_matrix, query_xyz_temp = create_affinity_matrix(p0, hg, o0, query_xyz)
                    affinity_list.append(affinity_matrix)
                    query_xyz_list.append(query_xyz_temp)
            
            _, hg, _ = self.intermediate_list[i]([p0, hg, o0])
            if self.fore:
                hg = hg + hg_identity
            pred = self.pred_list[i](hg)
            pred_list.append(pred)
            x = self.res_list[i](x) + self.feature_merge_list[i](hg) + self.pred_merge_list[i](pred)
        
        if at:
            if (self.stacked_num-1) in at:
                at_list.append(create_at(x, o0))
        # strategy1: last PTSeg module direct output. NOTE: this is better
        x = self.PTSeg_List[self.stacked_num-1](p0, x, o0)
        if kd_struct:
            if (self.stacked_num-1) in kd_struct:
                affinity_matrix, query_xyz_temp = create_affinity_matrix(p0, hg, o0, query_xyz[-1]) if isinstance(query_xyz,list) \
                    else create_affinity_matrix(p0, hg, o0, query_xyz)
                affinity_list.append(affinity_matrix)
                query_xyz_list.append(query_xyz_temp)
        pred_list.append(self.cls(x))
        # strategy2: last PTSeg module output add the previous module output through 1X1 residual module
        # x = self.res_list[self.stacked_num-1](x) + self.PTSeg_List[self.stacked_num-1](p0, x, o0)
        # pred_list.append(self.cls(x))
        return pred_list, affinity_list, query_xyz_list, at_list
