import copy
import torch
import torch.nn as nn
import logging
from ...utils import get_missing_parameters_message, get_unexpected_parameters_message
from ..build import MODELS, build_model_from_cfg
from .pointnetv2 import PointNetSAModuleMSG
from ..layers import furthest_point_sample, ball_query, grouping_operation
from ..layers import KPConvLayer, FastBatchNorm1d
import torch_points_kernels as tp


def create_affinity_matrix(support_xyz: torch.Tensor, support_features: torch.Tensor, \
    query_xyz: torch.tensor=None, sample_agg: bool=True):
    """
    support_xyz: (b, npoints, 3)
    support_features: (b, c, npoints)
    sample_agg: whether need to downsample the point cloud to compute affinity matrix
    """
    if query_xyz is not None:
        if support_xyz.size() == query_xyz.size():
            sample_agg = False
    if sample_agg:
        # sample stride set to 16; aggregate radius=0.2; local aggregate nsample=128
        stride = 16
        radius = 0.2
        nsample = 64 # 128->64 (there are about 80 samples in a sphere of radius 0.2)
        # downsample to fetch the query_xyz as the center point to aggregate local points
        if query_xyz == None:
            idx = furthest_point_sample(support_xyz, support_xyz.shape[1] // stride).long()
            query_xyz = torch.gather(support_xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        # aggregate local feature according to query_xyz
        neighbour_idx = ball_query(radius, nsample, support_xyz, query_xyz)
        grouped_features = grouping_operation(support_features, neighbour_idx) # (B, C, npoint after downsample, nsample)
       
        # NOTE: Normalize new_features before compute affinity matrix
        new_features = torch.mean(grouped_features, dim=-1, keepdim=False) #（(b), c, npoints）
        new_features = torch.div(new_features,torch.norm(new_features,dim=1, keepdim=True)) # method2: l2 normalization
        affinity_matrix = torch.einsum("...cn,...cm->...nm",[new_features, new_features]) 
        # affinity_matrix = affinity_matrix / new_features.size(0) # method1: div feature dim
    else:
        support_features = torch.div(support_features,torch.norm(support_features,dim=1, keepdim=True)) # method2: l2 normalization
        affinity_matrix = torch.einsum("...cn,...cm->...nm",[support_features, support_features])
        # affinity_matrix = affinity_matrix[0] / support_features.size(1) # method1: div feature dim
    return affinity_matrix, query_xyz

def create_at(features: torch.Tensor):
    """
    create attention map
    features: (b, c, npoints)
    return: (b, npoints)
    """
    # # method 1: div max(point feature mean of squares) 
    # batch_mean = features.pow(2).mean(1) # (b, npoints)
    # batch_max = batch_mean.max(1)[0].unsqueeze(1) # (b, 1)
    # attention = torch.div(batch_mean, batch_max)
    # method 2: l2 normalization
    features = torch.div(features, torch.norm(features,dim=1, keepdim=True))
    attention = torch.mean(features, dim=1)
    return attention


class KPConvSimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, prev_grid_size, sigma=1.0, negative_slope=0.2, bn_momentum=0.02):
        super().__init__()
        self.kpconv = KPConvLayer(in_channels, out_channels, point_influence=prev_grid_size * sigma, add_one=False)
        self.bn = FastBatchNorm1d(out_channels, momentum=bn_momentum)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, xyz, feats):
        # feats: [B, N, C]
        # xyz: [B, N, 3]
        offset = torch.tensor([xyz.size()[1] for _ in range(xyz.size()[0])])
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset)], 0).long().cuda(device=xyz.device)
        xyz = torch.cat([xyz_split.squeeze() for xyz_split in xyz.split(1,0)])  # [B, N, 3]->[B*N, 3]
        feats = torch.cat([feats_split.squeeze() for feats_split in feats.split(1,0)])  # [B, N, 3]->[B*N, C]
        sigma = 1.0
        grid_size = 0.04
        max_num_neighbors = 34
        radius = 2.5 * grid_size * sigma
        neighbor_idx = tp.ball_query(radius, max_num_neighbors, xyz, xyz, batch_x=batch, batch_y=batch, mode="partial_dense")[0]
        feats = self.kpconv(xyz, xyz, neighbor_idx, feats) # input need to be (n, 3), (n,c)
        feats = self.activation(self.bn(feats))
        feats = torch.cat([feats_split.unsqueeze(0) for feats_split in feats.split(offset[0].item(),0)])
        return feats

class Intermediate_Feature(nn.Module):
    def __init__(self, channels, aggr_args, group_args, conv_args, act_args, norm_args):
        super().__init__()
        # this setting is same with the first SA module in pointnet++
        self.SA = PointNetSAModuleMSG(
                    stride=1,
                    radii=[0.1],
                    nsamples=[32],
                    channel_list=[[channels, int(channels/2), int(channels/4)]],
                    aggr_args=aggr_args,
                    group_args=group_args,
                    conv_args=conv_args,
                    norm_args=norm_args,
                    act_args=act_args,
                    sample_method='fps',
                    use_res=False,
                    query_as_support=False
                )
        self.mlp = nn.Sequential(nn.Conv1d(int(channels/4),int(channels/4),kernel_size=1), nn.BatchNorm1d(int(channels/4)), nn.ReLU(inplace=True))
    
    def forward(self, xyz, feature):
        _, feature = self.SA(xyz, feature)
        feature = self.mlp(feature)
        return feature

class Intermediate_Pred(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1), 
            nn.BatchNorm1d(channels), 
            nn.ReLU(inplace=True), 
            nn.Conv1d(channels, num_classes, kernel_size=1)
        )
    def forward(self, in_feature):
        out_pred = self.pred(in_feature)
        return out_pred


@MODELS.register_module()
class StackedSeg(nn.Module):
    def __init__(self,
                 stem_layer=False,
                 stem_oupchannels=None,
                 stacked_num=None,
                 encoder_args=None,
                 decoder_args=None,
                 cls_args=None,
                 **kwargs):
        super().__init__()
        self.stem_layer_mode = stem_layer
        if stem_layer:
            stem_oupchannels = stem_oupchannels if stem_oupchannels is not None else encoder_args.mlps[0][0][0]
            if stem_layer=='mlp':
                self.stem_layer = nn.Linear(encoder_args.in_channels+3, stem_oupchannels)
            elif stem_layer=='kpconv':
                self.stem_layer = KPConvSimpleBlock(encoder_args.in_channels+3, stem_oupchannels, prev_grid_size=0.04, sigma=1)
            elif stem_layer=='sa':
                self.stem_layer = PointNetSAModuleMSG(
                                    stride=1,
                                    radii=[0.1],
                                    nsamples=[32],
                                    channel_list=[[encoder_args.in_channels+3, stem_oupchannels, stem_oupchannels]],
                                    aggr_args=encoder_args.aggr_args,
                                    group_args=encoder_args.group_args,
                                    conv_args=encoder_args.conv_args,
                                    norm_args=encoder_args.norm_args,
                                    act_args=encoder_args.act_args,
                                    sample_method='fps',
                                    use_res=False,
                                    query_as_support=False)
            encoder_args.in_channels = stem_oupchannels
        self.stacked_num = stacked_num
        self.encoder_list, self.decoder_list, self.fore_list, self.fore_res_list, self.intermediate_list, self.pred_list,  \
            self.feature_merge_list, self.pred_merge_list, self.res_list = [],[],[],[],[],[],[],[],[]
        for i in range(stacked_num):
            self.encoder_list.append(build_model_from_cfg(encoder_args))
            if decoder_args is not None:
                decoder_args_merged_with_encoder = copy.deepcopy(encoder_args)
                decoder_args_merged_with_encoder.update(decoder_args)
                decoder_args_merged_with_encoder.encoder_channel_list = self.encoder_list[i].channel_list if hasattr(self.encoder_list[i], 'channel_list') else None
                self.decoder_list.append(build_model_from_cfg(decoder_args_merged_with_encoder))
            else:
                self.decoder_list.append(None)
            try:
                decoder_oupchannels = self.decoder_list[0].out_channels
                num_classes = cls_args.num_classes
            except:
                logging.info("The decoder doesn't have 'out_channels' key or cls_args doesn't have 'num_classes' key, please check here!")
            if i < stacked_num - 1 :
                # self.fore_list.append(nn.Sequential(nn.Conv1d(decoder_oupchannels, decoder_oupchannels, kernel_size=1), \
                #     nn.BatchNorm1d(decoder_oupchannels), nn.ReLU(inplace=True)))
                # self.fore_res_list.append(nn.Conv1d(decoder_oupchannels, int(decoder_oupchannels/4), kernel_size=1))
                self.intermediate_list.append(Intermediate_Feature(decoder_oupchannels, encoder_args.aggr_args, \
                    encoder_args.group_args, encoder_args.conv_args, encoder_args.act_args, encoder_args.norm_args))
                # Intermediate_Feature module compress channel from decoder_oupchannels to decoder_oupchannels/4 == 32
                self.pred_list.append(Intermediate_Pred(int(decoder_oupchannels/4), num_classes))
                self.feature_merge_list.append(nn.Conv1d(int(decoder_oupchannels/4), encoder_args.in_channels, kernel_size=1))
                self.pred_merge_list.append(nn.Conv1d(num_classes, encoder_args.in_channels, kernel_size=1))
                self.res_list.append(nn.Conv1d(encoder_args.in_channels, encoder_args.in_channels, kernel_size=1))

        self.encoder_list = nn.ModuleList(self.encoder_list)
        self.decoder_list = nn.ModuleList(self.decoder_list)
        # self.fore_list = nn.ModuleList(self.fore_list)
        # self.fore_res_list = nn.ModuleList(self.fore_res_list)
        self.intermediate_list = nn.ModuleList(self.intermediate_list)
        self.pred_list = nn.ModuleList(self.pred_list)
        self.feature_merge_list = nn.ModuleList(self.feature_merge_list)
        self.pred_merge_list = nn.ModuleList(self.pred_merge_list)
        self.res_list = nn.ModuleList(self.res_list)

        if cls_args is not None:
            if hasattr(self.decoder_list[0], 'out_channels'):
                decoder_oupchannels = self.decoder_list[0].out_channels
            elif hasattr(self.encoder_list[0], 'out_channels'):
                decoder_oupchannels = self.encoder_list[0].out_channels
            else:
                decoder_oupchannels = cls_args.get('decoder_oupchannels', None)
            cls_args.in_channels = decoder_oupchannels
            self.head = build_model_from_cfg(cls_args)
        else:
            self.head = None

    def forward(self, p0, f0=None, query_xyz=None, kd_struct=None, at=None, st_keep=False): #(b, n, 3); (b, c(4), n); 
        """
        query_xyz: input student net to assist computing affinity matrix according to the same coordinate of teacher net
        kd_struct: if set as True, will compute affinity matrix list and return it
        st_keep: if set as True, means the student net keep the same structure with pointnet++
        NOTE: kd_struct=True & st_keep=True, the pointnet decoder layer output the affinity matrix and the sample point
        at: if set as True, will compute spacial attention map list and return it
        """
        if hasattr(p0, 'keys'):
            p0, f = p0['pos'], p0['x']
        else:
            if f0 is None:
                f = p0.transpose(1, 2).contiguous()
        if self.stem_layer_mode:
            f = torch.cat((p0, f.transpose(1, 2).contiguous()), dim=-1) # (b, n, 3+c)
        if self.stem_layer_mode == 'mlp':
            f = self.stem_layer(f)
            f = f.transpose(1, 2).contiguous() # (b, n, 32) -> (b, 32, n)
        elif self.stem_layer_mode == 'kpconv':
            f = self.stem_layer(p0, f)
            f = f.transpose(1, 2).contiguous() # (b, n, 32) -> (b, 32, n)
        elif self.stem_layer_mode == 'sa':
            f = f.transpose(1,2).contiguous()
            _, f = self.stem_layer(p0, f)
        
        pred_list, affinity_list, query_xyz_list, at_list = [], [], [], []
        for i in range(self.stacked_num-1):
            # NOTE: insert attention map before each module
            # save the attention map computed by each module output feature(or logits), which will be used for attention transfer
            if at:
                if i in at:
                    at_list.append(create_at(f))
            p, hg = self.encoder_list[i].forward_all_features(p0, f)
            if self.decoder_list[i] is not None:
                hg = self.decoder_list[i](p, hg).squeeze(-1)
            # hg_identity = self.fore_res_list[i](hg)
            hg_identity = hg
            # hg = self.fore_list[i](hg)

            # save the inter-point affinity matrix computed by each module output feature, which will be used for structure kd
            if kd_struct:
                if i in kd_struct:
                    if isinstance(query_xyz,list):
                        affinity_matrix, query_xyz_temp = create_affinity_matrix(p0, hg, query_xyz[kd_struct.index(i)])
                    else:
                        affinity_matrix, query_xyz_temp = create_affinity_matrix(p0, hg, query_xyz)
                    affinity_list.append(affinity_matrix)
                    query_xyz_list.append(query_xyz_temp)
                
            # connect between two module
            hg = self.intermediate_list[i](p0, hg)
            if hg_identity.size(1)==hg.size(1):
                hg = hg + hg_identity
            pred = self.pred_list[i](hg)
            pred_list.append(pred)
            f = self.res_list[i](f) + self.pred_merge_list[i](pred) + self.feature_merge_list[i](hg)

        # NOTE: insert attention map before each module
        if at:
            if (self.stacked_num-1) in at:
                at_list.append(create_at(f))
        p, hg = self.encoder_list[-1].forward_all_features(p0, f)
        # NOTE: original structure or last module processing
        # kd_struct_decoder==True means student net keep the original structure and use structure kd
        kd_struct_decoder = kd_struct and st_keep 
        if self.decoder_list[-1] is not None:
            hg = self.decoder_list[-1](p, hg, kd_struct_decoder)
            if isinstance(hg, list):
                hg = [hg_.squeeze(-1) for hg_ in hg]
            else:
                hg = hg.squeeze(-1)
        if kd_struct_decoder:
            for i in range(len(hg)):
                affinity_matrix, _ = create_affinity_matrix(p[i], hg[i], sample_agg=False)
                affinity_list.append(affinity_matrix)
            query_xyz_list = p
        elif kd_struct and (not st_keep):
            if (self.stacked_num-1) in kd_struct:
                affinity_matrix, query_xyz_temp = create_affinity_matrix(p0, hg, query_xyz[-1]) if isinstance(query_xyz,list) \
                    else create_affinity_matrix(p0, hg, query_xyz)
                affinity_list.append(affinity_matrix)
                query_xyz_list.append(query_xyz_temp)
    
        if self.head is not None:
            if isinstance(hg,list):
                pred_list.append(self.head(hg[0])) 
            else:
                pred_list.append(self.head(hg))
        
        # return a list of each intermediate output
        return pred_list, affinity_list, query_xyz_list, at_list

    def get_loss(self, ret, gt):
        return self.criterion(ret, gt.long())

    def load_model_from_ckpt(self, ckpt_path, only_encoder=False):
        ckpt = torch.load(ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['model'].items()}

        if only_encoder:
            base_ckpt = {k.replace("encoder.", ""): v for k, v in base_ckpt.items()}
            incompatible = self.encoder_list[0].load_state_dict(base_ckpt, strict=False)
        else:
            incompatible = self.load_state_dict(base_ckpt, strict=False)
        if incompatible.missing_keys:
            logging.info('missing_keys')
            logging.info(
                get_missing_parameters_message(incompatible.missing_keys),
            )
        if incompatible.unexpected_keys:
            logging.info('unexpected_keys')
            logging.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
            )
        logging.info(f'Successful Loading the ckpt from {ckpt_path}')

