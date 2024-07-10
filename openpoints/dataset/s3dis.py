import os
import copy
import random
import pickle
import logging
import numpy as np
import open3d as o3d
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from .data_util import crop_pc, voxelize
from .build import DATASETS


@DATASETS.register_module()
class S3DIS(Dataset):
    classes = ['ceiling',
               'floor',
               'wall',
               'beam',
               'column',
               'window',
               'door',
               'chair',
               'table',
               'sofa',
               'bookcase',
               'board',
               'clutter']
    num_classes = 13
    # the right statistics need to be voxelized
    num_per_class = np.array([2428309, 2088621, 3487893, 312234, 275787, 276604, 773785, 
                            451155, 650932, 71601, 671821, 160376, 1633573], dtype=np.int32) # voxelized (train)
    # num_per_class = np.array([3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
    #                         650464, 791496, 88727, 1284130, 229758, 2272837], dtype=np.int32) # voxelized (train+test)
    # num_per_class = np.array([37334028, 32206900, 53133563, 4719832, 4145093, 4127868, 10681455, 
    #                         6318085, 7930065, 949299, 9188737, 2457821, 21826246], dtype=np.int32) # not voxelized
    class2color = {'ceiling':     [0, 255, 0],
                   'floor':       [0, 0, 255],
                   'wall':        [0, 255, 255],
                   'beam':        [255, 255, 0],
                   'column':      [255, 0, 255],
                   'window':      [100, 100, 255],
                   'door':        [200, 200, 100],
                   'table':       [170, 120, 200],
                   'chair':       [255, 0, 0],
                   'sofa':        [200, 100, 100],
                   'bookcase':    [10, 200, 100],
                   'board':       [200, 200, 200],
                   'clutter':     [50, 50, 50]}
    cmap = [*class2color.values()]
    """S3DIS dataset, loading the subsampled entire room as input without block/sphere subsampling.
    Args:
        data_root (str, optional): Defaults to 'data/S3DIS'.
        test_area (int, optional): Defaults to 5.
        voxel_size (float, optional): the voxel size for donwampling. Defaults to 0.04.
        voxel_max (_type_, optional): subsample the max number of point per point cloud. Set None to use all points.  Defaults to None.
        split (str, optional): Defaults to 'train'.
        transform (_type_, optional): Defaults to None.
        loop (int, optional): split loops for each epoch. Defaults to 1.
        presample (bool, optional): wheter to downsample each point cloud before training. Set to False to downsample on-the-fly. Defaults to False.
        variable (bool, optional): where to use the original number of points. The number of point per point cloud is variable. Defaults to False.
        n_shifted (int, optional): the number of shifted coordinates to be used. Defaults to 1 to use the height.
        views (bool, optional): whether load two different transform data
        sources (bool, optional): whether load two different scene data
        # with_height (bool, optional): data['x'] whether add "z" coordinate feature; 
        # control by cfg.model.in_channels, if cfg.model.in_channels==3 equivalent to with_height=False
    """
    def __init__(self,
                 data_root: str = 'data/S3DIS',
                 test_area: int = 5,
                 voxel_size: float = 0.04,
                 voxel_max = None,
                 split: str = 'train',
                 transform = None,
                 loop: int = 1,
                 presample: bool = False,
                 variable: bool = False,
                 n_shifted: int = 1,
                 return_name: bool = False,
                 views: bool = False,
                 views_source: bool = False,
                 sources: bool = False,
                 contact: bool = False,
                 contact_minor: bool = False,
                 mix_prob: int = 0,
                 use_normal: bool = False,
                 sample_minor = None,
                 with_height: bool = True,
                 **kwargs
                 ):

        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.loop = \
            split, voxel_size, transform, voxel_max, loop
        self.presample = presample
        self.variable = variable
        self.n_shifted = n_shifted
        self.return_name = return_name
        self.views = views
        self.views_source = views_source
        self.sources = sources
        self.contact = contact
        self.contact_minor = contact_minor
        self.mix_prob = mix_prob
        self.use_normal = use_normal
        self.sample_minor = sample_minor
        self.with_height = with_height

        if self.use_normal:
            raw_root = os.path.join(data_root, 'raw_normal')
        elif self.sample_minor != None:
            raw_root = os.path.join(data_root, f'minor_{self.sample_minor}_3_4_5_6_7_8_9_10_11')
        else:
            raw_root = os.path.join(data_root, 'raw')
        if self.contact_minor: 
            minor_root = os.path.join(data_root, 'minor_3_4_5_9_11')
            self.minor_root = minor_root
            minor_data_list = sorted(os.listdir(minor_root))
            minor_data_list = [item[:-4] for item in minor_data_list if 'Area_' in item]
        self.raw_root = raw_root
        data_list = sorted(os.listdir(raw_root))
        data_list = [item[:-4] for item in data_list if 'Area_' in item]
        
        if split == 'train':
            self.data_list = [item for item in data_list if not 'Area_{}'.format(test_area) in item]
            if self.contact_minor: 
                self.minor_data_list = [item for item in minor_data_list if not 'Area_{}'.format(test_area) in item]
        else:
            self.data_list = [item for item in data_list if 'Area_{}'.format(test_area) in item]
        if self.contact_minor: 
            self.minor_data_idx = np.arange(len(self.minor_data_list))

        processed_root = os.path.join(data_root, 'processed')
        if use_normal:
            filename = os.path.join(processed_root, f's3dis_{split}_area{test_area}_{voxel_size:.3f}_normal.pkl')
        else:
            filename = os.path.join(processed_root, f's3dis_{split}_area{test_area}_{voxel_size:.3f}.pkl')
        if presample and not os.path.exists(filename):
            np.random.seed(0)
            self.data = []
            for item in tqdm(self.data_list, desc=f'Loading S3DISFull {split} split on Test Area {test_area}'):
                data_path = os.path.join(raw_root, item + '.npy')
                cdata = np.load(data_path).astype(np.float32)
                cdata[:, :3] -= np.min(cdata[:, :3], 0)
                if voxel_size:
                    if use_normal:
                        coord, feat, label = cdata[:,0:3], cdata[:, 3:9], cdata[:, 9:10]
                    else:
                        coord, feat, label = cdata[:,0:3], cdata[:, 3:6], cdata[:, 6:7]
                    uniq_idx = voxelize(coord, voxel_size)
                    coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
                    cdata = np.hstack((coord, feat, label))
                self.data.append(cdata)
            npoints = np.array([len(data) for data in self.data])
            logging.info('split: %s, median npoints %.1f, avg num points %.1f, std %.1f' % (
                self.split, np.median(npoints), np.average(npoints), np.std(npoints)))
            os.makedirs(processed_root, exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(self.data, f)
                print(f"{filename} saved successfully")
        elif presample:
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)
                print(f"{filename} load successfully")
        self.data_idx = np.arange(len(self.data_list))
        assert len(self.data_idx) > 0
        logging.info(f"\nTotally {len(self.data_idx)} samples in {split} set")

    def __getitem__(self, idx):
        data_idx = idx % len(self.data_idx)
        # data_prior_idx = random.choice(self.data_idx)
        data_prior_idx = (idx+50) % len(self.data_idx)
        if self.presample:
            if self.use_normal:
                coord, feat, normal, label = np.split(self.data[data_idx], [3, 6, 9], axis=1)
            else:
                coord, feat, label = np.split(self.data[data_idx], [3, 6], axis=1)
        else:
            data_path = os.path.join(self.raw_root, self.data_list[data_idx] + '.npy')
            cdata = np.load(data_path).astype(np.float32)
            cdata[:, :3] -= np.min(cdata[:, :3], 0)
            if self.use_normal:
                coord, feat, normal, label = cdata[:, :3], cdata[:, 3:6], cdata[:, 6:9], cdata[:, 9:10]
            else:
                coord, feat, label = cdata[:, :3], cdata[:, 3:6], cdata[:, 6:7]
                normal = None
            coord, feat, label, normal = crop_pc(
                coord, feat, label, normal, self.split, self.voxel_size, self.voxel_max,
                downsample=not self.presample, variable=self.variable)
            
            if self.sources:
                data_prior_path = os.path.join(self.raw_root, self.data_list[data_prior_idx] + '.npy')
                cdata_prior = np.load(data_prior_path).astype(np.float32)
                cdata_prior[:, :3] -= np.min(cdata_prior[:, :3], 0)
                if self.use_normal:
                    coord_prior, feat_prior, normal_prior, label_prior = cdata_prior[:, :3], cdata_prior[:, 3:6], cdata[:, 6:9], cdata_prior[:, 6:7]
                else:
                    coord_prior, feat_prior, label_prior = cdata_prior[:, :3], cdata_prior[:, 3:6], cdata_prior[:, 6:7]
                    normal_prior = None
                coord_prior, feat_prior, label_prior, normal_prior = crop_pc(
                    coord_prior, feat_prior, label_prior, normal_prior, self.split, self.voxel_size, self.voxel_max,
                    downsample=not self.presample, variable=self.variable)
            if self.contact:
                data_idx_cat = random.choice(self.data_idx)
                data_prior_idx_cat = random.choice(self.data_idx)

                data_path_cat = os.path.join(self.raw_root, self.data_list[data_idx_cat] + '.npy')
                cdata_cat = np.load(data_path_cat).astype(np.float32)
                cdata_cat[:, :3] -= np.min(cdata_cat[:, :3], 0)
                coord_cat, feat_cat, label_cat = cdata_cat[:, :3], cdata_cat[:, 3:6], cdata_cat[:, 6:7]
                coord_cat, feat_cat, label_cat = crop_pc(
                    coord_cat, feat_cat, label_cat, self.split, self.voxel_size, self.voxel_max,
                    downsample=not self.presample, variable=self.variable)
                
                data_prior_path_cat = os.path.join(self.raw_root, self.data_list[data_prior_idx_cat] + '.npy')
                cdata_prior_cat = np.load(data_prior_path_cat).astype(np.float32)
                cdata_prior_cat[:, :3] -= np.min(cdata_prior_cat[:, :3], 0)
                coord_prior_cat, feat_prior_cat, label_prior_cat = cdata_prior_cat[:, :3], cdata_prior_cat[:, 3:6], cdata_prior_cat[:, 6:7]
                coord_prior_cat, feat_prior_cat, label_prior_cat = crop_pc(
                    coord_prior_cat, feat_prior_cat, label_prior_cat, self.split, self.voxel_size, self.voxel_max,
                    downsample=not self.presample, variable=self.variable)
        
        if self.contact_minor:
            minor_data_idx = random.choice(self.minor_data_idx)
            minor_data_path = os.path.join(self.minor_root, self.minor_data_list[minor_data_idx] + '.npy')
            minor_cdata = np.load(minor_data_path).astype(np.float32)
            minor_cdata[:, :3] -= np.min(minor_cdata[:, :3], 0)
            minor_coord, minor_feat, minor_label, minor_normal = minor_cdata[:, :3], minor_cdata[:, 3:6], minor_cdata[:, 6:7], None
            minor_coord, minor_feat, minor_label, minor_normal = crop_pc(
                minor_coord, minor_feat, minor_label, minor_normal, self.split, self.voxel_size, self.voxel_max//4,   # NOTE: minor data size set as 1/4 of voxel_max
                downsample=not self.presample, variable=self.variable)
            minor_label = minor_label.squeeze(-1).astype(np.long)

        label = label.squeeze(-1).astype(np.long)
        if self.use_normal:
            if self.contact_minor: 
                data_src = {'pos': np.concatenate((coord, minor_coord), axis=0), 'x': np.concatenate((feat, minor_feat), axis=0), \
                    'normal': np.concatenate((normal, minor_normal), axis=0), 'y': np.concatenate((label, minor_label), axis=0)}
            else: data_src = {'pos': coord, 'x': feat, 'normal': normal, 'y': label}
        else:
            if self.contact_minor: 
                data_src = {'pos': np.concatenate((coord, minor_coord), axis=0), 'x': np.concatenate((feat, minor_feat), axis=0), \
                    'y': np.concatenate((label, minor_label), axis=0)}
            else: data_src = {'pos': coord, 'x': feat, 'y': label}
        # pre-process
        data = copy.deepcopy(data_src)
        data = self.transform(data)
        if self.with_height:
            data['x'] = torch.cat((data['x'], torch.from_numpy(data_src['pos'][:, 3-self.n_shifted:3])), dim=-1)
        if self.use_normal:
            data['x'] = torch.cat((data['x'], data['normal']), dim=-1)
        
        if self.views:
            data_prior = copy.deepcopy(data_src)
            if self.views_source:
                keys = data_prior.keys() if callable(data_prior.keys) else data_prior.keys
                for key in keys:
                    if not torch.is_tensor(data_prior[key]):
                        data_prior[key] = torch.from_numpy(np.array(data_prior[key]))
                if data_prior['x'][:, :3].max() > 1:
                    data_prior['x'][:, :3] /= 255.
            else:
                data_prior = self.transform(data_prior)
            if self.with_height:
                data_prior['x'] = torch.cat((data_prior['x'], torch.from_numpy(data_src['pos'][:, 3-self.n_shifted:3])), dim=-1)
            if self.use_normal:
                data_prior['x'] = torch.cat((data_prior['x'], data_prior['normal']), dim=-1)
        
        if self.sources:
            label_prior = label_prior.squeeze(-1).astype(np.long)
            data_prior = {'pos': coord_prior, 'x': feat_prior, 'y': label_prior}
            if self.views_source:
                keys = data_prior.keys() if callable(data_prior.keys) else data_prior.keys
                for key in keys:
                    if not torch.is_tensor(data_prior[key]):
                        data_prior[key] = torch.from_numpy(np.array(data_prior[key]))
                if data_prior['x'][:, :3].max() > 1:
                    data_prior['x'][:, :3] /= 255.
            else:
                data_prior = self.transform(data_prior)
            if self.with_height:
                data_prior['x'] = torch.cat((data_prior['x'], torch.from_numpy(coord_prior[:, 3-self.n_shifted:3].astype(np.float32))), dim=-1)
            if self.use_normal:
                data_prior['x'] = torch.cat((data_prior['x'], data_prior['normal']), dim=-1)

        if self.contact:
            label_cat = label_cat.squeeze(-1).astype(np.long)
            data_src_cat = {'pos': coord_cat, 'x': feat_cat, 'y': label_cat}
            data_cat = self.transform(data_src_cat)
            data_cat['x'] = torch.cat((data_cat['x'], torch.from_numpy(
            coord_cat[:, 3-self.n_shifted:3].astype(np.float32))), dim=-1)
            
            label_prior_cat = label_prior_cat.squeeze(-1).astype(np.long)
            data_src_prior_cat = {'pos': coord_prior_cat, 'x': feat_prior_cat, 'y': label_prior_cat}
            data_prior_cat = self.transform(data_src_prior_cat)
            data_prior_cat['x'] = torch.cat((data_prior_cat['x'], torch.from_numpy(
            coord_prior_cat[:, 3-self.n_shifted:3].astype(np.float32))), dim=-1)

            for key in data.keys():
                dim = -1 if key == 'y' else -2
                data[key] = torch.cat((data[key], data_cat[key]), dim=dim)
                data_prior[key] = torch.cat((data_prior[key], data_prior_cat[key]), dim=dim)

        # record data name
        if self.return_name:
            if self.presample:
                name = data_idx 
            else:
                name = self.data_list[data_idx]
            data.update({'name': name})
            if self.sources:
                data_prior.update({'name': self.data_list[data_prior_idx]})
        # return original point color to visualization
        data.update({'color': feat})
            
        # entire scene as input
        if self.variable and (not self.return_name):
            if self.sources:
                return data['pos'], data['x'], data_prior['pos'], data_prior['x'], data['y'], data_prior['y']
            elif self.views:
                return data['pos'], data['x'], data_prior['pos'], data_prior['x'], data['y']
            else:
                return data['pos'], data['x'], data['y']
        elif self.sources or self.views:
            return data, data_prior
        return data

    def __len__(self):
        return len(self.data_idx) * self.loop

    # entire scene as input, scene points number varies, cat batch examples at first dim
    def collate_fn(self, batch): 
        data_len = len(list(zip(*batch)))
        if data_len == 3:
            coord, feat, label = list(zip(*batch))
        elif data_len == 5:
            coord, feat, coord_prior, feat_prior, label = list(zip(*batch))
        elif data_len == 6:
            coord, feat, coord_prior, feat_prior, label, label_prior = list(zip(*batch))
            offset_prior, count_prior = [], 0
            for item in coord_prior:
                count_prior += item.shape[0]
                offset_prior.append(count_prior)
            offset_prior = torch.IntTensor(offset_prior)
        
        offset, count = [], 0
        for item in coord:
            count += item.shape[0]
            offset.append(count)
        offset = torch.IntTensor(offset)
        if random.random() < self.mix_prob:
            offset = torch.cat([offset[1:-1:2], offset[-1].unsqueeze(0)], dim=0) if offset.size(0)>2 else offset[-1].unsqueeze(0)
            if data_len == 6:
                offset_prior = torch.cat([offset_prior[1:-1:2], offset_prior[-1].unsqueeze(0)], dim=0) if offset_prior.size(0)>2 else offset_prior[-1].unsqueeze(0)

        if data_len == 3:
            data = {'pos': torch.cat(coord), 'x': torch.cat(feat), 'y': torch.cat(label), 'offset': offset}
        if data_len == 5:
            data_prior = {'pos': torch.cat(coord_prior), 'x': torch.cat(feat_prior), 'y': torch.cat(label), 'offset': offset}
            data = {'pos': torch.cat(coord), 'x': torch.cat(feat), 'y': torch.cat(label), 'offset': offset}
            return data_prior, data
        elif data_len == 6:
            data_prior = {'pos': torch.cat(coord_prior), 'x': torch.cat(feat_prior), 'y': torch.cat(label_prior), 'offset': offset_prior}
            data = {'pos': torch.cat(coord), 'x': torch.cat(feat), 'y': torch.cat(label), 'offset': offset}
            return data_prior, data
        return data
