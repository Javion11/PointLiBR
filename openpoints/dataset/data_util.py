from statistics import median
import numpy as np
import torch
import os
import os.path as osp
import ssl
import sys
import urllib
from typing import Optional
import openpoints.cpp.subsampling.grid_subsampling as cpp_subsampling


# download
def download_url(url: str, folder: str, log: bool = True,
                 filename: Optional[str] = None):
    r"""Downloads the content of an URL to a specific folder. 
    Borrowed from https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/download.py
    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    if filename is None:
        filename = url.rpartition('/')[2]
        filename = filename if filename[0] == '?' else filename.split('?')[0]

    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path

    if log:
        print(f'Downloading {url}', file=sys.stderr)

    os.makedirs(folder, exist_ok=True)
    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path


def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert arr.ndim == 2
    arr = arr.copy()
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def voxelize(coord, voxel_size=0.05, hash_type='fnv', mode=0):
    discrete_coord = np.floor(coord / np.array(voxel_size))
    if hash_type == 'ravel':
        key = ravel_hash_vec(discrete_coord)
    else:
        key = fnv_hash_vec(discrete_coord)

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, count = np.unique(key_sort, return_counts=True)
    if mode == 0:  # train mode
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
        idx_unique = idx_sort[idx_select]
        return idx_unique
    else:  # val mode
        return idx_sort, count


def crop_pc(coord, feat, label, normal=None, split='train', 
            voxel_size=0.04, voxel_max=None, 
            random=False, downsample=True, variable=True, shuffle=True):
    crop_idx = None
    if voxel_size and downsample:
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, normal, label = coord[uniq_idx], feat[uniq_idx] if feat is not None else None, \
                                      normal[uniq_idx] if normal is not None else None, label[uniq_idx]
    N = len(label)  # the number of points    
    if voxel_max is not None:
        if N >= voxel_max:
            if not random:
                init_idx = np.random.randint(N) if 'train' in split else N // 2
                crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
            else:
                crop_idx = np.random.choice(N, voxel_max)
        elif not variable:
            # fill more points for non-variable case (batched data) 
            cur_num_points = N
            query_inds = np.arange(cur_num_points)
            padding_choice = np.random.choice(cur_num_points, voxel_max - cur_num_points)
            crop_idx = np.hstack([query_inds, query_inds[padding_choice]])
        crop_idx = np.arange(coord.shape[0]) if crop_idx is None else crop_idx
        if shuffle: 
            shuffle_choice = np.random.permutation(np.arange(len(crop_idx)))
            crop_idx = crop_idx[shuffle_choice]
        coord, feat, normal, label = coord[crop_idx], feat[crop_idx] if feat is not None else None,\
                                      normal[crop_idx] if normal is not None else None, label[crop_idx]
    return coord, feat, label, normal


def get_scene_seg_features(input_features_dim, 
                           points, 
                           feats=None, 
                           use_colors_not_points=True,
                           feature_mode=None
                           ):
    """_summary_
    """
    if feature_mode == None:
        if input_features_dim == 1:
            features = feats[:, -2:-1] 
        elif input_features_dim == 3:
            features = feats[:, :3] if use_colors_not_points else points.transpose(1, 2)
        elif 3<input_features_dim<5:
            if input_features_dim <= feats.size(1):
                features = feats[:, :input_features_dim]
            else:
                points = points.transpose(1, 2) if len(points.shape)>2 else points
                features = torch.cat([feats, points[:, (6-input_features_dim):3]], 1)
        else:
            points = points.transpose(1, 2) if len(points.shape)>2 else points
            features = torch.cat([points, feats[:, :input_features_dim-3]], 1)
    else:
        if feature_mode == 'xyzrgbn':
            features = torch.cat([points, feats], axis=1)
        elif feature_mode == 'xyzrgbz':
            features = torch.cat([points, feats[:, 0:3], points[:, 2:]], axis=1)
        elif feature_mode == 'xyzrgb':
            features = torch.cat([points, feats[:, 0:3]], axis=1)
        elif feature_mode == 'xyz':
            features = points
        elif feature_mode == 'rgbz':
            features = torch.cat([feats[:, 0:3], points[:, 2:]], axis=1)
        elif feature_mode == 'rgb':
            features = feats[:, 0:3]
        elif feature_mode == 'n':
            features = feats[:, 3:]
        elif feature_mode == 'rgbn':
            features = feats
        elif feature_mode == 'xyzn':
            features = torch.cat([points, feats[:, 3:]], axis=1)
        else:
            features = feats
    return features.contiguous()


def get_class_weights(num_per_class, normalize=False, type='sum'):
    if type=='sum':
        weight = num_per_class / float(sum(num_per_class))
    elif type=='median':
        weight = num_per_class / float(median(num_per_class))
    ce_label_weight = 1 / (weight + 0.02)
    
    if normalize:
        ce_label_weight = (ce_label_weight * len(ce_label_weight)) / ce_label_weight.sum() 
    return torch.from_numpy(ce_label_weight.astype(np.float32))


def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """
    if (features is None) and (labels is None):
        return cpp_subsampling.compute(points, sampleDl=sampleDl, verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.compute(points, features=features, sampleDl=sampleDl, verbose=verbose)
    elif (features is None):
        return cpp_subsampling.compute(points, classes=labels, sampleDl=sampleDl, verbose=verbose)
    else:
        return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=sampleDl, verbose=verbose)