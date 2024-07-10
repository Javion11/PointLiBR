import random
from functools import partial
from collections.abc import Mapping, Sequence
from torch.utils.data.dataloader import default_collate
import numpy as np
import torch
from easydict import EasyDict as edict
from openpoints.utils import registry
# from torch_geometric.loader.dataloader import Collater as Collater_PyG
# pyg_variable_concat = Collater_PyG(None, None)
from openpoints.transforms import build_transforms_from_cfg


DATASETS = registry.Registry('dataset')

def collate_fn(batch):
        """
        collate function for point cloud which support dict and list,
        'coord' is necessary to determine 'offset'
        """
        if not isinstance(batch, Sequence):
            raise TypeError(f"{batch.dtype} is not supported.")

        if isinstance(batch[0], torch.Tensor):
            return torch.cat(list(batch))
        elif isinstance(batch[0], str):
            # str is also a kind of Sequence, judgement should before Sequence
            return list(batch)
        elif isinstance(batch[0], Sequence):
            for data in batch:
                data.append(torch.tensor([data[0].shape[0]]))
            batch = [collate_fn(samples) for samples in zip(*batch)]
            batch[-1] = torch.cumsum(batch[-1], dim=0).int()
            return batch
        elif isinstance(batch[0], Mapping):
            batch = {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
            for key in batch.keys():
                if "offset" in key:
                    batch[key] = torch.cumsum(batch[key], dim=0)
            return batch
        else:
            return default_collate(batch)

def point_collate_fn(batch, mix_prob=0):
    if isinstance(batch[0], Mapping):
        batch = collate_fn(batch)
        if "offset" in batch.keys():
            # Mix3d (https://arxiv.org/pdf/2110.02210.pdf)
            if random.random() < mix_prob:
                batch["offset"] = torch.cat(
                    [batch["offset"][1:-1:2], batch["offset"][-1].unsqueeze(0)], dim=0
                )
    elif isinstance(batch[0], tuple): # hardcode for len(batch[0])==2
        batch_prior = []
        batch_source = []
        for i in range(len(batch)):
            batch_prior.append(batch[i][0])
            batch_source.append(batch[i][1])
        batch_prior = collate_fn(batch_prior)
        batch_source = collate_fn(batch_source)
        return batch_prior, batch_source
    return batch


def build_dataset_from_cfg(cfg, default_args=None):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        cfg (eDICT):
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    return DATASETS.build(cfg, default_args=default_args)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def build_dataloader_from_cfg(batch_size,
                              dataset_cfg=None,
                              dataloader_cfg=None,
                              datatransforms_cfg=None,
                              split='train',
                              distributed=True,
                              dataset=None
                              ):
    if dataset is None:
        if datatransforms_cfg is not None:
            # in case only val or test transforms are provided. 
            if split not in datatransforms_cfg.keys() and split in ['val', 'test']:
                trans_split = 'val'
            else:
                trans_split = split
            data_transform = build_transforms_from_cfg(trans_split, datatransforms_cfg)
        else:
            data_transform = None

        if split not in dataset_cfg.keys() and split in ['val', 'test']:
            dataset_split = 'test' if split == 'val' else 'val'
        else:
            dataset_split = split
        split_cfg = dataset_cfg.get(dataset_split, edict())
        if split_cfg.get('split', None) is None:    # add 'split' in dataset_split_cfg
            split_cfg.split = split
        split_cfg.transform = data_transform
        dataset = build_dataset_from_cfg(dataset_cfg.common, split_cfg)
    
    if split == 'test':
        if 'test_num_workers' in dataloader_cfg.keys():
            num_workers = dataloader_cfg.test_num_workers
        else:
            num_workers = dataloader_cfg.num_workers
    else:
        num_workers = dataloader_cfg.num_workers

    if dataset_cfg.common.get('collate_fn', False):
        flag_collate_fn = True
    elif dataset_cfg.common.get('variable', False) and (dataset_cfg.common.get('collate_fn', None) is None):
        flag_collate_fn = True
    else:
        flag_collate_fn = False
    collate_fn = None
    if flag_collate_fn:
        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        else:
            mix_prob = dataset_cfg.common.get('mix_prob', 0)
            collate_fn = partial(point_collate_fn, mix_prob=mix_prob)

    shuffle = split == 'train'
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 num_workers=int(num_workers),
                                                 worker_init_fn=worker_init_fn,
                                                 drop_last=split == 'train',
                                                 sampler=sampler,
                                                 collate_fn=collate_fn, 
                                                 pin_memory=True
                                                 )
    else:
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 num_workers=int(num_workers),
                                                 worker_init_fn=worker_init_fn,
                                                 drop_last=split == 'train',
                                                 shuffle=shuffle,
                                                 collate_fn=collate_fn,
                                                 pin_memory=True)
    return dataloader