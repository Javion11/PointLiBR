import torch
from ..utils.registry import Registry

DataTransforms = Registry('datatransforms')


def concat_collate_fn(datas):
    """collate fn for point transformer"""
    pts, feats, labels, offset, count = [], [], [], [], 0
    for data in datas:
        count += len(data['pos'])
        offset.append(count)
        pts.append(data['pos'])
        feats.append(data['x'])
        labels.append(data['y'])
    data = {'pos': torch.cat(pts), 'x': torch.cat(feats), 'y': torch.cat(labels),
            'o': torch.IntTensor(offset)}
    return data


class Compose(object):
    """Composes several transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, args):
        for t in self.transforms:
            if isinstance(args, list):
                for data in args:
                    data = t(data)
            else:
                args = t(args)
        return args


class ListCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, coord, feat, label):
        for t in self.transforms:
            coord, feat, label = t(coord, feat, label)
        return coord, feat, label


def build_transforms_from_cfg(split, datatransforms_cfg):
    """
    Build a dataset transform for a certrain split, defined by `datatransforms_cfg`.
    """
    transform_list = datatransforms_cfg.get(split, None)
    transform_args = datatransforms_cfg.get('kwargs', None)
    compose_fn = eval(datatransforms_cfg.get('compose_fn', 'Compose'))
    if transform_list is None or len(transform_list) == 0:
        return None
    point_transforms = []
    if len(transform_list) > 1:
        for t in transform_list:
            if type(t) == type(''):
                point_transforms.append(DataTransforms.build(
                    {'NAME': t}, default_args=transform_args))
            elif type(t) == type(dict()):
                args_dict = {key: val for key, val in t.items() if key != 'name'}
                point_transforms.append(DataTransforms.build(
                    {'NAME': t['name']}, default_args=args_dict))
        return compose_fn(point_transforms)
    else:
        t = transform_list[0]
        if type(t) == type(''):
            return DataTransforms.build({'NAME': t}, default_args=transform_args)
        elif type(t) == type(dict()):
            args_dict = {key: val for key, val in t.items() if key != 'name'}
            return DataTransforms.build({'NAME': t['name']}, default_args=args_dict)
