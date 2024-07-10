import os, csv, time, numpy as np
from pickle import LIST
from tqdm import tqdm
import logging
import torch
from torch import distributed as dist
from torch_scatter import scatter
from openpoints.utils import set_random_seed
from openpoints.utils import ConfusionMatrix, get_mious
from openpoints.dataset import get_scene_seg_features
from openpoints.dataset.data_util import voxelize
from openpoints.transforms import build_transforms_from_cfg


def write_to_csv(oa, macc, miou, ious, best_epoch, cfg, write_header=True, area=5):
    ious_table = [f'{item:.2f}' for item in ious]
    # NOTE: Because multiprocess will cause "wandb.run.get_url()" error, check weather "cfg.world_size==1" to decide write wandb link.
    # header = ['method', 'Area', 'OA', 'mACC', 'mIoU'] + cfg.classes + ['best_epoch', 'log_path', 'wandb link']
    # data = [cfg.cfg_basename, str(area), f'{oa:.2f}', f'{macc:.2f}', f'{miou:.2f}'] + ious_table + \
    #     [str(best_epoch), cfg.run_dir, wandb.run.get_url() if cfg.wandb.use_wandb and cfg.rank==0 else '-']
    header = ['method', 'Area', 'OA', 'mACC', 'mIoU'] + cfg.classes + ['best_epoch', 'log_path']
    data = [cfg.cfg_basename, str(area), f'{oa:.2f}', f'{macc:.2f}', f'{miou:.2f}'] + ious_table + \
        [str(best_epoch), cfg.run_dir]
    with open(cfg.csv_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)
        f.close()

@torch.no_grad()
def test_s3dis(model, area, cfg, global_cm=None):
    """using a part of original point cloud as input to save memory.
    Args:
        model (_type_): _description_
        test_loader (_type_): _description_
        cfg (_type_): _description_
        global_cm (_type_, optional): _description_. Defaults to None.
        num_votes (int, optional): _description_. Defaults to 1.
    Returns:
        _type_: _description_
    """
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    global_cm =  ConfusionMatrix(num_classes=cfg.num_classes) if global_cm is None else global_cm
    set_random_seed(0)
    cfg.visualize = cfg.get('visualize', False)
        
    # data
    trans_split = 'val' if cfg.datatransforms.get('test', None) is None else 'test'
    transform =  build_transforms_from_cfg(trans_split, cfg.datatransforms)

    raw_root = os.path.join(cfg.dataset.common.data_root, 'raw')
    data_list = sorted(os.listdir(raw_root))
    data_list = [item[:-4] for item in data_list if 'Area_' in item]
    data_list = [item for item in data_list if 'Area_{}'.format(area) in item]

    voxel_size =  cfg.dataset.common.voxel_size
    for cloud_idx, item in enumerate(tqdm(data_list)):
        data_path = os.path.join(raw_root, item + '.npy')
        cdata = np.load(data_path).astype(np.float32)  # xyz, rgb, label, N*7
        coord_min = np.min(cdata[:, :3], 0)
        cdata[:, :3] -= coord_min
        label = torch.from_numpy(cdata[:, 6].astype(np.int).squeeze()).cuda(non_blocking=True)
        colors = np.clip(cdata[:, 3:6] / 255., 0, 1).astype(np.float32)

        all_logits, all_point_inds = [], []
        if voxel_size is not None:
            uniq_idx, count = voxelize(cdata[:, :3], voxel_size, mode=1)
            for i in range(count.max()):
                idx_select = np.cumsum(
                    np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = uniq_idx[idx_select]
                np.random.shuffle(idx_part)
                all_point_inds.append(idx_part)
                coord, feat = cdata[idx_part][:,0:3] - np.min(cdata[idx_part][:, :3], 0), cdata[idx_part][:, 3:6]

                data = {'pos': coord, 'x': feat}
                if transform is not None:
                    data = transform(data)
                if 'heights' in data.keys():
                    data['x'] = torch.cat((data['x'], data['heights']), dim=1)
                else:
                    data['x'] = torch.cat((data['x'], torch.from_numpy(
                        coord[:, 3-cfg.dataset.common.get('n_shifted', 1):3].astype(np.float32))), dim=-1)

                if not cfg.dataset.common.get('variable', False):
                    data['x'] = data['x'].transpose(1, 0).unsqueeze(0)
                    data['pos'] = data['pos'].unsqueeze(0)
                else:
                    data['o'] = torch.IntTensor([len(coord)])

                keys = data.keys() if callable(data.keys) else data.keys
                for key in keys:
                    data[key] = data[key].cuda(non_blocking=True)
                if 'student_model' in cfg.keys(): # when kd, need to use student model cfg
                    in_channels = cfg.student_model.in_channels if 'in_channels' in cfg.student_model.keys() \
                        else cfg.student_model.encoder_args.in_channels
                else:
                    in_channels = cfg.model.in_channels if 'in_channels' in cfg.model.keys() else cfg.model.encoder_args.in_channels
                data['x'] = get_scene_seg_features(in_channels, data['pos'], data['x'])

                logits, _, _, _ = model(data)

                if isinstance(logits, list):
                    logits = logits[-1]
                all_logits.append(logits)

            all_logits = torch.cat(all_logits, dim=0)
            if not cfg.dataset.common.get('variable', False) and len(all_logits.size())>2:
                all_logits = all_logits.transpose(1, 2).reshape(-1, cfg.num_classes)
            all_point_inds = torch.from_numpy(np.hstack(all_point_inds)).cuda(non_blocking=True)

            # project voxel subsampled to original set
            all_logits = scatter(all_logits, all_point_inds, dim=0, reduce='mean')
            all_point_inds = scatter(all_point_inds, all_point_inds, dim=0, reduce='mean')

            cm.update(all_logits.argmax(dim=1), label)
            global_cm.update(all_logits.argmax(dim=1), label)
    
    tp, union, count = cm.tp, cm.union, cm.count
    miou, macc, oa, ious, accs = get_mious(tp, union, count)
    return miou, macc, oa, ious, accs, global_cm
