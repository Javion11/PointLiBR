"""
Generate visual ply for model input and output
"""
import __init__
import argparse, yaml, os, logging, numpy as np, csv
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist, multiprocessing as mp
import open3d as o3d
from sklearn import manifold
import matplotlib.pyplot as plt
from torch_scatter import scatter
from openpoints.utils import set_random_seed, load_checkpoint, setup_logger_dist, cal_model_parm_nums, generate_exp_directory, \
    resume_exp_directory, EasyConfig, dist_utils, find_free_port
from openpoints.utils import ConfusionMatrix, get_mious
from openpoints.dataset import build_dataloader_from_cfg, get_scene_seg_features
from openpoints.dataset.data_util import voxelize
from openpoints.models import build_model_from_cfg
from openpoints.transforms import build_transforms_from_cfg
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def main(gpu, cfg):
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        torch.cuda.set_device(cfg.rank)
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
    # logger
    if not cfg.debug:
        setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME) # all mode create log
    set_random_seed(0)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels

    #NOTE: create model from cfg file
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_prior = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    # logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    
    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model_prior = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_prior)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        model_prior = nn.parallel.DistributedDataParallel(model_prior.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')

    cfg.batch_size = 1
    cfg.dataset.common.collate_fn = False
    cfg.dataset.train.loop = 1
    cfg.dataset.train.return_name = True
    train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             )
    num_classes = train_loader.dataset.num_classes if hasattr(train_loader.dataset, 'num_classes') else None
    if num_classes is not None:
        assert cfg.num_classes == num_classes
    logging.info(f"number of classes of the dataset: {num_classes}")
    cfg.classes = train_loader.dataset.classes if hasattr(train_loader.dataset, 'classes') else np.range(num_classes)
    cfg.cmap = np.array(train_loader.dataset.cmap) if hasattr(train_loader.dataset, 'cmap') else None
    cfg.num_per_class = train_loader.dataset.num_per_class if hasattr(train_loader.dataset, 'num_per_class') else None
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")

    cfg.dataset.val.return_name = True
    cfg.dataset.common.collate_fn = True
    val_loader = build_dataloader_from_cfg(cfg.batch_size,
                                        cfg.dataset,
                                        cfg.dataloader,
                                        datatransforms_cfg=cfg.datatransforms,
                                        split='val',
                                        distributed=cfg.distributed)
    test_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                    cfg.dataset,
                                    cfg.dataloader,
                                    datatransforms_cfg=cfg.datatransforms,
                                    split='test',
                                    distributed=cfg.distributed)
    # load checkpoint
    if cfg.pretrained_path is not None:
        _, _ = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
        # _, _ = load_checkpoint(model_prior, pretrained_path=cfg.model_prior)
    
    cfg.vis_dir = os.path.join(cfg.run_dir, 'visualization')
    os.makedirs(cfg.vis_dir, exist_ok=True)

    test_s3dis_demo(model, cfg.dataset.common.test_area, cfg, cloud_idx_selected='all', save_raw=False, save_class=False, \
        save_rawfeats=False, save_pred=True, save_feats=True) # 'all' / [0,20,40,60]
    return True


@torch.no_grad()
def test_s3dis_demo(model, area, cfg, cloud_idx_selected, save_raw=False, save_class=False, save_rawfeats=False, save_pred=False, save_feats=False):
    """using a part of original point cloud as input to save memory.
    Args:
        model (_type_): _description_
        cfg (_type_): _description_
        cloud_idx_selected(list): the visualization cloud index list
    Returns:
        _type_: _description_
    """
    torch.cuda.empty_cache()
    model.eval() # set model and model_prior to eval mode
    set_random_seed(0)
    cfg.cmap = cfg.cmap.astype(np.float32) / 255.
    voxel_size = cfg.dataset.common.voxel_size
    # data
    trans_split = 'val' if cfg.datatransforms.get('test', None) is None else 'test'
    transform =  build_transforms_from_cfg(trans_split, cfg.datatransforms)

    raw_root = os.path.join(cfg.dataset.common.data_root, 'raw')
    data_list = sorted(os.listdir(raw_root))
    data_list = [item[:-4] for item in data_list if 'Area_' in item]
    data_list = [item for item in data_list if 'Area_{}'.format(area) in item]
    if cloud_idx_selected == 'all': cloud_idx_selected = list(range(len(data_list)))

    cm_all = ConfusionMatrix(num_classes=cfg.num_classes)
    for cloud_idx in cloud_idx_selected:
        cm = ConfusionMatrix(num_classes=cfg.num_classes)
        item = data_list[cloud_idx]
        data_path = os.path.join(raw_root, item + '.npy')
        cdata = np.load(data_path).astype(np.float32)  # xyz, rgb, label, N*7
        coord_min = np.min(cdata[:, :3], 0)
        cdata[:, :3] -= coord_min
        label = torch.from_numpy(cdata[:, 6].astype(int).squeeze()).cuda(non_blocking=True)
        colors = np.clip(cdata[:, 3:6] / 255., 0, 1).astype(np.float32)

        # save raw data visualization ply
        if save_raw:
            gt = label.cpu().numpy().squeeze()
            gt = cfg.cmap[gt, :]
            vis_dir = os.path.join(cfg.vis_dir, "raw")
            os.makedirs(vis_dir, exist_ok=True)
            np.savetxt(os.path.join(vis_dir, item + "_input.txt"), np.concatenate((cdata[:, :3], colors), axis=1))
            pcd =o3d.io.read_point_cloud(os.path.join(vis_dir, item + "_input.txt"), format='xyzrgb')
            o3d.io.write_point_cloud(os.path.join(vis_dir, item + "_input.ply"), pcd)
            np.savetxt(os.path.join(vis_dir, item + "_gt.txt"), np.concatenate((cdata[:, :3], gt),axis=1))
            pcd =o3d.io.read_point_cloud(os.path.join(vis_dir, item + "_gt.txt"), format='xyzrgb')
            o3d.io.write_point_cloud(os.path.join(vis_dir, item + "_gt.ply"), pcd)
        if save_class:
            for index in range(cfg.num_classes):
                mask_class = label.cpu().numpy() == index
                if mask_class.sum() > 0:
                    np.savetxt(os.path.join(vis_dir, item + f"_input_class{index}.txt"), np.concatenate((cdata[:, :3][mask_class, :], colors[mask_class, :]), axis=1))
                    pcd =o3d.io.read_point_cloud(os.path.join(vis_dir, item + f"_input_class{index}.txt"), format='xyzrgb')
                    o3d.io.write_point_cloud(os.path.join(vis_dir, item + f"_input_class{index}.ply"), pcd)
                    np.savetxt(os.path.join(vis_dir, item + f"_gt_class{index}.txt"), np.concatenate((cdata[:, :3][mask_class, :], gt[mask_class, :]), axis=1))
                    pcd =o3d.io.read_point_cloud(os.path.join(vis_dir, item + f"_gt_class{index}.txt"), format='xyzrgb')
                    o3d.io.write_point_cloud(os.path.join(vis_dir, item + f"_gt_class{index}.ply"), pcd)
        if save_rawfeats:
            cfg.feats_dir = os.path.join(cfg.vis_dir, 'raw_feature')
            os.makedirs(cfg.feats_dir, exist_ok=True)
            mask = torch.rand(label.size(0)) < 0.01
            raw_feats = np.concatenate((cdata[:,:3], colors), axis=1)[mask, :]
            target_feats = label[mask].cpu().numpy()
            fig_path = os.path.join(cfg.feats_dir, item + '_' + 'raw' + '.jpg')
            tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
            X_tsne = tsne.fit_transform(raw_feats)
            x_min, x_max = X_tsne.min(0), X_tsne.max(0)
            X_norm = (X_tsne - x_min) / (x_max - x_min)
            plt.figure(figsize=(8, 8), dpi=720)
            colors = cfg.cmap[target_feats, :]
            plt.scatter(X_norm[:, 0], X_norm[:, 1], s=20, c=colors, marker='o')
            plt.xticks([])
            plt.yticks([])
            plt.savefig(fig_path)
            plt.close()

        final_pred, final_feats, all_point_inds = [], [], []
        uniq_idx, count_idx = voxelize(cdata[:, :3], voxel_size, mode=1)
        for i, _ in enumerate(tqdm(range(count_idx.max()))):
            idx_select = np.cumsum(np.insert(count_idx, 0, 0)[0:-1]) + i % count_idx
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
                data['offset'] = torch.IntTensor([len(coord)])
            keys = data.keys() if callable(data.keys) else data.keys
            for key in keys:
                data[key] = data[key].cuda(non_blocking=True)
            in_channels = cfg.model.in_channels if 'in_channels' in cfg.model.keys() else cfg.model.encoder_args.in_channels
            data['x'] = get_scene_seg_features(in_channels, data['pos'], data['x'])

            logits, feats = model(data, return_feats=True)
            if len(logits.size())>2:
                logits = logits.transpose(1, 2).reshape(-1, logits.size(1))
                feats = feats.transpose(1, 2).reshape(-1, feats.size(1))
            final_pred.append(logits)
            final_feats.append(feats)
        
        # model final predication output
        final_pred = torch.cat(final_pred, dim=0).cpu()
        final_feats = torch.cat(final_feats, dim=0).cpu()
        all_point_inds = torch.from_numpy(np.hstack(all_point_inds))
        # project voxel subsampled to original set
        final_pred = scatter(final_pred, all_point_inds, dim=0, reduce='mean')
        final_feats = scatter(final_feats, all_point_inds, dim=0, reduce='mean')
        label = label.cpu()

        cm.update(final_pred.argmax(dim=1), label)
        cm_all.update(final_pred.argmax(dim=1), label)
        tp, union, count = cm.tp, cm.union, cm.count
        miou, _, oa, _, _ = get_mious(tp, union, count)
        pred_metric_path = os.path.join(cfg.vis_dir, 'pred.csv')
        with open(pred_metric_path, 'a', newline = '', encoding = 'utf-8_sig') as f:
            csv_write = csv.writer(f)      
            csv_write.writerow([item, round(miou, 2), round(oa, 2)])
        
        if save_feats:
            cfg.feats_dir = os.path.join(cfg.vis_dir, 'feature')
            os.makedirs(cfg.feats_dir, exist_ok=True)
            mask = torch.rand(final_feats.size(0)) < 0.01
            final_feats = final_feats[mask, :].numpy()
            target_feats = label[mask].cpu().numpy()
            fig_path = os.path.join(cfg.feats_dir, item + '_' + str(round(miou,2)) + '_' + str(round(oa,2)) + '.jpg')
            tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
            X_tsne = tsne.fit_transform(final_feats)
            
            # filter outliers
            x_indices = np.argsort(X_tsne[:, 0])[-20:]
            y_indices = np.argsort(X_tsne[:, 1])[-20:]
            all_indices = np.unique(np.concatenate((x_indices, y_indices)))
            mask = np.ones(X_tsne.shape[0], dtype=bool)
            mask[all_indices] = False
            X_tsne = X_tsne[mask]
            target_feats = target_feats[mask]

            x_min, x_max = X_tsne.min(0), X_tsne.max(0)
            X_norm = (X_tsne - x_min) / (x_max - x_min)
            plt.figure(figsize=(8, 8), dpi=720)
            colors = cfg.cmap[target_feats, :]
            plt.scatter(X_norm[:, 0], X_norm[:, 1], s=20, c=colors, marker='o')
            plt.xticks([])
            plt.yticks([])
            plt.savefig(fig_path)
            plt.close()

        # save pred visualization ply
        if save_pred:
            final_pred = final_pred.argmax(dim=1)
            final_pred = cfg.cmap[final_pred, :]
            vis_dir = os.path.join(cfg.vis_dir, "pred")
            os.makedirs(vis_dir, exist_ok=True)
            np.savetxt(os.path.join(vis_dir, item + "_pred.txt"), np.concatenate((cdata[:, :3], final_pred), axis=1))
            pcd =o3d.io.read_point_cloud(os.path.join(vis_dir, item + "_pred.txt"), format='xyzrgb')
            o3d.io.write_point_cloud(os.path.join(vis_dir, item + "_" + str(round(miou, 2)) + "_" + str(round(oa, 2)) + "_pred.ply"), pcd)
            print(f"{item}/{cloud_idx} visualization has been saved!!!")
            os.remove(os.path.join(vis_dir, item + "_pred.txt"))

    tp_all, union_all, count_all = cm_all.tp, cm_all.union, cm_all.count
    miou_all, _, oa_all, _, _ = get_mious(tp_all, union_all, count_all)
    with open(pred_metric_path, 'a', newline = '', encoding = 'utf-8_sig') as f:
        csv_write = csv.writer(f)      
        csv_write.writerow(["Total", round(miou_all, 2), round(oa_all, 2)])
    print(f"******End****** total_miou:{round(miou_all, 2)}, total_oa:{round(oa_all, 2)}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Scene segmentation training/testing')
    parser.add_argument('--debug', type=bool, default=False, help='setting debug mode to control not create tensorboard')
    parser.add_argument('--cfg', type=str, help='config file')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args()
    args.cfg = ""
    args.profile = True

    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(vars(args))    # overwrite the default arguments in yaml 
    # cfg.pretrained_path = ""

    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2] + "_demo"  # task/dataset+(kd)+(demo) name, \eg s3dis_kd_demo
    cfg.mode = "visualization"
    cfg.cfg_basename = args.cfg.split('.')[-2].split('/')[-1]   # cfg_basename, \eg pointnext-xl 
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.cfg_basename,  # cfg file name
    ]
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)

    if cfg.pretrained_path is not None:
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
    else:
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path
    # rm the checkoutpoint path in the visualization path
    # os.system('rm -rf %s' % (os.path.join(cfg.run_dir, 'checkpoint')))

    # multi processing.
    if cfg.mp:
        port = find_free_port()
        cfg.dist_url = f"tcp://localhost:{port}"
        print('using mp spawn for distributed training')
        mp.spawn(main, nprocs=cfg.world_size, args=(cfg,))
    else:
        main(0, cfg)
