"""
SPG: mitigate class imbalance in point cloud semantic segmentation through separate subspace prototypes.
"""
import __init__
import argparse, yaml, os, copy, logging, time, numpy as np
import nni
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist, multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, save_prior, load_prior, \
    setup_logger_dist, cal_model_parm_nums, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from openpoints.dataset import build_dataloader_from_cfg, get_scene_seg_features, get_class_weights
from openpoints.dataset.data_util import voxelize
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
from openpoints.models.layers import furthest_point_sample
from main.test_utils import write_to_csv
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def main(gpu, cfg):
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
    # logger
    if not cfg.debug:
        setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME) # all mode create log
    if cfg.rank == 0 and (not cfg.debug): 
        writer = SummaryWriter(log_dir=os.path.join(cfg.run_dir, 'tensorboard')) if cfg.is_training else None # only training create tensorboard
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    # logging.info(cfg)

    # # get parameters form tuner
    # tuner_params = nni.get_next_parameter()
    # logging.info(tuner_params)
    # cfg.update(tuner_params)

    train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                            cfg.dataset,
                                            cfg.dataloader,
                                            datatransforms_cfg=cfg.datatransforms,
                                            split='train',
                                            distributed=cfg.distributed,
                                            )
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")
    cfg.model.beta = 1.0 - 1.0 / train_loader.__len__()
    cfg.model_prior.beta = 1.0 - 1.0 / train_loader.__len__()

    #NOTE: create model from cfg file
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    # logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    model_prior = build_model_from_cfg(cfg.model_prior).to(cfg.rank)
    model_prior_size = cal_model_parm_nums(model_prior)
    # logging.info(model_prior)
    logging.info('Number of prior branch params: %.4f M' % (model_prior_size / 1e6))

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if cfg.model_prior.NAME != "BaseSeg_Balance_Prior":
            model_prior = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_prior)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank, broadcast_buffers=True)
        model_prior = nn.parallel.DistributedDataParallel(model_prior.cuda(), device_ids=[cfg.rank], output_device=cfg.rank, broadcast_buffers=True)
        logging.info('Using Distributed Data parallel ...')

    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)
    optimizer_prior = build_optimizer_from_cfg(model_prior, lr=cfg.lr, **cfg.optimizer)
    scheduler_prior = build_scheduler_from_cfg(cfg, optimizer_prior)

    # build dataset
    val_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='val',
                                           distributed=cfg.distributed
                                           )
    logging.info(f"length of validation dataset: {len(val_loader.dataset)}")
    num_classes = val_loader.dataset.num_classes if hasattr(val_loader.dataset, 'num_classes') else None
    if num_classes is not None:
        assert cfg.num_classes == num_classes
    logging.info(f"number of classes of the dataset: {num_classes}")
    cfg.classes = val_loader.dataset.classes if hasattr(val_loader.dataset, 'classes') else np.range(num_classes)
    cfg.cmap = np.array(val_loader.dataset.cmap) if hasattr(val_loader.dataset, 'cmap') else None

    if ("semantic3d" in cfg.dataset.common.NAME.lower()) or ("toronto3d" in cfg.dataset.common.NAME.lower()):
        test_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='test',
                                           distributed=cfg.distributed
                                           )
        logging.info(f"length of test dataset: {len(test_loader.dataset)}")
    elif "scannetv2" in cfg.dataset.common.NAME.lower():
        cfg.test_batch_size = 1
        cfg.dataset.common.collate_fn = False # NOTE: train_loader's collate_fn need to be True
        cfg.dataloader.num_workers = 2
        test_loader = build_dataloader_from_cfg(cfg.test_batch_size,
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='test',
                                           distributed=False,
                                           )
        logging.info(f"length of test dataset: {len(test_loader.dataset)}")
    
    # optionally resume from a checkpoint
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler, pretrained_path=cfg.pretrained_path)
            resume_checkpoint(cfg, model_prior, optimizer_prior, scheduler_prior, pretrained_path=cfg.prior_model_path)
            val_miou = validate(model, val_loader, cfg)
            logging.info(f'\nresume val miou is {val_miou}\n ')
            # resume the SummaryWriter instance from the crashed epoch
            if cfg.rank == 0:
                writer = SummaryWriter(log_dir=os.path.join(cfg.run_dir, 'tensorboard'), purge_step=cfg.start_epoch)
        elif cfg.mode == 'test':
            if cfg.rank == 0:
                if "s3dis" in cfg.dataset.common.NAME.lower():
                    best_epoch, best_val = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
                    miou, macc, oa, ious = test_s3dis(model, cfg.dataset.common.test_area, cfg)
                    with np.printoptions(precision=2, suppress=True):
                        logging.info(f'Best ckpt @E{best_epoch},  test_oa , test_macc, test_miou: {oa:.2f} {macc:.2f} {miou:.2f}, '
                                    f'\nEach cls Test IoU: {ious}')
                    cfg.csv_path = os.path.join(cfg.run_dir, cfg.run_name + '_test.csv')
                    write_to_csv(oa, macc, miou, ious, best_epoch, cfg)
                    return miou
            else:
                return True
        else:
            logging.info(f'Finetuning from {cfg.pretrained_path}')
            model.load_model_from_ckpt(cfg.pretrained_path, only_encoder=cfg.only_encoder)
    else:
        logging.info('Training from scratch')
    
    # cfg.criterion.weight = None 
    # if cfg.get('cls_weighed_loss', False):
    #     if hasattr(train_loader.dataset, 'num_per_class'):    
    #         cfg.criterion.weight = get_class_weights(train_loader.dataset.num_per_class, normalize=False)
    #     else: 
    #         logging.info('`num_per_class` attribute is not founded in dataset')
    criterion = build_criterion_from_cfg(cfg.criterion)
    criterion_supcon = build_criterion_from_cfg(cfg.criterion_SupCon)
    criterion_l1 = nn.SmoothL1Loss(reduction='mean')

    # ===> start training
    val_miou, val_macc, val_oa, val_ious = 0., 0., 0., []
    best_val, macc_when_best, oa_when_best, ious_when_best, best_epoch = 0., 0., 0., [], 0
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        # if epoch >2: break # debug
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'epoch'):  # some dataset sets the dataset length as a fixed steps.
            train_loader.dataset.epoch = epoch - 1
        loss_prior, loss_prior_supcon, loss_main_l1, loss_main_ce, train_miou, train_macc = \
            train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, cfg,\
                model_prior, optimizer_prior, scheduler_prior, criterion_supcon, criterion_l1)

        is_best = False
        if epoch % cfg.val_freq == 0:
            val_miou, val_macc, val_oa, val_ious = validate(model, val_loader, cfg)
            if val_miou > best_val:
                is_best = True
                best_val = val_miou
                macc_when_best = val_macc
                oa_when_best = val_oa
                ious_when_best = val_ious
                best_epoch = epoch
                with np.printoptions(precision=2, suppress=True):
                    logging.info(f'Find a better ckpt @E{epoch}, val_miou {val_miou:.2f} val_macc {macc_when_best:.2f}, val_oa {oa_when_best:.2f}')
        with np.printoptions(precision=2, suppress=True):
            logging.info(f'mious: {val_ious}')
        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} LR {lr:.6f} '
                     f'train_miou {train_miou:.2f}, val_miou {val_miou:.2f}, best val miou {best_val:.2f}')
        if writer is not None:
            writer.add_scalar('val/best_val', best_val, epoch)
            writer.add_scalar('val/val_miou', val_miou, epoch)
            writer.add_scalar('val/macc_when_best', macc_when_best, epoch)
            writer.add_scalar('val/oa_when_best', oa_when_best, epoch)
            writer.add_scalar('val/val_macc', val_macc, epoch)
            writer.add_scalar('val/val_oa', val_oa, epoch)
            writer.add_scalar('train/loss_prior', loss_prior, epoch)
            writer.add_scalar('train/loss_prior_supcon', loss_prior_supcon, epoch)
            writer.add_scalar('train/loss_main_l1', loss_main_l1, epoch)
            # writer.add_scalar('train/loss_main_struct', loss_main_struct, epoch)
            writer.add_scalar('train/loss_main_ce', loss_main_ce, epoch)
            writer.add_scalar('train/train_miou', train_miou, epoch)
            writer.add_scalar('train/train_macc', train_macc, epoch)
            writer.add_scalar('lr', lr, epoch)

        if cfg.sched_on_epoch:
            scheduler.step(epoch)
            scheduler_prior.step(epoch)
        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val': best_val},
                            is_best=is_best, post_fix='main_model_ckpt')
            save_checkpoint(cfg, model_prior, epoch, optimizer_prior, scheduler_prior,
                            additioanl_dict={'best_val': best_val},
                            is_best=is_best, post_fix='prior_model_ckpt', is_logging=False)
            is_best = False
            # nni.report_intermediate_result(val_miou)

    # validate
    with np.printoptions(precision=2, suppress=True):
        logging.info(
            f'Best ckpt @E{best_epoch},  val_oa {oa_when_best:.2f}, val_macc {macc_when_best:.2f}, val_miou {best_val:.2f}, '
            f'\nEach cls IoU: {ious_when_best}')
    
    # test
    if cfg.rank == 0: # NOTE: only test on the main process, because in the ddp mode, the test code will be used for repeated times
        if "s3dis" in cfg.dataset.common.NAME.lower():
            load_checkpoint(model, pretrained_path=os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_main_model_ckpt_best.pth'))
            load_checkpoint(model_prior, pretrained_path=os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_prior_model_ckpt_best.pth'))
            cfg.csv_path = os.path.join(cfg.run_dir, cfg.run_name + f'_area5.csv')
            test_miou, test_macc, test_oa, test_ious = test_s3dis(model, cfg.dataset.common.test_area, cfg)
            with np.printoptions(precision=2, suppress=True):
                logging.info(f'Best ckpt @E{best_epoch},  test_oa {test_oa:.2f}, test_macc {test_macc:.2f}, test_miou {test_miou:.2f}, '
                            f'\nEach cls IoU: {test_ious}')
            if writer is not None:
                writer.add_scalar('test_miou', test_miou, epoch)
                writer.add_scalar('test_macc', test_macc, epoch)
                writer.add_scalar('test_oa', test_oa, epoch)
            write_to_csv(test_oa, test_macc, test_miou, test_ious, best_epoch, cfg, write_header=True)
            logging.info(f'save results in {cfg.csv_path}')
            # nni.report_final_result(test_miou)
    return True


def gen_sample(numbers, mu, std=0.05):
    """mu: (class_num, prior_feas_dim)"""
    mu = mu.unsqueeze(0).repeat(numbers, 1, 1)
    eps = torch.randn_like(mu)
    return eps * std + mu

def create_affinity_matrix(xyz: torch.Tensor=None, feats: torch.Tensor=None, offset: torch.Tensor=None, target: torch.Tensor=None, sample: bool=False, size: int=2000):
    """
    xyz: (b*n, 3)->[b, npoints, 3](List len==b)
    features: (b*n, c)->[b, npoints, c](List len==b)
    return: affinity_matrix and according target(optional)
    """
    affinity_matrix = []
    select_target = []
    if sample:
        offset_list = []
        for i in range(offset.size(0)):
            if i>0: offset_list.append(offset[i].item() - offset[i-1].item()) 
            else: offset_list.append(offset[0])
        xyz_list = list(torch.split(xyz, offset_list, dim=0))
        feats_list = torch.split(feats, offset_list, dim=0)
        target_list = torch.split(target, offset_list, dim=0)
        
        for i in range(offset.size(0)):
            xyz_list[i] = xyz_list[i].unsqueeze(0) # (1, npoints, 3)
            idx = furthest_point_sample(xyz_list[i], size).long().squeeze(0)
            select_feats = feats_list[i][idx, :] #（npoints, c）
            # select_feats = torch.div(select_feats, torch.norm(select_feats, dim=1, keepdim=True)) 
            affinity_matrix.append(torch.einsum("...nc,...mc->...nm", [select_feats, select_feats]).unsqueeze(0))
            select_target.append(target_list[i][idx])
        affinity_matrix = torch.cat(affinity_matrix, dim=0)
        select_target = torch.cat(select_target, dim=0)
        return affinity_matrix, select_target
    else:
        feats = torch.split(feats, size, dim=0)
        feats = [feats[i].unsqueeze(0) for i in range(len(feats))]
        feats = torch.cat(feats, dim=0)
        # feats = torch.div(feats, torch.norm(feats, dim=2, keepdim=True)) 
        affinity_matrix = torch.einsum("...nc,...mc->...nm", [feats, feats])
        return affinity_matrix

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, cfg, \
                    model_prior, optimizer_prior, scheduler_prior, criterion_supcon, criterion_l1):
    loss_prior_meter = AverageMeter()
    loss_prior_supcon_meter = AverageMeter()
    loss_main_l1_meter = AverageMeter()
    loss_main_ce_meter = AverageMeter()
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    # set model and model_prior to training mode
    model.train()  
    model_prior.train()
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    for idx, data_ in pbar:
        # if idx>3: break #debug
        data_prior = data_[0]
        data = data_[1]
        data['mask'], data_prior['mask'] = None, None
        # some datasets need to ignore 'unlabeled' class
        if ('semantic3d' in cfg.dataset.common.NAME.lower()) or ('toronto3d' in cfg.dataset.common.NAME.lower()):
            data['mask'] = ~(data['y']==0) 
            data_prior['mask'] = ~(data_prior['y']==0)
        elif 'scannetv2' in cfg.dataset.common.NAME.lower():
            data['mask'] = ~(data['y']==255) 
            data_prior['mask'] = ~(data_prior['y']==255)
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            if data[key] is None:
                continue
            elif not isinstance(data[key], list):
                data_prior[key] = data_prior[key].cuda(non_blocking=True)
                data[key] = data[key].cuda(non_blocking=True)
            elif torch.is_tensor(data[key][0]):
                for i in range(len(data[key])):
                    data_prior[key][i] = data_prior[key][i].cuda(non_blocking=True)
                    data[key][i] = data[key][i].cuda(non_blocking=True)
        num_iter += 1
        target_prior, target = data_prior['y'].squeeze(-1), data['y'].squeeze(-1)
        if len(target_prior.size())>1: target_prior = torch.cat([target_split.squeeze() for target_split in target_prior.split(1,0)])
        if len(target.size())>1: target = torch.cat([target_split.squeeze() for target_split in target.split(1,0)])
        if cfg.model.get('dim_modifed', True):
            if len(data['x'].shape) > 2:
                data['x'] = data['x'].transpose(1, 2)
                data_prior['x'] = data_prior['x'].transpose(1, 2)
            data['x'] = get_scene_seg_features(cfg.model.in_channels, data['pos'], data['x'])
            data_prior['x'] = get_scene_seg_features(cfg.model_prior.in_channels, data_prior['pos'], data_prior['x'])

        prior_feas, prior_prototype = model_prior(data_prior, is_train=True, mask=data_prior['mask'], ignore_index=cfg.ignore_index)
        logits, main_feas, main_prototype = model(data, is_train=True, mask=data['mask'], ignore_index=cfg.ignore_index) 

        # NOTE: prior branch loss
        loss_prior_supcon = cfg.loss_prior_supcon * criterion_supcon(prior_feas[:, :-1], prior_feas[:, -1])
        prior_feas_target = main_prototype[prior_feas[:, -1].long(), :]
        loss_prior = cfg.loss_prior * criterion_l1(prior_feas[:,:-1], prior_feas_target)
        # NOTE: main branch loss
        if data['mask'] is None:
            target_main_feas = prior_prototype[target, :]
            loss_main_l1 = cfg.loss_main_l1 * criterion_l1(main_feas, target_main_feas)
            loss_main_ce = cfg.loss_main_ce * criterion(logits, target)
        else:
            target_mask = target[data['mask']]-1 if cfg.ignore_index==0 else target[data['mask']]
            target_main_feas = prior_prototype[target_mask, :]
            loss_main_l1 = cfg.loss_main_l1 * criterion_l1(main_feas, target_main_feas)
            loss_main_ce = cfg.loss_main_ce * criterion(logits, target, data['mask'])

        loss = loss_prior + loss_prior_supcon + loss_main_l1 + loss_main_ce
        loss.backward()

        # compute prior_prototype, loss_main, loss_prior and loss average on different gpu
        if cfg.distributed:
            dist.all_reduce(loss_prior)
            loss_prior /= cfg.world_size
            dist.all_reduce(loss_prior_supcon)
            loss_prior_supcon /= cfg.world_size
            dist.all_reduce(loss_main_l1)
            loss_main_l1 /= cfg.world_size
            dist.all_reduce(loss_main_ce)
            loss_main_ce /= cfg.world_size

        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip, norm_type=2)
                torch.nn.utils.clip_grad_norm_(model_prior.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0
            optimizer.step()
            optimizer.zero_grad()
            optimizer_prior.step()
            optimizer_prior.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)
                scheduler_prior.step(epoch)

        # update confusion matrix
        predits = logits.argmax(dim=1)
        if ('Semantic3D' in cfg.dataset.common.NAME) or ('Toronto3D' in cfg.dataset.common.NAME):
            predits = predits + 1 # restore mapping in training: from [0,7] to [1,8] 
            predits = predits[data['mask']]
            target = target[data['mask']]
        elif 'scannetv2' in cfg.dataset.common.NAME:
            predits = predits[data['mask']]
            target = target[data['mask']]
        cm.update(predits, target)
        loss_prior_meter.update(loss_prior.item())
        loss_prior_supcon_meter.update(loss_prior_supcon.item())
        loss_main_l1_meter.update(loss_main_l1.item())
        loss_main_ce_meter.update(loss_main_ce.item())
        loss_log = loss_main_l1_meter.val + loss_main_ce_meter.val + loss_prior_meter.val + loss_prior_supcon_meter.val

        if idx % cfg.print_freq:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                f"Loss {loss_log:.3f} Acc {cm.overall_accuray:.2f}")
    miou, macc, _, _, _ = cm.all_metrics()
    return loss_prior_meter.avg, loss_prior_supcon_meter.avg, loss_main_l1_meter.avg, loss_main_ce_meter.avg, miou, macc


@torch.no_grad()
def validate(model, val_loader, cfg):
    torch.cuda.empty_cache()
    # set model and model_prior to eval mode
    model.eval()
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    for idx, data in pbar:
        # if idx>2: break #debug
        # some datasets need to ignore 'unlabeled' class
        if ('semantic3d' in cfg.dataset.common.NAME.lower()) or ('toronto3d' in cfg.dataset.common.NAME.lower()):
            data['mask'] = ~(data['y']==0) 
        elif 'scannetv2' in cfg.dataset.common.NAME.lower():
            data['mask'] = ~(data['y']==255) 
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            if not isinstance(data[key], list):
                data[key] = data[key].cuda(non_blocking=True)
            elif torch.is_tensor(data[key][0]):
                for i in range(len(data[key])):
                    data[key][i] = data[key][i].cuda(non_blocking=True)
        target = data['y'].squeeze(-1)
        if cfg.model.get('dim_modifed', True):
            if len(data['x'].shape) > 2:
                data['x'] = data['x'].transpose(1, 2)
            data['x'] = get_scene_seg_features(cfg.model.in_channels, data['pos'], data['x'])

        logits = model(data) 

        # all points are concatnated into the first dim without batch dim in ptnet, so moidfy target dim
        if len(logits.size())==2 and len(target.size())>1:
            target = torch.cat([target_split.squeeze() for target_split in target.split(1,0)])
        # some datasets need to ignore 'unlabeled' class
        predits = logits.argmax(dim=1)
        if 'semantic3d' in cfg.dataset.common.NAME.lower() or ('toronto3d' in cfg.dataset.common.NAME.lower()):
            predits = predits + 1 # restore mapping in training: from [0,7] to [1,8] 
        elif 'scannetv2' in cfg.dataset.common.NAME.lower():
            predits = predits[data['mask']] 
            target = target[data['mask']] 
        cm.update(predits, target)
    tp, union, count = cm.tp, cm.union, cm.count
    if cfg.distributed:
        dist.all_reduce(tp), dist.all_reduce(union), dist.all_reduce(count)
    miou, macc, oa, ious, _ = get_mious(tp, union, count)
    return miou, macc, oa, ious


@torch.no_grad()
def test_s3dis(model, area, cfg):
    """using a part of original point cloud as input to save memory.
    Args:
        model (_type_): _description_
        test_loader (_type_): _description_
        cfg (_type_): _description_
        model_prior (_type_): _description_
        prior_prototype (_type_): _description_
    Returns:
        _type_: _description_
    """
    torch.cuda.empty_cache()
    model.eval() # set model and model_prior to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    set_random_seed(0)
    voxel_size =  cfg.dataset.common.voxel_size
    cfg.visualize = cfg.get('visualize', False)
    
    # data
    trans_split = 'val' if cfg.datatransforms.get('test', None) is None else 'test'
    transform =  build_transforms_from_cfg(trans_split, cfg.datatransforms)
    raw_root = os.path.join(cfg.dataset.common.data_root, 'raw')
    data_list = sorted(os.listdir(raw_root))
    data_list = [item[:-4] for item in data_list if 'Area_' in item]
    data_list = [item for item in data_list if 'Area_{}'.format(area) in item]

    # count_total = 0 # record the test iter nums
    for cloud_idx, item in enumerate(tqdm(data_list)):
        data_path = os.path.join(raw_root, item + '.npy')
        cdata = np.load(data_path).astype(np.float32)  # xyz, rgb, label, N*7
        coord_min = np.min(cdata[:, :3], 0)
        cdata[:, :3] -= coord_min
        label = torch.from_numpy(cdata[:, 6].astype(np.int64).squeeze()).cuda(non_blocking=True)
        colors = np.clip(cdata[:, 3:6] / 255., 0, 1).astype(np.float32)

        all_logits, all_point_inds = [], []
        uniq_idx, count = voxelize(cdata[:, :3], voxel_size, mode=1)
        for i in range(count.max()):
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
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

            logits = model(data) 
            all_logits.append(logits)
        
        all_logits = torch.cat(all_logits, dim=0)
        all_point_inds = torch.from_numpy(np.hstack(all_point_inds)).cuda(non_blocking=True)
        # project voxel subsampled to original set
        all_logits = scatter(all_logits, all_point_inds, dim=0, reduce='mean')
        all_point_inds = scatter(all_point_inds, all_point_inds, dim=0, reduce='mean')
        cm.update(all_logits.argmax(dim=1), label)
        
    tp, union, count = cm.tp, cm.union, cm.count
    miou, macc, oa, ious, _ = get_mious(tp, union, count)
    return miou, macc, oa, ious


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Scene segmentation training/testing')
    parser.add_argument('--debug', type=bool, default=False, help='setting debug mode to control not create tensorboard')
    # parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--cfg', type=str, help='config file')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args()
    # args.debug = True # set debug mode
    args.cfg = "cfgs/spg/spg_pointnet++.yaml"
    # args.cfg = "cfgs/spg/spg_ptv1.yaml"
    # args.cfg = "cfgs/spg/spg_ptv2.yaml"
    args.profile = True

    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(vars(args))    # overwrite the default arguments in yml  
    
    # NOTE: 'test'/'val' or 'resume' mode
    # cfg.mode = 'test'
    # cfg.mode = 'val'
    # cfg.mode = 'resume'
    # cfg.pretrained_path = ""
    # cfg.prior_model_path = ""
    cfg.seed = np.random.randint(1, 10000)
    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]  # task/dataset name, \eg s3dis, modelnet40_cls
    cfg.cfg_basename = args.cfg.split('.')[-2].split('/')[-1]   # cfg_basename, \eg pointnext-xl 
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.cfg_basename,  # cfg file name
        f'ngpus{cfg.world_size}',
        f'seed{cfg.seed}',
    ]
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)

    cfg.is_training = cfg.mode in ['train', 'training', 'finetune', 'finetuning']
    if cfg.mode == 'train':
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
    else:  # resume from the existing ckpt and reuse the folder.
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    if not cfg.debug:
        with open(cfg_path, 'w') as f:
            yaml.dump(cfg, f, indent=2)
            os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path

    # multi processing.
    if cfg.mp:
        port = find_free_port()
        cfg.dist_url = f"tcp://localhost:{port}"
        print('using mp spawn for distributed training')
        mp.spawn(main, nprocs=cfg.world_size, args=(cfg,))
    else:
        main(0, cfg)
