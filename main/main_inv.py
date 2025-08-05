"""
InvSpaceNet: one main branch for all classes, and one branch for inverse sampling. The feature information interaction method uses SmoothL1 loss
"""
import __init__
import argparse, yaml, os, time, logging, numpy as np
# import nni
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist, multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, save_prior, load_prior, \
    setup_logger_dist, cal_model_parm_nums, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from openpoints.dataset import build_dataloader_from_cfg, get_scene_seg_features
from openpoints.dataset.data_util import voxelize
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
from openpoints.models.layers import furthest_point_sample, ball_query, grouping_operation
from main.test_utils import write_to_csv
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
    if cfg.rank == 0 and (not cfg.debug): 
        writer = SummaryWriter(log_dir=os.path.join(cfg.run_dir, 'tensorboard')) if cfg.is_training else None # only training create tensorboard
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    # logging.info(cfg)
    
    # get parameters form tuner
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
    # cfg.model.beta = 1.0 - 1.0 / train_loader.__len__()

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
    cfg.num_per_class = val_loader.dataset.num_per_class if hasattr(val_loader.dataset, 'num_per_class') else None

    if cfg.dataset.common.name.lower() == "toronto3d":
        test_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='test',
                                           distributed=cfg.distributed
                                           )
        logging.info(f"length of test dataset: {len(test_loader.dataset)}")
    
    # optimizer & scheduler
    cfg.iters = len(train_loader) * cfg.epochs
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)
    optimizer_prior = build_optimizer_from_cfg(model_prior, lr=cfg.lr, **cfg.optimizer)
    scheduler_prior = build_scheduler_from_cfg(cfg, optimizer_prior)
    
    # optionally resume from a checkpoint
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler, pretrained_path=cfg.pretrained_path)
            resume_checkpoint(cfg, model_prior, optimizer_prior, scheduler_prior, pretrained_path=cfg.model_prior)
            val_miou = validate(model, val_loader, cfg)
            logging.info(f'\nresume val miou is {val_miou}\n ')
            # resume the SummaryWriter instance from the crashed epoch
            if cfg.rank == 0:
                writer = SummaryWriter(log_dir=os.path.join(cfg.run_dir, 'tensorboard'), purge_step=cfg.start_epoch)
        elif cfg.mode == 'test':
            if cfg.rank == 0:
                if "s3dis" in cfg.dataset.common.name.lower():
                    best_epoch, best_val = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
                    miou, macc, oa, ious = test_s3dis(model, cfg.dataset.common.test_area, cfg)
                    with np.printoptions(precision=2, suppress=True):
                        logging.info(f'Best ckpt @E{best_epoch},  test_oa , test_macc, test_miou: {oa:.2f} {macc:.2f} {miou:.2f}, '
                                    f'\nEach cls Test IoU: {ious}')
                    cfg.csv_path = os.path.join(cfg.run_dir, cfg.run_name + '_test.csv')
                    write_to_csv(oa, macc, miou, ious, best_epoch, cfg)
                    return miou 
                elif "toronto3d" in cfg.dataset.common.name.lower():
                    best_epoch, best_val = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
                    test_toronto3d(model, test_loader, cfg)
                    return True
            else:
                return True
        else:
            load_checkpoint(model, pretrained_path=cfg.pretrained_path)
            logging.info(f'Finetuning from {cfg.pretrained_path}')
    else:
        logging.info('Training from scratch')
    
    if cfg.criterion.get('weight', None) is not None:
        cfg.criterion.weight = torch.tensor(cfg.criterion.weight)
    criterion = build_criterion_from_cfg(cfg.criterion).cuda()
    if cfg.get('criterion_mask', None) is None:
        criterion_mask = None
    else:
        criterion_mask = build_criterion_from_cfg(cfg.criterion_mask).cuda()
    criterion_l1 = nn.SmoothL1Loss(reduction='mean')
    criterion_supcon = build_criterion_from_cfg(cfg.criterion_SupCon).cuda()
    
    # ===> start training
    val_miou, val_macc, val_oa, val_ious = 0., 0., 0., []
    best_val, macc_when_best, oa_when_best, ious_when_best, best_epoch = 0., 0., 0., [], 0
    scaler, scaler_prior = None, None
    if cfg.get('enable_amp', False):
        scaler = torch.cuda.amp.GradScaler()
        scaler_prior = torch.cuda.amp.GradScaler()
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        # if epoch >2: break # debug
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'epoch'):  # some dataset sets the dataset length as a fixed steps.
            train_loader.dataset.epoch = epoch - 1
        l1_loss, struct_loss, ce_loss, supcon_prior_loss, ce_prior_loss, train_miou, train_macc = \
            train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, cfg,\
                model_prior, optimizer_prior, scheduler_prior, scaler_prior, criterion_l1, criterion_supcon, criterion_mask)

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
                    logging.info(
                        f'Find a better ckpt @E{epoch}, val_miou {val_miou:.2f} val_macc {macc_when_best:.2f}, val_oa {oa_when_best:.2f}')
        with np.printoptions(precision=2, suppress=True):
            logging.info(f'\nmious: {val_ious}')
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
            writer.add_scalar('train/l1_loss', l1_loss, epoch)
            writer.add_scalar('train/struct_loss', struct_loss, epoch)
            writer.add_scalar('train/ce_loss', ce_loss, epoch)
            writer.add_scalar('train/supcon_prior_loss', supcon_prior_loss, epoch)
            writer.add_scalar('train/ce_prior_loss', ce_prior_loss, epoch)
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
                            is_best=is_best, post_fix='model_prior_ckpt', is_logging=False)
            is_best = False
            # nni.report_intermediate_result(val_miou)

    # validate
    with np.printoptions(precision=2, suppress=True):
        logging.info(
            f'Best ckpt @E{best_epoch},  val_oa {oa_when_best:.2f}, val_macc {macc_when_best:.2f}, val_miou {best_val:.2f}, '
            f'\nEach cls IoU: {ious_when_best}')
    
    # test
    if cfg.rank == 0: # NOTE: only test on the main process, because in the ddp mode, the test code will be used for repeated times
        if "s3dis" in cfg.dataset.common.name.lower():
            load_checkpoint(model, pretrained_path=os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_main_model_ckpt_best.pth'))
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
        elif "toronto3d" in cfg.dataset.common.name.lower():
            cfg.pretrained_path = os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_main_model_ckpt_best.pth')
            load_checkpoint(model, pretrained_path=cfg.pretrained_path)
            test_toronto3d(model, test_loader, cfg)
    return True


def mask_target(target, offset, pos, ignore_index):
    new_mask = target!=ignore_index
    if ignore_index == 0:
        new_target = target[new_mask] - 1 # [1,8]->[0,7]
    new_pos = pos[new_mask]
    new_offset = []
    for i in range(offset.size(0)):
        new_offset.append(new_mask[:(offset[i])].sum())
    new_offset = torch.tensor(new_offset, dtype=offset.dtype, device=offset.device)
    return new_mask, new_target, new_offset, new_pos

def bernoulli_mask(target, point_num, offset, ignore_index):
    if len(point_num) == 13: 
        truncation = -6 # s3dis class "bookcase"
    elif len(point_num) == 8: 
        truncation = -4 # toronto3d class "car"
    elif len(point_num) == 19: 
        truncation = -7 # semantickitti class "car"
    prob = 1 / (point_num / np.sort(point_num)[truncation])
    prob = torch.tensor(prob, device=target.device)
    prob = torch.clip(prob, 0, 1)
    if ignore_index==0:  
        prob = torch.cat((torch.zeros(1, device=prob.device), prob))
    berno_mask = torch.bernoulli(prob[target]).bool()
    if ignore_index==0: 
        berno_target = target[berno_mask] - 1 
        prob = prob[1:]
    else: 
        berno_target = target[berno_mask]
    berno_offset = []
    for i in range(offset.size(0)):
        berno_offset.append(berno_mask[:(offset[i])].sum())
    berno_offset = torch.tensor(berno_offset, dtype=offset.dtype, device=offset.device)
    return berno_mask, berno_target, berno_offset, prob

def create_affinity_matrix(xyz: torch.Tensor=None, feats: torch.Tensor=None, offset: torch.Tensor=None, target: torch.Tensor=None, sample: bool=False, num_classes: int=13):
    """
    xyz: (b*n, 3)->[b, npoints, 3](List len==b)
    features: (b*n, c)->[b, c, npoints](List len==b)
    return: affinity_matrix and according target(optional)
    """
    affinity_matrix = []
    if sample:
        stride = 16
        if num_classes == 13: # s3dis
            radius = 0.16
            nsample = 32 # there are about 40 samples in a sphere of radius 0.16
        elif num_classes == 8: # toronto3d
            radius = 0.48
            nsample = 32 # there are about 40 samples in a sphere of radius 0.48
        elif num_classes == 19: # semantickitti
            stride = 32
            radius = 0.80
            nsample = 32 # there are about 40 samples in a sphere of radius 1.60
        offset_list = []
        select_target = []
        for i in range(offset.size(0)):
            if i>0: offset_list.append(offset[i].item() - offset[i-1].item()) 
            else: offset_list.append(offset[0].item())
        xyz_list = list(torch.split(xyz, offset_list, dim=0))
        feats = feats.transpose(0,1).contiguous()
        feats_list = torch.split(feats, offset_list, dim=1)
        target_list = torch.split(target, offset_list, dim=0)
        for i in range(offset.size(0)):
            aff_feats = []
            aff_target = []
            for c in range(num_classes):
                mask_c = target_list[i]==c
                if mask_c.sum()>0:
                    xyz_c = xyz_list[i][mask_c, :].unsqueeze(0) # (1, npoints, 3)
                    feat_c = feats_list[i][:, mask_c].unsqueeze(0) # (1, c, npoints)
                    target_c = target_list[i][mask_c]
                    spe_size = xyz_c.size(1)//stride if xyz_c.size(1)>stride else 1
                    idx = furthest_point_sample(xyz_c, spe_size).long()
                    query_xyz = torch.gather(xyz_c, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
                    aff_target_c = torch.gather(target_c, 0, idx.squeeze(0))
                    neighbour_idx = ball_query(radius, nsample, xyz_c, query_xyz)
                    # observe the number of points in each neighbour
                    # print(f"class{c}", torch.tensor([neighbour_idx.view(-1,neighbour_idx.size(2))[i,:].unique().size() \
                    #     for i in range(neighbour_idx.size(1))], dtype=torch.float).mean())
                    grouped_features = grouping_operation(feat_c, neighbour_idx) # (B, C, npoint after downsample, nsample)
                    aff_feats_c = torch.mean(grouped_features, dim=-1, keepdim=False) #（b(1), c, npoints）
                    aff_feats.append(aff_feats_c.squeeze(0).transpose(0,1).contiguous())
                    aff_target.append(aff_target_c)
            if len(aff_feats)>0:
                aff_feats = torch.cat(aff_feats)
                # aff_feats = nn.functional.normalize(aff_feats, dim=1)
                aff_target = torch.cat(aff_target)
                affinity_matrix.append(torch.einsum("...nc,...mc->...nm", [aff_feats, aff_feats]))
                select_target.append(aff_target)
        return affinity_matrix, select_target, nsample
    else:
        """
        feats: feature prototype, (num_classes, d)
        target: List, used to compute affinity_matrix
        """
        for target_b in target:
            feats_use = feats[target_b, :]
            affinity_matrix.append(torch.einsum("...nc,...mc->...nm", [feats_use, feats_use]))
        return affinity_matrix

def supcon_input(xyz=None, feats=None, offset=None, target=None, num_classes=13, prob=None):
    """
    xyz: (b*n, 3)->[b, npoints, 3](List len==b)
    features: (b*n, c)->[b, npoints, c](List len==b)
    return: supcon loss input feature and target
    """
    stride = 8  if num_classes!=19 else 24
    offset_list = []
    for i in range(offset.size(0)):
        if i>0: offset_list.append(offset[i].item() - offset[i-1].item()) 
        else: offset_list.append(offset[0].item())
    xyz_list = list(torch.split(xyz, offset_list, dim=0))
    feats = feats.transpose(0,1).contiguous()
    feats_list = torch.split(feats, offset_list, dim=1)
    target_list = torch.split(target, offset_list, dim=0)
    sup_feats, sup_target = [], []
    for i in range(offset.size(0)):
        for c in range(num_classes):
            mask_c = target_list[i]==c
            if mask_c.sum()>0:
                if num_classes == 13:
                    radius = 0.16 / (prob[c].item()**0.5)
                    nsample = 32 # there are about 38 samples in a sphere of radius 0.16
                elif num_classes == 8: # toronto3d
                    radius = 0.48 / (prob[c].item()**0.5)
                    nsample = 32
                elif num_classes == 19: # semantickitti
                    radius = 0.80 / (prob[c].item()**0.5)
                    nsample = 32
                xyz_c = xyz_list[i][mask_c, :].unsqueeze(0) # (1, npoints, 3)
                feat_c = feats_list[i][:, mask_c].unsqueeze(0) # (1, c, npoints)
                target_c = target_list[i][mask_c]
                spe_size = xyz_c.size(1)//stride if xyz_c.size(1)>stride else 1
                idx = furthest_point_sample(xyz_c, spe_size).long()
                query_xyz = torch.gather(xyz_c, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
                sup_target_c = torch.gather(target_c, 0, idx.squeeze(0))
                neighbour_idx = ball_query(radius, nsample, xyz_c, query_xyz)
                # observe the number of points in each neighbour
                # print(f"class{c}, ", torch.tensor([neighbour_idx.view(-1,neighbour_idx.size(2))[i,:].unique().size() \
                #     for i in range(neighbour_idx.size(1))], dtype=torch.float).mean())
                grouped_features = grouping_operation(feat_c, neighbour_idx) # (B, C, npoint after downsample, nsample)
                sup_feats_c = torch.mean(grouped_features, dim=-1, keepdim=False) #（b(0), c, npoints）
                sup_feats.append(sup_feats_c.squeeze(0).transpose(0,1).contiguous())
                sup_target.append(sup_target_c)
    sup_feats = torch.cat(sup_feats, dim=0)
    sup_target = torch.cat(sup_target, dim=0)
    return sup_feats, sup_target


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, cfg, \
                    model_prior, optimizer_prior, scheduler_prior, scaler_prior, criterion_l1, criterion_supcon, criterion_mask):
    loss_struct_meter = AverageMeter()
    loss_l1_meter = AverageMeter()
    loss_ce_meter = AverageMeter()
    loss_kd_meter = AverageMeter()
    # loss_l1_prior_meter = AverageMeter()
    loss_ce_prior_meter = AverageMeter()
    loss_prior_supcon_meter = AverageMeter()
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    # set model and model_prior to training mode
    model.train()  
    model_prior.train()
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    for idx, data_ in pbar:
        # if idx>2: break #debug
        num_iter += 1
        total_iter = (epoch-1) * len(train_loader) + num_iter
        # data from two sources
        data_prior = data_[0]
        data = data_[1]
        if len(data_prior['y'].size())>1: data_prior['y'] = torch.flatten(data_prior['y'])
        if len(data['y'].size())>1: data['y'] = torch.flatten(data['y'])
        # Toronto3D need to ignore 'unlabeled' class
        data['mask'], data_prior['mask'] = None, None
        if ('Toronto3D' in cfg.dataset.common.NAME) or ('SemanticKITTI' in cfg.dataset.common.NAME):
            data['mask'] = ~(data['y']==0)
            data_prior['mask'] = ~(data_prior['y']==0)
        for key in data_prior.keys():
            if data_prior[key] is not None:
                data_prior[key] = data_prior[key].cuda(non_blocking=True)
        for key in data.keys():
            if data[key] is not None:
                data[key] = data[key].cuda(non_blocking=True)
        if cfg.model.get('dim_modifed', True):
            if len(data['x'].shape) > 2:
                data['x'] = data['x'].transpose(1, 2)
                data_prior['x'] = data_prior['x'].transpose(1, 2)
            data['x'] = get_scene_seg_features(cfg.model.in_channels, data['pos'], data['x'])
            data_prior['x'] = get_scene_seg_features(cfg.model.in_channels, data_prior['pos'], data_prior['x'])
        target_prior = data_prior['y']
        target = data['y']

        if 'offset' not in data.keys(): 
            data['offset'] = torch.tensor([i*data['pos'].size(1) for i in range(1, data['pos'].size(0)+1)])
            data_prior['offset'] = data['offset']
        minor_mask, minor_target, minor_offset, minor_pos = None, target, data['offset'], data['pos']
        if len(minor_pos.size())==3: minor_pos = minor_pos.view(-1, minor_pos.size(-1)).contiguous()
        if cfg.ignore_index == 0: # TODO for SemanticKITTI
            minor_mask, minor_target, minor_offset, minor_pos = mask_target(target, data['offset'], minor_pos, cfg.ignore_index)
        berno_mask_r, berno_target_r, berno_offset, prob = bernoulli_mask(target_prior, cfg.num_per_class, data_prior['offset'], cfg.ignore_index)
        
        cfg.enable_amp = cfg.get('enable_amp', False)
        with torch.cuda.amp.autocast(enabled=cfg.enable_amp):
            logits, feas_norm, _ = model(data, is_train=True, minor_mask=minor_mask)
            affinity_matrix, select_target, nsample = create_affinity_matrix(minor_pos, feas_norm, minor_offset, minor_target, sample=True, num_classes=cfg.num_classes)
            logits_prior, feas_norm_prior, feas_memory_prior = model_prior(data_prior, is_train=True, minor_mask=berno_mask_r)
            
            # main branch
            # alpha = math.exp(-epoch/(0.4*cfg.epochs))
            offset_list = []
            for i in range(minor_offset.size(0)):
                if i>0: offset_list.append(minor_offset[i].item() - minor_offset[i-1].item()) 
                else: offset_list.append(minor_offset[0].item())
            offset_mean = torch.mean(torch.tensor(offset_list).float())
            ##########l1############
            feas_norm_target = feas_memory_prior[minor_target, :]
            loss_l1 = cfg.loss_l1 * criterion_l1(feas_norm, feas_norm_target)
            l1_norm = torch.mean(torch.norm(feas_norm - feas_norm_target, dim=1)).item() / offset_mean
            ##########struct############
            loss_struct, struct_norm = 0, 0
            if len(select_target)>0:
                affinity_matrix_target = create_affinity_matrix(feats=feas_memory_prior, target=select_target, sample=False)
                for i in range(len(affinity_matrix)):
                    loss_struct += cfg.loss_struct * criterion_l1(affinity_matrix[i], affinity_matrix_target[i])
                    struct_norm += torch.sum(torch.abs(affinity_matrix[i] - affinity_matrix_target[i])).item() / \
                                    (nsample * affinity_matrix[i].size(0) ** 3 + torch.finfo(affinity_matrix[i].dtype).eps)
            struct_norm = struct_norm / len(affinity_matrix)
            l1_weight = 2 * struct_norm / (l1_norm + struct_norm)
            struct_weight = 2 * l1_norm / (l1_norm + struct_norm)
            loss_l1 = l1_weight * loss_l1
            loss_struct = struct_weight * loss_struct
            # print(f"l1_weight:{l1_weight}; struct_weight:{struct_weight}")
            ##########ce############
            if data['mask'] is None:
                loss_ce = cfg.loss_ce * criterion(logits, target)
            else:
                loss_ce = cfg.loss_ce * criterion_mask(logits, target, data['mask']) 
            # loss = (1-alpha) * (loss_l1 + loss_struct) + alpha * loss_ce
            loss = loss_l1 + loss_struct + loss_ce

            # # NOTE: watch the grad of ptv2 decoder last fc and seg first fc
            # loss_l1.backward(retain_graph=True)
            # grad_l1 = torch.norm(model.dec_stages[2].blocks.blocks[0].fc3.weight.grad)
            # optimizer.zero_grad()
            # loss_struct.backward(retain_graph=True)
            # grad_struct = torch.norm(model.dec_stages[2].blocks.blocks[0].fc3.weight.grad)
            # optimizer.zero_grad()
            # loss_ce.backward(retain_graph=True)
            # grad_ce = torch.norm(model.dec_stages[2].blocks.blocks[0].fc3.weight.grad)
            # # grad_ce_ = torch.norm(model.seg_head[0].weight.grad) # An order of magnitude larger than grad_ce
            # optimizer.zero_grad()

            # prior branch
            ##########supcon############
            if len(data_prior['pos'].size())==3: data_prior['pos'] = data_prior['pos'].view(-1, minor_pos.size(-1)).contiguous()
            sup_feas, sup_target = supcon_input(xyz=data_prior['pos'][berno_mask_r, :], feats=feas_norm_prior, offset=berno_offset,\
                target=berno_target_r, num_classes=cfg.num_classes, prob=prob)
            loss_prior_supcon = cfg.loss_prior_supcon * criterion_supcon(sup_feas, sup_target)
            ##########ce############
            loss_ce_prior = cfg.loss_ce_prior * criterion_mask(logits_prior, target_prior, berno_mask_r)
            loss_prior = loss_prior_supcon + loss_ce_prior

            # # NOTE: watch the grad of ptv2 decoder last fc and seg first fc
            # loss_prior_supcon.backward(retain_graph=True)
            # grad_supcon_prior = torch.norm(model_prior.dec_stages[2].blocks.blocks[0].fc3.weight.grad)
            # optimizer_prior.zero_grad()
            # loss_ce_prior.backward(retain_graph=True)
            # grad_ce_prior = torch.norm(model_prior.dec_stages[2].blocks.blocks[0].fc3.weight.grad)
            # # grad_ce_prior_ = torch.norm(model_prior.seg_head[0].weight.grad) # An order of magnitude larger than grad_ce
            # optimizer_prior.zero_grad()
            # print(f"grad_l1 {grad_l1:.2e}, grad_struct {grad_struct:.2e}, grad_ce {grad_ce:.2e}; grad_supcon_prior {grad_supcon_prior:.2e}, grad_ce_prior {grad_ce_prior:.2e}")
        
        # optimize
        optimizer.zero_grad()
        optimizer_prior.zero_grad()
        if cfg.enable_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler_value = scaler.get_scale()
            scaler.update()
            if not cfg.sched_on_epoch:
                if scaler_value <= scaler.get_scale():
                    scheduler.step(total_iter)
            scaler_prior.scale(loss_prior).backward()
            scaler_prior.step(optimizer_prior)
            scaler_prior_value = scaler_prior.get_scale()
            scaler_prior.update()
            if not cfg.sched_on_epoch:
                if scaler_prior_value <= scaler_prior.get_scale():
                    scheduler_prior.step(total_iter)
        else:
            loss.backward()
            loss_prior.backward()
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip, norm_type=2)
                torch.nn.utils.clip_grad_norm_(model_prior.parameters(), cfg.grad_norm_clip, norm_type=2)
            optimizer.step()
            optimizer_prior.step()
            if not cfg.sched_on_epoch:
                scheduler.step(total_iter)
                scheduler_prior.step(total_iter)

        # compute loss_main, loss_prior average on different gpu
        if cfg.distributed:
            dist.all_reduce(loss_l1)
            loss_l1 /= cfg.world_size
            dist.all_reduce(loss_struct)
            loss_struct /= cfg.world_size
            dist.all_reduce(loss_ce)
            loss_ce /= cfg.world_size
            dist.all_reduce(loss_ce_prior)
            loss_ce_prior /= cfg.world_size
            dist.all_reduce(loss_prior_supcon)
            loss_prior_supcon /= cfg.world_size

        # update confusion matrix
        predits = logits.argmax(dim=1)
        if ('Toronto3D' in cfg.dataset.common.NAME) or ('SemanticKITTI' in cfg.dataset.common.NAME):
            predits = predits + 1 # restore mapping in training: from [0,7] to [1,8] 
            predits = predits[data['mask']]
            target = target[data['mask']]
        cm.update(predits, target)
        loss_l1_meter.update(loss_l1.item())
        loss_struct_meter.update(loss_struct.item())
        loss_ce_meter.update(loss_ce.item())
        loss_prior_supcon_meter.update(loss_prior_supcon.item())
        loss_ce_prior_meter.update(loss_ce_prior.item())
        loss_log = loss_l1_meter.val + loss_struct_meter.val + loss_ce_meter.val + loss_kd_meter.val

        if idx % cfg.print_freq:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                f"Loss {loss_log:.3f} Acc {cm.overall_accuray:.2f}")
    miou, macc, _, _, _ = cm.all_metrics()
    return loss_l1_meter.avg, loss_struct_meter.avg, loss_ce_meter.avg, loss_prior_supcon_meter.avg, loss_ce_prior_meter.avg, miou, macc


@torch.no_grad()
def validate(model, val_loader, cfg):
    torch.cuda.empty_cache()
    # set model and model_prior to eval mode
    model.eval()
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    for idx, data in pbar:
        # if idx>2: break #debug
        if len(data['y'].size())>1: data['y'] = torch.flatten(data['y'])
        if ('Toronto3D' in cfg.dataset.common.NAME) or ('SemanticKITTI' in cfg.dataset.common.NAME):
            data['mask'] = ~(data['y']==0) 
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

        # Toronto3D need to ignore 'unlabeled' class
        predits = logits.argmax(dim=1)
        if ('Toronto3D' in cfg.dataset.common.NAME) or ('SemanticKITTI' in cfg.dataset.common.NAME):
            predits = predits + 1 # restore mapping in training: from [0,7] to [1,8] 
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
        memory_prior (_type_): _description_
    Returns:
        _type_: _description_
    """
    torch.cuda.empty_cache()
    # set model to eval mode
    model.eval()
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    set_random_seed(0)
    voxel_size =  cfg.dataset.common.voxel_size
    cfg.visualize = cfg.get('visualize', False)
    if cfg.visualize:
        from openpoints.dataset.vis3d import write_obj
        cfg.vis_dir = os.path.join(cfg.run_dir, 'visualization')
        os.makedirs(cfg.vis_dir, exist_ok=True)
        cfg.cmap = cfg.cmap.astype(np.float32) / 255.
    
    # data
    trans_split = 'val' if cfg.datatransforms.get('test', None) is None else 'test'
    transform =  build_transforms_from_cfg(trans_split, cfg.datatransforms)
    raw_root = os.path.join(cfg.dataset.common.data_root, 'raw')
    data_list = sorted(os.listdir(raw_root))
    data_list = [item[:-4] for item in data_list if 'Area_' in item]
    data_list = [item for item in data_list if 'Area_{}'.format(area) in item]

    for cloud_idx, item in enumerate(tqdm(data_list)):
        data_path = os.path.join(raw_root, item + '.npy')
        cdata = np.load(data_path).astype(np.float32)  # xyz, rgb, label, N*7
        coord_min = np.min(cdata[:, :3], 0)
        cdata[:, :3] -= coord_min
        label = torch.from_numpy(cdata[:, 6].astype(np.int).squeeze()).cuda(non_blocking=True)
        colors = np.clip(cdata[:, 3:6] / 255., 0, 1).astype(np.float32)

        all_logits, all_point_inds = [], []
        uniq_idx, count = voxelize(cdata[:, :3], voxel_size, mode=1)
        # count_total += count.max()
        # continue
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
                data['offset'] = torch.IntTensor([len(coord)])

            keys = data.keys() if callable(data.keys) else data.keys
            for key in keys:
                data[key] = data[key].cuda(non_blocking=True)
            if 'student_model' in cfg.keys(): # when kd, need to use student model cfg
                in_channels = cfg.student_model.in_channels if 'in_channels' in cfg.student_model.keys() \
                    else cfg.student_model.encoder_args.in_channels
            else:
                in_channels = cfg.model.in_channels if 'in_channels' in cfg.model.keys() else cfg.model.encoder_args.in_channels
            data['x'] = get_scene_seg_features(in_channels, data['pos'], data['x'])

            logits = model(data) 

            # modify from logits(a list include each intermediate output) to logits[-1]
            if isinstance(logits, list):
                logits = logits[-1]
            all_logits.append(logits)
            """visualization in debug mode 
            from openpoints.dataset.vis3d import vis_points, vis_multi_points
            # vis_points(cdata[:, :3], cdata[:, 3:6]/255.)
            # vis_multi_points([cdata[:, :3], coord], [cdata[:, 3:6].astype(np.uint8), feat.astype(np.uint8)])
            vis_multi_points([cdata[:, :3], coord], labels=[label.cpu().numpy(), logits.argmax(dim=1).squeeze().cpu().numpy()])
            """
        all_logits = torch.cat(all_logits, dim=0)
        if not cfg.dataset.common.get('variable', False) and len(all_logits.size())>2:
            all_logits = all_logits.transpose(1, 2).reshape(-1, cfg.num_classes)
        all_point_inds = torch.from_numpy(np.hstack(all_point_inds)).cuda(non_blocking=True)

        # project voxel subsampled to original set
        all_logits = scatter(all_logits, all_point_inds, dim=0, reduce='mean')
        all_point_inds = scatter(all_point_inds, all_point_inds, dim=0, reduce='mean')
        cm.update(all_logits.argmax(dim=1), label)
        
        if cfg.visualize:
            gt = label.cpu().numpy().squeeze()
            pred = all_logits.argmax(dim=1).cpu().numpy().squeeze()
            gt = cfg.cmap[gt, :]
            pred = cfg.cmap[pred, :]
            # output pred labels
            write_obj(cdata[:, :3], colors,
                        os.path.join(cfg.vis_dir, f'input-Area{area}-{cloud_idx}.obj'))
            # output ground truth labels
            write_obj(cdata[:, :3], gt,
                        os.path.join(cfg.vis_dir, f'gt-Area{area}-{cloud_idx}.obj'))
            # output pred labels
            write_obj(cdata[:, :3], pred,
                        os.path.join(cfg.vis_dir, f'pred-Area{area}-{cloud_idx}.obj'))
    tp, union, count = cm.tp, cm.union, cm.count
    miou, macc, oa, ious, _ = get_mious(tp, union, count)
    return miou, macc, oa, ious


@torch.no_grad()
def test_toronto3d(model, test_loader, cfg, num_votes=100):
    """
    Toronto3D L002
    """
    model.eval()
    #####################
    # Network predictions parameter init
    #####################
    test_probs = [np.zeros((cfg.num_classes, l.data.shape[0]), dtype=np.float16)
                    for l in test_loader.dataset.input_trees]
    test_labels = [test_loader.dataset.input_labels[i] for i in range(len(test_loader.dataset.input_labels))]
    test_labels = [np.zeros(l.shape[0], dtype=np.int32) for l in test_labels]

    test_smooth = 0.75  # Smoothing parameter for votes
    epoch_id = 0
    last_min = -0.5
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    while last_min < num_votes:
        pbar = tqdm(enumerate(test_loader), total=test_loader.__len__())
        t1 = time.time()
        for _, data in pbar:
            cloud_idx = data['cloud_idx']
            point_idx = data['point_idx']
            if len(data['y'].size())>1: data['y'] = torch.flatten(data['y'])
            keys = data.keys() if callable(data.keys) else data.keys
            for key in keys:
                if not isinstance(data[key], list) and torch.is_tensor(data[key]):
                    data[key] = data[key].cuda(non_blocking=True)
                elif isinstance(data[key], list) and torch.is_tensor(data[key][0]):
                    for i in range(len(data[key])):
                        data[key][i] = data[key][i].cuda(non_blocking=True)

            if len(data['x'].shape) > 2:
                data['x'] = data['x'].transpose(1, 2)
            data['x'] = get_scene_seg_features(cfg.model.in_channels, data['pos'], data['x'])
            stacked_probs = nn.functional.softmax(model(data), dim=1)
            stacked_labels = data['y']
            stacked_probs = stacked_probs.cpu().numpy()
            stacked_labels = stacked_labels.cpu().numpy()

            for j in range(np.shape(point_idx)[0]):
                probs = stacked_probs[j*point_idx.shape[1]:(j+1)*point_idx.shape[1]]
                inds = point_idx[j, :]
                c_i = cloud_idx[j]
                test_probs[c_i][:, inds] = test_smooth * test_probs[c_i][:, inds] + (1 - test_smooth) * probs.T
                test_labels[c_i][inds] = stacked_labels[j*point_idx.shape[1]:(j+1)*point_idx.shape[1]]
            pbar.set_description(f"Test Epoch [{epoch_id}] "
                                f"Min Possibility {np.min(test_loader.dataset.min_possibility):.2f}")
        
        new_min = np.min(test_loader.dataset.min_possibility)
        logging.info(f'Test Epoch [{epoch_id}], end. Min possibility = {new_min:.2f}')
        epoch_id += 1
        
        if last_min + 4 < new_min:
            logging.info(f"Saving point clouds predictions. Reproject Vote #{int(np.floor(new_min)):d}")
            files = test_loader.dataset.test_files
            for test_i, file_path in enumerate(files):
                # Get file
                data = read_ply(file_path)
                points = np.vstack((data['x'], data['y'], data['z'])).T.astype(np.float16)
                # Reproject probs
                probs = np.zeros(shape=[np.shape(points)[0], 8], dtype=np.float16)
                proj_index = test_loader.dataset.test_proj[test_i]
                probs = test_probs[test_i][:, proj_index]
                labels = test_labels[0]
                predits = test_probs[test_i].argmax(axis=0)
                mask = ~(labels==0)
                predits = predits + 1 # restore mapping in training: from [0,7] to [1,8] 
                predits = predits[mask]
                labels = labels[mask]
                predits = torch.from_numpy(predits).cuda(non_blocking=True)
                labels = torch.from_numpy(labels).cuda(non_blocking=True)
                cm.update(predits, labels)
        else:
            continue
        t2 = time.time()
        tp, union, count = cm.tp, cm.union, cm.count
        miou, macc, oa, ious, accs = get_mious(tp, union, count)
        logging.info(f'Toronto3D_Test miou:{round(miou,2)}, macc:{round(macc,2)}, oa:{round(oa,2)}')
        logging.info(f'IoU for each class: {ious}')
        logging.info(f'Toronto3D_Test done in {(t2 - t1):.2f} s!')
        return True
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Scene segmentation training/testing')
    parser.add_argument('--debug', type=bool, default=False, help='setting debug mode to control not create tensorboard')
    # parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--cfg', type=str, help='config file')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args()
    args.debug = True # set debug mode
    # args.cfg = "cfgs/inv/inv_ptnet.yaml"
    args.cfg = "cfgs/inv/inv_spvcnn.yaml"
    args.profile = True

    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(vars(args))    # overwrite the default arguments in yaml 

    # NOTE: 'test'/'val' or 'resume' mode
    # cfg.mode = 'test'
    # cfg.mode = 'val'
    # cfg.mode = 'resume'
    # cfg.pretrained_path = ""
    # cfg.model_prior = ""
    cfg.seed = np.random.randint(1, 10000)

    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1 if 'spvcnn' not in args.cfg else False

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
