"""
LPFP(*): construct a large-scale model for the base model.
"""
import __init__
import argparse, yaml, os, logging, numpy as np
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist, multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port #, Wandb
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from openpoints.dataset import build_dataloader_from_cfg, get_scene_seg_features, get_class_weights
from openpoints.dataset.data_util import voxelize
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
from main.test_utils import write_to_csv, test_s3dis, validate_sphere
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
        setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME) # all mode create log(except debug mode), cfg.log_path in ./log/S3DIS/...
    if cfg.rank == 0 and (not cfg.debug):
        writer = SummaryWriter(log_dir=os.path.join(cfg.run_dir, 'tensorboard')) if cfg.is_training else None # only training create tensorboard
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels

    # NOTE: create model from cfg file
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')

    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

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
    validate_fn = validate if 'sphere' not in cfg.dataset.common.NAME.lower() else validate_sphere
    
    # optionally resume from a checkpoint
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler, pretrained_path=cfg.pretrained_path)
            val_miou = validate_fn(model, val_loader, cfg)
            logging.info(f'\nresume val miou is {val_miou}\n ')
            # resume the SummaryWriter instance from the crashed epoch
            if cfg.rank == 0:
                writer = SummaryWriter(log_dir=os.path.join(cfg.run_dir, 'tensorboard'), purge_step=cfg.start_epoch)
        else:
            if cfg.mode == 'val':
                best_epoch, best_val = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
                val_miou, val_macc, val_oa, val_ious, val_accs = validate_fn(model, val_loader, cfg, num_votes=1)
                with np.printoptions(precision=2, suppress=True):
                    logging.info(
                        f'Best ckpt @E{best_epoch},  val_oa , val_macc, val_miou: {val_oa:.2f} {val_macc:.2f} {val_miou:.2f}, '
                        f'\niou per cls is: {val_ious}')
                return miou 
            elif cfg.mode == 'test':
                best_epoch, best_val = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
                miou, macc, oa, ious, accs, _ = test_s3dis(model, cfg.dataset.common.test_area, cfg)
                with np.printoptions(precision=2, suppress=True):
                    logging.info(
                        f'Best ckpt @E{best_epoch},  test_oa , test_macc, test_miou: {oa:.2f} {macc:.2f} {miou:.2f}, '
                        f'\niou per cls is: {ious}')
                cfg.csv_path = os.path.join(cfg.run_dir, cfg.run_name + '_test.csv')
                write_to_csv(oa, macc, miou, ious, best_epoch, cfg)
                return miou 
            else:
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                model.load_model_from_ckpt(cfg.pretrained_path, only_encoder=cfg.only_encoder)
    else:
        logging.info('Training from scratch')

    train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             )
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")
    
    # build criterion without weight and with weight
    cfg.criterion.weight = None 
    criterion_wo_weight = build_criterion_from_cfg(cfg.criterion).cuda()
    criterion_w_weight = None
    if cfg.get('cls_weighed_loss', False):
        if hasattr(train_loader.dataset, 'num_per_class'):    
            cfg.criterion.weight = get_class_weights(train_loader.dataset.num_per_class, normalize=False)
        else: 
            logging.info('`num_per_class` attribute is not founded in dataset')
        criterion_w_weight = build_criterion_from_cfg(cfg.criterion).cuda()
    criterion_lovasz = None
    if cfg.get('lovasz', False):
        from openpoints.loss import lovasz_softmax
        criterion_lovasz = lovasz_softmax

    # ===> start training
    val_miou, val_macc, val_oa, val_ious, val_accs = 0., 0., 0., [], []
    best_val, macc_when_best, oa_when_best, ious_when_best, best_epoch = 0., 0., 0., [], 0
    

    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'epoch'):  # some dataset sets the dataset length as a fixed steps.
            train_loader.dataset.epoch = epoch - 1
        train_inter_loss, train_inter_lovasz_loss, train_loss, train_miou, train_macc, train_oa, _, _ = \
            train_one_epoch(model, train_loader, criterion_wo_weight, criterion_w_weight, criterion_lovasz, optimizer, scheduler, epoch, cfg)
        train_inter_loss_dic = dict(zip([f'inter_loss{i}' for i in range(cfg.model.stacked_num)], train_inter_loss))
        if cfg.get('lovasz', False):
            train_inter_lovasz_loss_dic = dict(zip([f'inter_lovasz_loss{i}' for i in range(cfg.model.stacked_num)], train_inter_lovasz_loss))
        is_best = False
        if epoch % cfg.val_freq == 0:
            val_miou, val_macc, val_oa, val_ious, val_accs = validate_fn(model, val_loader, cfg)
            with np.printoptions(precision=2, suppress=True):
                logging.info(f'\nval iou per cls is: {val_ious}')
            if val_miou > best_val:
                is_best = True
                best_val = val_miou
                macc_when_best = val_macc
                oa_when_best = val_oa
                ious_when_best = val_ious
                best_epoch = epoch
                with np.printoptions(precision=2, suppress=True):
                    logging.info(f'Find a better ckpt @E{epoch}, val_miou {val_miou:.2f} val_macc {macc_when_best:.2f}, val_oa {oa_when_best:.2f}')
        
        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} LR {lr:.6f} '
                     f'train_miou {train_miou:.2f}, val_miou {val_miou:.2f}, best val miou {best_val:.2f}')
        #NOTE: record each class iou/acc for each epoch
        val_ious_dic = dict(zip([f'Class_{i}' for i in range(cfg.num_classes)], val_ious)) 
        val_accs_dic = dict(zip([f'Class_{i}' for i in range(cfg.num_classes)], val_accs)) 
        if writer is not None:
            writer.add_scalar('val/best_val', best_val, epoch)
            writer.add_scalar('val/val_miou', val_miou, epoch)
            writer.add_scalar('val/macc_when_best', macc_when_best, epoch)
            writer.add_scalar('val/oa_when_best', oa_when_best, epoch)
            writer.add_scalar('val/val_macc', val_macc, epoch)
            writer.add_scalar('val/val_oa', val_oa, epoch)
            writer.add_scalars('val/val_ious', val_ious_dic, epoch)
            writer.add_scalars('val/val_accs', val_accs_dic, epoch)
            writer.add_scalars("train/train_inter_loss", train_inter_loss_dic, epoch) # add inter loss
            if cfg.get('lovasz', False):
                writer.add_scalars("train/train_inter_lovasz_loss", train_inter_lovasz_loss_dic, epoch) # add inter lovasz loss
            writer.add_scalar('train/train_loss', train_loss, epoch)
            writer.add_scalar('train/train_miou', train_miou, epoch)
            writer.add_scalar('train/train_macc', train_macc, epoch)
            writer.add_scalar('train/lr', lr, epoch)
            writer.flush()

        if cfg.sched_on_epoch:
            scheduler.step(epoch)
        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val': best_val},
                            is_best=is_best
                            )
            is_best = False

    # validate
    with np.printoptions(precision=2, suppress=True):
        logging.info(
            f'Best ckpt @E{best_epoch},  val_oa {oa_when_best:.2f}, val_macc {macc_when_best:.2f}, val_miou {best_val:.2f}, '
            f'\nval iou per cls is: {ious_when_best}')
    # test
    if cfg.task_name == "semantic3d":
        return True
    if cfg.rank == 0: # NOTE: only test on the main process, because in the ddp mode, the test code will be used for repeated times
        load_checkpoint(model, pretrained_path=os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
        cfg.csv_path = os.path.join(cfg.run_dir, cfg.run_name + f'_area5.csv')
        if 'sphere' in cfg.dataset.common.NAME.lower():
            test_miou, test_macc, test_oa, test_ious, test_accs = validate_sphere(model, val_loader, cfg)
        else:
            test_miou, test_macc, test_oa, test_ious, test_accs, _  = test_s3dis(model, cfg.dataset.common.test_area, cfg)
        with np.printoptions(precision=2, suppress=True):
            logging.info(f'Best ckpt @E{best_epoch},  test_oa {test_oa:.2f}, test_macc {test_macc:.2f}, test_miou {test_miou:.2f}, '
                        f'\ntest iou per cls is: {test_ious}')
        if writer is not None:
            writer.add_scalar('test/test_miou', test_miou, epoch)
            writer.add_scalar('test/test_macc', test_macc, epoch)
            writer.add_scalar('test/test_oa', test_oa, epoch)
        write_to_csv(test_oa, test_macc, test_miou, test_ious, best_epoch, cfg, write_header=True)
        logging.info(f'save results in {cfg.csv_path}')
        writer.close()
    return True


def train_one_epoch(model, train_loader, criterion_wo_weight, criterion_w_weight, criterion_lovasz, optimizer, scheduler, epoch, cfg):
    if criterion_w_weight==None:
        criterion_w_weight = criterion_wo_weight
    loss_meter = AverageMeter()
    inter_loss_meter = [AverageMeter() for _ in range(cfg.model.stacked_num)] 
    if cfg.get('lovasz', False):
        inter_lovasz_loss_meter = [AverageMeter() for _ in range(cfg.model.stacked_num)] 
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0

    #NOTE: loss0*1/4 + loss1*1/3 + loss2*1/2 + loss3(if cfg.model.stacked_num=4)
    loss_weight_list = [1/(cfg.model.stacked_num-i) for i in range(cfg.model.stacked_num)]
    
    for idx, data in pbar:
        # if idx > 5: break # debug
        # Semantic3D need to ignore 'unlabeled' class
        if ('Semantic3D' in cfg.dataset.common.NAME) or ('SemanticKITTI' in cfg.dataset.common.NAME):
            data['mask'] = ~(data['y']==0) 
        elif 'scannetv2' in cfg.dataset.common.NAME:
            data['mask'] = ~(data['y']==255) 
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            if not isinstance(data[key], list):
                data[key] = data[key].cuda(non_blocking=True)
            elif torch.is_tensor(data[key][0]):
                for i in range(len(data[key])):
                    data[key][i] = data[key][i].cuda(non_blocking=True)
        num_iter += 1
        target = data['y'].squeeze(-1)
        if cfg.model.get('dim_modifed', True):
            if len(data['x'].shape) > 2:
                data['x'] = data['x'].transpose(1, 2).contiguous()
            feature_mode = cfg.model.get('feature_mode', None)
            data['x'] = get_scene_seg_features(cfg.model.in_channels, data['pos'], data['x'], feature_mode=feature_mode)
        
        logits, _, _, _ = model(data)
        if isinstance(logits, list):
            # in pointtransformer, all points are concatnated into the first dim without batch dim, so deal with the target value
            if len(logits[0].size())==2 and len(target.size())>1: target = torch.cat([target_split.squeeze() for target_split in target.split(1,0)])
        # NOTE: all the intermediate loss need to be applied
        loss_list, lovasz_loss_list = [], []
        if 'mask' not in cfg.criterion.NAME.lower():
            for i in range(len(logits)):
                if cfg.cls_weighed_loss:
                    loss_list.append(criterion_w_weight(logits[i], target))
                else:
                    loss_list.append(criterion_wo_weight(logits[i], target))
                if cfg.get('lovasz', False):
                    lovasz_loss_list.append(criterion_lovasz(logits[i], target))
        else: 
            for i in range(len(logits)):
                if cfg.cls_weighed_loss:
                    loss_list.append(criterion_w_weight(logits[i], target, data['mask']))
                else:
                    loss_list.append(criterion_wo_weight(logits[i], target, data['mask']))
                if cfg.get('lovasz', False):
                    lovasz_loss_list.append(criterion_lovasz(logits[i], target, data['mask']))
        judge_nan = np.array([temp.detach().cpu().numpy() for temp in loss_list])

        loss, lovasz_loss = 0, 0
        if 'lovasz_weight' not in cfg:
            cfg.lovasz_weight = 0
        for i in range(len(loss_list)):
            loss_list[i] = loss_list[i] * loss_weight_list[i]
            loss += loss_list[i]
            if cfg.get('lovasz', False):
                lovasz_loss_list[i] = lovasz_loss_list[i] * loss_weight_list[i]
                lovasz_loss += lovasz_loss_list[i]
        loss = (1-cfg.lovasz_weight) * loss + cfg.lovasz_weight * lovasz_loss
        if not np.isnan(judge_nan).any():
            loss.backward()
        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0
            optimizer.step()
            optimizer.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        # update confusion matrix
        predits = logits[-1].argmax(dim=1)
        if ('Semantic3D' in cfg.dataset.common.NAME) or ('SemanticKITTI' in cfg.dataset.common.NAME):
            predits = predits + 1 # restore mapping in training: from [0,7] to [1,8] 
        elif 'scannetv2' in cfg.dataset.common.NAME:
            predits = predits[data['mask']]
            target = target[data['mask']]
        cm.update(predits, target)
        if not np.isnan(judge_nan).any():
            loss_meter.update(loss.item())
            for i in range(len(loss_list)):
                inter_loss_meter[i].update(loss_list[i].item())
                if cfg.get('lovasz', False):
                    inter_lovasz_loss_meter[i].update(lovasz_loss_list[i].item())

        if idx % cfg.print_freq:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                f"Loss {loss_meter.val:.3f} Acc {cm.overall_accuray:.2f}")
    miou, macc, oa, ious, accs = cm.all_metrics()
    inter_loss_avg = [inter_loss_meter[i].avg for i in range(len(loss_list))]
    inter_lovasz_loss_avg = None
    if cfg.get('lovasz', False):
        inter_lovasz_loss_avg = [inter_lovasz_loss_meter[i].avg for i in range(len(lovasz_loss_list))]
    return inter_loss_avg, inter_lovasz_loss_avg, loss_meter.avg, miou, macc, oa, ious, accs


@torch.no_grad()
def validate(model, val_loader, cfg):
    torch.cuda.empty_cache()
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    for idx, data in pbar:
        # if idx > 2 : break # debug
        if ('Semantic3D' in cfg.dataset.common.NAME) or ('SemanticKITTI' in cfg.dataset.common.NAME):
            data['mask'] = ~(data['y']==0) 
        elif 'scannetv2' in cfg.dataset.common.NAME:
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
                data['x'] = data['x'].transpose(1, 2).contiguous()
            feature_mode = cfg.model.get('feature_mode', None)
            data['x'] = get_scene_seg_features(cfg.model.in_channels, data['pos'], data['x'], feature_mode=feature_mode)

        logits, _, _, _ = model(data)
        if isinstance(logits, list):
            if len(logits[0].size())==2 and len(target.size())>1: target = torch.cat([target_split.squeeze() for target_split in target.split(1,0)])
            logits = logits[-1]
        # Semantic3D need to ignore 'unlabeled' class
        predits = logits.argmax(dim=1)
        if ('Semantic3D' in cfg.dataset.common.NAME) or ('SemanticKITTI' in cfg.dataset.common.NAME):
            predits = predits + 1 # restore mapping in training: from [0,7] to [1,8] 
        elif 'scannetv2' in cfg.dataset.common.NAME:
            predits = predits[data['mask']]
            target = target[data['mask']]
        cm.update(predits, target)
    tp, union, count = cm.tp, cm.union, cm.count
    if cfg.distributed:
        dist.all_reduce(tp), dist.all_reduce(union), dist.all_reduce(count)
    miou, macc, oa, ious, accs = get_mious(tp, union, count)
    return miou, macc, oa, ious, accs


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Scene segmentation training/testing')
    parser.add_argument('--debug', type=bool, default=False, help='setting debug mode to control not create tensorboard and wandb')
    # parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--cfg', type=str, help='config file')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args()
    # args.debug = True # set debug mode
    # NOTE: cfg file path read
    # args.cfg = "cfgs/lpfp_pcln/lpfp_pointnet++_s.yaml"
    # args.cfg = "cfgs/lpfp_pcln/lpfp_pointnet++_t.yaml"
    # args.cfg = "cfgs/lpfp_pcln/lpfp_ptnet_setmodify_s.yaml"
    args.cfg = "cfgs/lpfp_pcln/lpfp_ptnet_setmodify_t.yaml"
    args.profile = True

    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)    # overwrite the default arguments in yml 
    cfg.debug = args.debug
    cfg.wandb.use_wandb = False # online wandb will slow training speed

    # NOTE: 'test' or 'resume' mode
    # cfg.mode = 'test'
    # cfg.mode = 'resume' # if it's need to resume training model
    # cfg.pretrained_path = ""
    
    if cfg.seed is None:
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
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            tags.append(opt)
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

    # multi processing
    if cfg.mp:
        port = find_free_port()
        cfg.dist_url = f"tcp://localhost:{port}"
        print('using mp spawn for distributed training')
        mp.spawn(main, nprocs=cfg.world_size, args=(cfg,))
    else:
        main(0, cfg)
