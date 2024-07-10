"""
PCLN(*): lightweight the large-scale model.
"""
import __init__
import argparse, yaml, os, logging, numpy as np
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist, multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from openpoints.dataset import build_dataloader_from_cfg, get_scene_seg_features
from openpoints.dataset.data_util import voxelize
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
from main.test_utils import write_to_csv, test_s3dis
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
        Wandb.launch(cfg, cfg.wandb.use_wandb) # all wandb setting is here, it's sync with tensorboard
        writer = SummaryWriter(log_dir=os.path.join(cfg.run_dir, 'tensorboard')) if cfg.is_training else None # only training create tensorboard
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    if cfg.student_model.get('in_channels', None) is None:
        cfg.student_model.in_channels = cfg.student_model.encoder_args.in_channels

    # NOTE: create Student model from cfg file
    teacher_model = build_model_from_cfg(cfg.teacher_model).to(cfg.rank)
    teacher_model_size = cal_model_parm_nums(teacher_model)
    student_model = build_model_from_cfg(cfg.student_model).to(cfg.rank)
    student_model_size = cal_model_parm_nums(student_model)
    logging.info(student_model)
    logging.info('\nNumber of Teacher_Net params: {:.4f} M\nNumber of Student_Net params: {:.4f} M'\
        .format((teacher_model_size / 1e6), (student_model_size / 1e6)))

    if cfg.sync_bn:
        teacher_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher_model)
        student_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student_model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        teacher_model = nn.parallel.DistributedDataParallel(teacher_model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        student_model = nn.parallel.DistributedDataParallel(student_model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')
    
    # load the teacher model
    _, _ = load_checkpoint(teacher_model, cfg.teacher_model.pretrain_path)
    logging.info(f'Successful Loading the ckpt from {cfg.teacher_model.pretrain_path} as teacher_model')

    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(student_model, lr=cfg.lr, **cfg.optimizer)
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
    validate_fn = validate
    
    # optionally resume from a checkpoint
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, student_model, optimizer, scheduler, pretrained_path=cfg.pretrained_path)
            val_miou = validate_fn(student_model, val_loader, cfg)
            logging.info(f'\nresume val miou is {val_miou}\n ')
            # resume the SummaryWriter instance from the crashed epoch
            if cfg.rank == 0:
                writer = SummaryWriter(log_dir=os.path.join(cfg.run_dir, 'tensorboard'), purge_step=cfg.start_epoch)
        else:
            if cfg.mode == 'val':
                best_epoch, best_val = load_checkpoint(student_model, pretrained_path=cfg.pretrained_path)
                val_miou, val_macc, val_oa, val_ious, val_accs = validate_fn(student_model, val_loader, cfg, num_votes=1)
                with np.printoptions(precision=2, suppress=True):
                    logging.info(
                        f'Best ckpt @E{best_epoch},  val_oa , val_macc, val_miou: {val_oa:.2f} {val_macc:.2f} {val_miou:.2f}, '
                        f'\niou per cls is: {val_ious}')
                return miou 
            elif cfg.mode == 'test':
                best_epoch, best_val = load_checkpoint(student_model, pretrained_path=cfg.pretrained_path)
                miou, macc, oa, ious, accs, _ = test_s3dis(student_model, cfg.dataset.common.test_area, cfg)
                with np.printoptions(precision=2, suppress=True):
                    logging.info(
                        f'Best ckpt @E{best_epoch},  test_oa , test_macc, test_miou: {oa:.2f} {macc:.2f} {miou:.2f}, '
                        f'\niou per cls is: {ious}')
                cfg.csv_path = os.path.join(cfg.run_dir, cfg.run_name + '_test.csv')
                write_to_csv(oa, macc, miou, ious, best_epoch, cfg)
                return miou 
            else:
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                best_epoch, best_val = load_checkpoint(student_model, pretrained_path=cfg.pretrained_path)
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
    
    # build criterion with 
    criterion_kd_affinity = None # structure kd loss
    criterion_at = None # attention transfer loss
    criterion = build_criterion_from_cfg(cfg.criterion)
    criterion_kd = build_criterion_from_cfg(cfg.criterion_kd)
    if cfg.criterion_kd.structure:
        criterion_kd_affinity = nn.SmoothL1Loss(reduction='mean')
    if cfg.criterion_kd.at:
        criterion_at = nn.SmoothL1Loss(reduction='mean')

    # ===> start training
    val_miou, val_macc, val_oa, val_ious, val_accs = 0., 0., 0., [], []
    best_val, macc_when_best, oa_when_best, ious_when_best, best_epoch = 0., 0., 0., [], 0
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'epoch'):  # some dataset sets the dataset length as a fixed steps.
            train_loader.dataset.epoch = epoch - 1
        train_inter_loss, train_inter_kd_loss, train_inter_affinity_loss, train_inter_at_loss, train_loss, train_miou, train_macc, train_oa, _, _ \
            = train_one_epoch(student_model, teacher_model, train_loader, criterion, \
                criterion_kd, criterion_kd_affinity, criterion_at, optimizer, scheduler, epoch, cfg)
        train_inter_loss_dic = dict(zip([f'inter_loss{i}' for i in range(len(train_inter_loss))], train_inter_loss))
        if train_inter_kd_loss:
            train_inter_kd_loss_dic = dict(zip([f'inter_kd_loss{i}' for i in range(len(train_inter_kd_loss))], train_inter_kd_loss))
        if train_inter_affinity_loss:
            train_inter_affinity_loss_dic = dict(zip([f'inter_affinity_loss{i}' \
                for i in range(len(train_inter_affinity_loss))], train_inter_affinity_loss))
        if train_inter_at_loss:
            train_inter_at_loss_dic = dict(zip([f'inter_at_loss{i}' \
                for i in range(len(train_inter_at_loss))], train_inter_at_loss))

        is_best = False
        if epoch % cfg.val_freq == 0:
            val_miou, val_macc, val_oa, val_ious, val_accs = validate_fn(student_model, val_loader, cfg)
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
        # NOTE: record each class iou/acc for each epoch
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
            if train_inter_kd_loss: # add inter kd loss in kd training
                writer.add_scalars("train/train_inter_kd_loss", train_inter_kd_loss_dic, epoch) 
            if train_inter_affinity_loss: # add inter affinity loss in kd training
                writer.add_scalars("train/train_inter_affinity_loss", train_inter_affinity_loss_dic, epoch) 
            if train_inter_at_loss: # add inter affinity loss in kd training
                writer.add_scalars("train/train_inter_at_loss", train_inter_at_loss_dic, epoch)
            writer.add_scalar('train/train_loss', train_loss, epoch)
            writer.add_scalar('train/train_miou', train_miou, epoch)
            writer.add_scalar('train/train_macc', train_macc, epoch)
            writer.add_scalar('train/lr', lr, epoch)
            writer.flush()

        if cfg.sched_on_epoch:
            scheduler.step(epoch)
        if cfg.rank == 0:
            save_checkpoint(cfg, student_model, epoch, optimizer, scheduler,
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
    if cfg.rank == 0: # NOTE: only test on the main process, because in the ddp mode, the test code will be used for repeated times
        load_checkpoint(student_model, pretrained_path=os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
        cfg.csv_path = os.path.join(cfg.run_dir, cfg.run_name + f'_area5.csv')
        test_miou, test_macc, test_oa, test_ious, _, _  = test_s3dis(student_model, cfg.dataset.common.test_area, cfg)
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


def train_one_epoch(student_model, teacher_model, train_loader, criterion, \
    criterion_kd, criterion_kd_affinity, criterion_at, optimizer, scheduler, epoch, cfg):
    kd = cfg.criterion_kd.kd
    kd_struct = cfg.criterion_kd.structure
    at = cfg.criterion_kd.at
    st_keep = (cfg.student_model.stacked_num == 1)
    
    loss_meter = AverageMeter()
    # intermediate loss metric register
    inter_loss_meter = [AverageMeter() for _ in range(cfg.student_model.stacked_num)]
    # intermediate kd loss metric register
    if kd:
        inter_kd_loss_meter = [AverageMeter() for _ in range(len(kd))]
    # intermediate structure kd loss metric register
    if kd_struct:
        inter_affinity_loss_meter = [AverageMeter() for _ in range(len(kd_struct))]
    # intermediate at loss metric register
    if at:
        inter_at_loss_meter = [AverageMeter() for _ in range(len(at))]

    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    student_model.train()  # set student_model to training mode
    teacher_model.eval()  # set teacher_model to eval mode
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0

    #NOTE: loss0*1/4 + loss1*1/3 + loss2*1/2 + loss3(if cfg.student_model.stacked_num=4)
    loss_weight_list = [1/(4-i) for i in range(4)]
    # # NOTE NOT NORMALIZATION!!!
    # loss_weight_list = (loss_weight_list * 4 / sum(loss_weight_list)).tolist()
    # alpha = (np.array([1/16, 0.8, 1]) * 3 / sum(np.array([1/16, 0.8, 1]))).tolist() # params used for st-kd loss
    alpha = cfg.alpha # params used for st-kd loss
    beta = cfg.beta # params used for at loss
    gamma = cfg.gamma # params used for kd loss

    for idx, data in pbar:
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            if not isinstance(data[key], list):
                data[key] = data[key].cuda(non_blocking=True)
            elif torch.is_tensor(data[key][0]):
                for i in range(len(data[key])):
                    data[key][i] = data[key][i].cuda(non_blocking=True)
        num_iter += 1
        target = data['y'].squeeze(-1)

        if len(data['x'].shape) > 2:
            data['x'] = data['x'].transpose(1, 2)
        data['x'] = get_scene_seg_features(cfg.student_model.in_channels, data['pos'], data['x'])

        with torch.no_grad():
            teacher_logits, teacher_affinity_matrix, query_xyz, teacher_at = teacher_model(data, kd_struct=kd_struct, at=at)
        student_logits, student_affinity_matrix, _, student_at = student_model(data, query_xyz=query_xyz, kd_struct=kd_struct, at=at)
        if isinstance(student_logits, list):
            # in pointtransformer, all points are concatnated into the first dim without batch dim, so deal with the target value
            if len(student_logits[0].size())==2: target = torch.cat([target_split.squeeze() for target_split in target.split(1,0)])
        
        # NOTE: all the intermediate loss need to be applied
        loss_list, kd_loss_list, affinity_loss_list, at_loss_list = [], [], [], []
        if 'mask' not in cfg.criterion.NAME.lower():
            for i in range(len(student_logits)):
                loss_list.append(criterion(student_logits[i], target))
            if kd:
                for i in kd:
                    kd_loss_list.append(criterion_kd(student_logits[i], teacher_logits[i]))
            if kd_struct:
                for i in range(len(teacher_affinity_matrix)):
                    affinity_loss_list.append(criterion_kd_affinity(student_affinity_matrix[i], teacher_affinity_matrix[i].detach()))
            if at:
                for i in range(len(teacher_at)):
                    at_loss_list.append(criterion_at(student_at[i], teacher_at[i].detach()))
        else: 
            # TODO: if mask work, complete here
            for i in range(len(student_logits)):
                loss_list.append(criterion(student_logits[i], target, data['mask']))
    
        loss, kd_loss, affinity_loss, at_loss = 0, 0, 0, 0
        for i in range(len(loss_list)):
            loss_list[i] = loss_list[i] * loss_weight_list[i]
            loss = loss + loss_list[i]
        if kd:
            for i, element in enumerate(kd):
                kd_loss_list[i] = kd_loss_list[i] * gamma[element] * loss_weight_list[element]
                kd_loss = kd_loss + kd_loss_list[i]
        if kd_struct:
            for i, element in enumerate(kd_struct):
                affinity_loss_list[i] = affinity_loss_list[i] * alpha[element] * loss_weight_list[element]
                affinity_loss = affinity_loss + affinity_loss_list[i] 
        if at:
            for i, element in enumerate(at):
                at_loss_list[i] = at_loss_list[i] * beta[element] * loss_weight_list[element]
                at_loss = at_loss + at_loss_list[i]
        loss = cfg.criterion_kd.loss_weight * loss + cfg.criterion_kd.kd_weight * kd_loss + \
            cfg.criterion_kd.structure_weight * affinity_loss + cfg.criterion_kd.at_weight * at_loss
        loss.backward()

        loss = loss.detach()
        if cfg.distributed:
            dist.all_reduce(loss)
            loss /= cfg.world_size
        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0
            optimizer.step()
            optimizer.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        # update confusion matrix
        cm.update(student_logits[-1].argmax(dim=1), target)
        loss_meter.update(loss.item())
        for i in range(len(loss_list)):
            inter_loss_meter[i].update(loss_list[i].item())
        for i in range(len(kd_loss_list)):
            inter_kd_loss_meter[i].update(kd_loss_list[i].item())
        if kd_struct:
            for i in range(len(affinity_loss_list)):
                inter_affinity_loss_meter[i].update(affinity_loss_list[i].item())
        if at:
            for i in range(len(at_loss_list)):
                inter_at_loss_meter[i].update(at_loss_list[i].item())
        if idx % cfg.print_freq:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                f"Loss {loss_meter.val:.3f} Acc {cm.overall_accuray:.2f}")
    
    miou, macc, oa, ious, accs = cm.all_metrics()
    inter_loss_avg = [inter_loss_meter[i].avg for i in range(cfg.student_model.stacked_num)]
    inter_affinity_loss_avg = None
    inter_at_loss_avg = None
    inter_kd_loss_avg = None
    if kd:
        inter_kd_loss_avg = [inter_kd_loss_meter[i].avg for i in range(len(kd))]
    if kd_struct:
        inter_affinity_loss_avg = [inter_affinity_loss_meter[i].avg for i in range(len(kd_struct))]
    if at:
        inter_at_loss_avg = [inter_at_loss_meter[i].avg for i in range(len(at))]
    return inter_loss_avg, inter_kd_loss_avg, inter_affinity_loss_avg, inter_at_loss_avg, loss_meter.avg, miou, macc, oa, ious, accs


@torch.no_grad()
def validate(model, val_loader, cfg):
    torch.cuda.empty_cache()
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    for idx, data in pbar:
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            if not isinstance(data[key], list):
                data[key] = data[key].cuda(non_blocking=True)
            elif torch.is_tensor(data[key][0]):
                for i in range(len(data[key])):
                    data[key][i] = data[key][i].cuda(non_blocking=True)
        target = data['y'].squeeze(-1)
        if len(data['x'].shape) > 2:
            data['x'] = data['x'].transpose(1, 2)
        data['x'] = get_scene_seg_features(cfg.student_model.in_channels, data['pos'], data['x'])

        logits, _, _, _ = model(data)
        if isinstance(logits, list):
            # in pointtransformer, all points are concatnated into the first dim without batch dim, so deal with the target value
            if len(logits[0].size())==2: target = torch.cat([target_split.squeeze() for target_split in target.split(1,0)])
            logits = logits[-1]
        cm.update(logits.argmax(dim=1), target)
    tp, union, count = cm.tp, cm.union, cm.count
    if cfg.distributed:
        dist.all_reduce(tp), dist.all_reduce(union), dist.all_reduce(count)
    miou, macc, oa, ious, accs = get_mious(tp, union, count)
    return miou, macc, oa, ious, accs


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Scene segmentation training/testing')
    parser.add_argument('--debug', type=bool, default=False, help='setting debug mode to control not create tensorboard')
    # parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--cfg', type=str, help='config file')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args()
    # args.debug = True # set debug mode
    # NOTE: cfg file path read
    args.cfg = "cfgs/lpfp_pcln/pcln_pointnet++.yaml.yaml"
    # args.cfg = "cfgs/s3dis_kd/pointtransformer_stacked_kd.yaml"
    args.profile = True

    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)    # overwrite the default arguments in yml 
    cfg.debug = args.debug

    # NOTE: 'test', 'resume' or 'finetune' mode
    # cfg.mode = 'test'
    # cfg.mode = 'resume' # if it's need to resume training model
    # cfg.mode = 'finetune'
    # cfg.pretrained_path = ""

    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)

    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]  # task/dataset name, \eg s3dis
    cfg.cfg_basename = args.cfg.split('.')[-2].split('/')[-1]   # cfg_basename, \eg pointnext-xl 
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.cfg_basename,  # cfg file name
        f'ngpus{cfg.world_size}',
        f'seed{cfg.seed}',
    ]
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'path' not in opt and '/' not in opt:
            tags.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)

    cfg.is_training = cfg.mode in ['train', 'training', 'finetune', 'finetuning']
    if cfg.mode in ['train','finetune']:
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
