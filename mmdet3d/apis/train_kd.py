# Copyright (c) OpenMMLab. All rights reserved.
import random
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, 
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, get_dist_info)
from mmcv.runner import EpochBasedRunner_KD # as EpochBasedRunner
from mmcv.utils import build_from_cfg
from torch import distributed as dist
from mmdet.core import multi_apply
from mmdet3d.datasets import build_dataset
from mmdet3d.utils import find_latest_checkpoint
from mmdet.core import DistEvalHook as MMDET_DistEvalHook
from mmdet.core import EvalHook as MMDET_EvalHook
from mmdet.datasets import build_dataloader as build_mmdet_dataloader
from mmdet.datasets import replace_ImageToTensor
from mmdet.utils import get_root_logger as get_mmdet_root_logger
from mmseg.core import DistEvalHook as MMSEG_DistEvalHook
from mmseg.core import EvalHook as MMSEG_EvalHook
from mmseg.datasets import build_dataloader as build_mmseg_dataloader
from mmseg.utils import get_root_logger as get_mmseg_root_logger
from collections import OrderedDict

def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, optional): The seed. Default to None.
        device (str, optional): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(f'{loss_name} is not a tensor or list of tensors')

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
    # loss = 4 * loss
    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            # if loss_name != 'acc':
            #     loss_value = 4 * loss_value
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars



def cls_loss_single(cls_score, cls_score_t, gt_labels_3d, kd_warm, kd_decay):
    labels = gt_labels_3d[0]
    indice_t = torch.max(cls_score_t, 1)[1]  # size 2048
    pos_inds = labels == indice_t
    if pos_inds.sum()==0:
        loss_hcls = torch.Tensor([0]).cuda()
    else:
        loss_hcls = BCELoss(cls_score, cls_score_t, pos_inds)
    head_cls_w = 2
    loss_hcls *= head_cls_w
    # if 'decay' in kd_cfg.type:
    loss_hcls *= kd_decay
    return loss_hcls
    # if kd_warm.get('head-cls', False):
    # loss_hcls *= 0



def batch_processor_kd(model, model_t, data, train_mode, kd_warm=dict(), kd_decay=1., epoch=0, **kwargs):
    '''cls_scores (list[torch.Tensor]): Multi-level class scores.
            bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            gt_bboxes (list[:obj:`BaseInstance3DBoxes`]): Gt bboxes
                of each sample.
            gt_labels (list[torch.Tensor]): Gt labels of each sample.''' 
    # kd_cfg = kwargs.get('kd_cfg')
    if True: # 'cls_score' in kd_cfg.type:
        num_classes = 10
        losses, head_det, gt_labels_3d = model(**data, **kwargs)
        cls_score, bbox_pred, dir_cls_preds = head_det # [0], head_det[1], head_det[2]
        cls_score_ = cls_score[0].permute(0, 2, 3, 1).reshape(-1, num_classes)
        if model_t is not None:
            _, head_det_t, _ = model(**data, **kwargs)
            cls_score_t, bbox_pred_t, dir_cls_preds_t= head_det_t # [0], head_det_t[1], head_det_t[2]
            cls_score_t_ = cls_score_t[0].permute(0, 2, 3, 1).reshape(-1, num_classes)
            # cls_loss = cls_loss_single()
            # print(cls_loss)
            # loss_hcls = multi_apply(cls_loss, cls_score, cls_score_t, gt_labels_3d, kd_warm, kd_decay)
            # print(len(cls_score_t), len(bbox_pred_t), len(dir_cls_preds_t))
            # labels = gt_labels_3d[0]

            # indice_t = torch.max(cls_score_t_, 1)[1]  # size 2048
            # print(len(labels), indice_t.shape, cls_score_t[0].shape)
            # pos_inds = labels == indice_t
            # pos_inds = (labels >= 0) & (labels < labels.max())
            # if pos_inds.sum()==0:
            #     loss_hcls = torch.Tensor([0]).cuda()
            # else:
            #     loss_hcls = BCELoss(cls_score_, cls_score_t_, pos_inds)
            loss_hcls = KLDivergenceLoss(cls_score_, cls_score_t_)
            loss_hcls *= kd_decay # kd_cfg.head_cls_w
            # if 'decay' in kd_cfg.type:
            #     loss_hcls *= kd_decay
            # if kd_warm.get('head-cls', False):
            #     loss_hcls *= 0
            losses['losskd_hcls'] = loss_hcls

    loss, log_vars = parse_losses(losses)
    # print(data.keys())
    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img_metas'].data))

    return outputs

def KLDivergenceLoss(y, teacher_scores, mask=None, T=1):
    if mask is not None:
        if mask.sum() > 0:
            p = F.log_softmax(y/T, dim=1)[mask]
            q = F.softmax(teacher_scores/T, dim=1)[mask]
            l_kl = F.kl_div(p, q, reduce=False)
            loss = torch.sum(l_kl)
            loss = loss / mask.sum()
        else:
            loss = torch.Tensor([0]).cuda()
    else:
        p = F.log_softmax(y/T, dim=1)
        q = F.softmax(teacher_scores/T, dim=1)
        l_kl = F.kl_div(p, q, reduce=False)
        loss = l_kl.sum() / l_kl.size(0)
    return loss * T**2


def BCELoss(y, teacher_scores, mask):
    p = F.softmax(y, dim=1)[mask]
    q = F.softmax(teacher_scores, dim=1)[mask]
    loss = F.binary_cross_entropy(p, q.detach()) * 10.0
    return loss


def l1loss(pred_s, pred_t, target):
    assert pred_s.size() == pred_t.size() == target.size() and target.numel() > 0
    loss_s_t = torch.abs(pred_s - pred_t).sum(1) / 4.0
    loss_s_gt = torch.abs(pred_s - target).sum(1) / 4.0
    loss = loss_s_t[loss_s_t<=loss_s_gt].sum() + loss_s_gt[loss_s_gt<loss_s_t].sum()
    return loss / target.size(0)


def l1rpnloss(pred_s, pred_t, target, weights):
    assert pred_s.size() == pred_t.size() == target.size()
    loss_s_t = torch.abs(pred_s * weights - pred_t * weights).sum(1) / 4.0
    loss_s_gt = torch.abs(pred_s * weights - target * weights).sum(1) / 4.0
    loss = loss_s_t[loss_s_t<=loss_s_gt].sum() + loss_s_gt[loss_s_gt<loss_s_t].sum()
    return loss, weights.sum()/4


def mseloss(pred_s, pred_t, target, weights):
    if weights is not None:
        pred_t = pred_t[weights.type(torch.bool)]
        pred_s = pred_s[weights.type(torch.bool)]
        if weights.sum() > 0:
            pred_s = pred_s.sigmoid()
            pred_t = pred_t.sigmoid()
            loss = F.mse_loss(pred_s, pred_t, reduction='none')
            return loss.sum(), weights.sum()
        else:
            return 0., 0.
    else:
        pred_s = pred_s.sigmoid()
        pred_t = pred_t.sigmoid()
        loss = F.mse_loss(pred_s, pred_t, reduction='none')
        return loss.sum(), loss.size(0)


def train_segmentor(model,
                    model_t,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Launch segmentor training."""
    logger = get_mmseg_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_mmseg_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        if model_t is not None:
            model_t = MMDataParallel(
                model_t.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
            '''model_t = MMDistributedDataParallel(
                model_t.cuda(),
                device_ids=[torch.cuda.current_device()], 
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)'''

    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        if model_t is not None:
            model_t = MMDistributedDataParallel(
                model_t.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)


    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            model_t=model_t, 
            # batch_processor=batch_processor_kd,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # register hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_mmseg_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = MMSEG_DistEvalHook if distributed else MMSEG_EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow) # , kd_cfg=cfg.model.hint_adapt)


def train_detector(model,
                   model_t, 
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    logger = get_mmdet_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    runner_type = 'EpochBasedRunner_KD'#  if 'runner' not in cfg else cfg.runner[
    #     'type']
    data_loaders = [
        build_mmdet_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # `num_gpus` will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            runner_type=runner_type,
            persistent_workers=cfg.data.get('persistent_workers', False))
        for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        if model_t is not None:
            model_t = MMDataParallel(
                model_t.cuda(),
                device_ids=[torch.cuda.current_device()])
            '''model_t = MMDistributedDataParallel(
                model_t.cuda(),
                device_ids=[torch.cuda.current_device()], 
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)'''

    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        if model_t is not None:
            model_t = MMDataParallel(
                model_t.cuda(),
                device_ids=cfg.gpu_ids)


    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner_KD',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            model_t=model_t, 
            # batch_processor=batch_processor_kd, 
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner_KD):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_mmdet_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = MMDET_DistEvalHook if distributed else MMDET_EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)

    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        print('LOAD: ', cfg.load_from)
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)# , kd_cfg=cfg.model.hint_adapt)


def train_model_kd(model,
                model_t, 
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """A function wrapper for launching model training according to cfg.

    Because we need different eval_hook in runner. Should be deprecated in the
    future.
    """
    if cfg.model.type in ['EncoderDecoder3D']:
        train_segmentor(
            model,
            model_t, 
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)
    else:
        train_detector(
            model,
            model_t, 
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)
