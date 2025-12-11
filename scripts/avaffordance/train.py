import math
import torch
import time
import torch.nn
import os
import random
import numpy as np
from mmengine import Config
import argparse

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import pyutils
from utils.loss_util import LossUtil
from utility import mask_iou
from utils.logger import getLogger
from model import build_model
from dataloader import build_dataset
from loss import IouSemanticAwareLoss
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import swanlab as wandb

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str, help='config file path')
    parser.add_argument('--log_dir', type=str, default='work_dir', help='log dir')
    parser.add_argument('--checkpoint_dir', type=str, default='work_dir', help='dir to save checkpoints')
    parser.add_argument("--session_name", default="AVAffordance", type=str, help="the AVAffordance setting")
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()

    # Fix seed
    FixSeed = 123 + rank  # 每个进程不同
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)

    # logger（只主进程记录）
    if rank == 0:
        log_name = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        dir_name = os.path.splitext(os.path.split(args.cfg)[-1])[0]
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, dir_name), exist_ok=True)
        log_file = os.path.join(args.log_dir, dir_name, f'{log_name}.log')
        logger = getLogger(log_file, __name__)
        logger.info(f'Load config from {args.cfg}')
        # writer = SummaryWriter(log_dir_tb)
    else:
        logger = None

    # config
    cfg = Config.fromfile(args.cfg)

    # wandb（只主进程记录）
    if rank == 0:
        wandb.init(
            project="AVAGFormer_bugfix",
            workspace="lulidong",
            config=cfg.to_dict()
        )

    if rank == 0:
        logger.info(cfg.pretty_text)
    checkpoint_dir = os.path.join(args.checkpoint_dir, os.path.splitext(os.path.split(args.cfg)[-1])[0])
    os.makedirs(checkpoint_dir, exist_ok=True)

    # model
    model = build_model(**cfg.model)
    model = model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    model.train()
    if rank == 0:
        logger.info("Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))

    # optimizer
    optimizer = pyutils.get_optimizer(model, cfg.optimizer)
    warm_up_epochs = cfg.process.warm_up_epochs
    total_epochs = cfg.process.train_epochs
    loss_util = LossUtil(**cfg.loss)
    avg_meter_miou = pyutils.AverageMeter('miou_func','miou_dep')

    # Resume
    best_epoch = 0
    global_step = 0
    miou_list = []
    max_miou = 0
    start_epoch = -1
    resume_path = os.path.join(checkpoint_dir, '%s_last_3.pth' % (args.session_name))
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        global_step = checkpoint['global_step']
        miou_list = checkpoint['miou_list']
        max_miou = checkpoint['max_miou']

    if start_epoch + 1 != 0 and (start_epoch + 1) % cfg.process.freeze_epochs == 0:
        cfg.dataset.train.batch_size = int(cfg.dataset.train.batch_size / 2)
        cfg.dataset.val.batch_size = int(cfg.dataset.val.batch_size / 2)

    # dataset
    train_dataset = build_dataset(**cfg.dataset.train)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.dataset.train.batch_size,
        sampler=train_sampler,
        num_workers=cfg.process.num_works,
        pin_memory=True,
        drop_last=True
    )
    max_step = (len(train_dataset) // cfg.dataset.train.batch_size) * cfg.process.train_epochs

    val_dataset = build_dataset(**cfg.dataset.val)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.dataset.val.batch_size,
        sampler=val_sampler,
        num_workers=cfg.process.num_works,
        pin_memory=True,
        drop_last=True
    )
    cnt = 0
    for epoch in range(start_epoch + 1, cfg.process.train_epochs):
        train_sampler.set_epoch(epoch)
        if epoch != 0 and epoch == cfg.process.freeze_epochs:
            model.module.freeze_backbone(False)

        for n_iter, batch_data in enumerate(tqdm(train_dataloader) if rank==0 else train_dataloader):
            imgs, audio, func_gt, dep_gt, img_label, _ = batch_data
            imgs = imgs.cuda(local_rank, non_blocking=True)
            audio = audio.cuda(local_rank, non_blocking=True)
            func_gt = func_gt.cuda(local_rank, non_blocking=True)
            dep_gt = dep_gt.cuda(local_rank, non_blocking=True)
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B * frame, C, H, W)
            mask_num = 1
            func_gt = func_gt.view(B * mask_num, 1, H, W)
            dep_gt = dep_gt.view(B * mask_num, 1, H, W)
            audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])
            mask_func, mask_dep, func_feat, dep_feat = model(audio, imgs)
            loss, loss_dict = IouSemanticAwareLoss(mask_func, mask_dep, func_gt, dep_gt, func_feat, dep_feat, imgs, **cfg.loss)
            loss_util.add_loss(loss, loss_dict)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            if rank == 0 and (global_step - 1) % 20 == 0:
                out, d = loss_util.pretty_out()
                train_log = 'Iter:%5d/%5d, %slr: %.6f' % (
                    global_step - 1, max_step, out, optimizer.param_groups[0]['lr'])
                logger.info(train_log)
                # writer.add_scalar('Train/Loss', loss.item(), global_step)
                # writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], global_step)
                # for key, value in d.items():
                #     writer.add_scalar(f'Train/{key}', value, global_step)
                wandb.log({"epoch": epoch, "global_step": global_step - 1, "total_step": max_step, "lr": optimizer.param_groups[0]['lr']} | d)

        # Validation:
        model.eval()
        with torch.no_grad():
            for n_iter, batch_data in enumerate(tqdm(val_dataloader) if rank==0 else val_dataloader):
                imgs, audio, func_gt, dep_gt, img_label, _ = batch_data
                imgs = imgs.cuda(local_rank, non_blocking=True)
                audio = audio.cuda(local_rank, non_blocking=True)
                func_gt = func_gt.cuda(local_rank, non_blocking=True)
                dep_gt = dep_gt.cuda(local_rank, non_blocking=True)
                B, frame, C, H, W = imgs.shape
                imgs = imgs.view(B * frame, C, H, W)
                mask_num = 1
                func_gt = func_gt.view(B * mask_num, 1, H, W)
                dep_gt = dep_gt.view(B * mask_num, 1, H, W)
                audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])
                mask_func, mask_dep, func_feat, dep_feat = model(audio, imgs)
                miou_func = mask_iou(mask_func.squeeze(1), func_gt.squeeze(1))
                miou_dep = mask_iou(mask_dep.squeeze(1), dep_gt.squeeze(1))
                avg_meter_miou.add({'miou_func': miou_func, "miou_dep": miou_dep})

            # gather miou_func/dep across all processes
            miou_func = avg_meter_miou.pop('miou_func')
            miou_dep = avg_meter_miou.pop('miou_dep')
            miou = (miou_func + miou_dep) / 2

            # 只主进程保存
            if rank == 0:
                if miou > max_miou:
                    model_save_path = os.path.join(
                        checkpoint_dir, '%s_best.pth' % (args.session_name))
                    torch.save(model.module.state_dict(), model_save_path)
                    best_epoch = epoch
                    logger.info('save best model to %s' % model_save_path)

                miou_list.append(miou)
                max_miou = max(miou_list)

                val_log = 'Epoch: {}, miou_func: {}, miou_dep: {}, maxMiou: {}'.format(
                    epoch, miou_func, miou_dep, max_miou)
                wandb.log({"epoch": epoch, "miou_func": miou_func, "miou_dep": miou_dep, "maxMiou": max_miou, "best_epoch": best_epoch})
                logger.info(val_log)
                # writer.add_scalar('Validation/Miou', miou, epoch)
                # writer.add_scalar('Validation/MaxMiou', max_miou, epoch)
                # writer.add_scalar('Validation/BestEpoch', best_epoch, epoch)

        model.train()
        if rank == 0:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "global_step": global_step,
                "miou_list": miou_list,
                "max_miou": max_miou
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, '%s_last_3.pth' % (args.session_name)))
        cnt += 1
        if epoch != 0 and epoch == cfg.process.freeze_epochs:
            break
    if rank == 0:
        logger.info('best val Miou {} at epoch: {}'.format(max_miou, best_epoch))
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
