# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, logger, 
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, logger, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = {k: v.to(device, non_blocking=True) for k, v in samples.items()}

        with torch.cuda.amp.autocast():
            _, loss_mask, loss_moe, _, _, loss_eth, acc = model(samples, mask_ratio=args.mask_ratio)

        loss = loss_mask + loss_moe if args.moe else loss_mask
        loss = loss + loss_eth if args.eth_cls else loss
        loss_value = loss.item()
        if args.moe:
            loss_mask_value = loss_mask.item() 
            loss_moe_value = loss_moe.item()
        if args.eth_cls:
            loss_eth_value = loss_eth.item()
            eth_acc = acc.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        if args.moe:
            metric_logger.update(loss_mask=loss_mask_value)
            metric_logger.update(loss_moe=loss_moe_value)
        if args.eth_cls:
            metric_logger.update(loss_eth=loss_eth_value)
            metric_logger.update(eth_acc=eth_acc)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if args.moe:
            loss_mask_reduce = misc.all_reduce_mean(loss_mask)
            loss_moe_reduce = misc.all_reduce_mean(loss_moe)
        if args.eth_cls:
            loss_eth_reduce = misc.all_reduce_mean(loss_eth)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            if args.moe:
                log_writer.add_scalar('loss_mask', loss_mask_reduce, epoch_1000x)
                log_writer.add_scalar('loss_moe', loss_moe_reduce, epoch_1000x)
            if args.eth_cls:
                log_writer.add_scalar('loss_eth', loss_eth_reduce, epoch_1000x)
                log_writer.add_scalar('acc_eth', eth_acc, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}