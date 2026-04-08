import argparse
import datetime
import json
import shutil
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from model import models_sc

from engine_pretrain import train_one_epoch

from model import logger
from model.tokenizer import GeneVocab
from datasets import load_dataset, concatenate_datasets
from data.cellxgene.data_config import ENTHNICITY_CONVERT, ENTHNICITY_DICT
from dataloader.data_collator import DataCollator


def get_args_parser():
    parser = argparse.ArgumentParser('scMETH pre-training', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU ')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--mask_ratio', default=0.4, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument('--moe', action='store_true',
                        help='Use MoE method')
    parser.set_defaults(moe=False)

    parser.add_argument('--eth_cls', action='store_true',
                        help='Classifying ethnic')
    parser.set_defaults(eth_cls=False)

    # # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=2, metavar='N',
                        help='epochs to warmup LR')

    # # Dataset parameters
    parser.add_argument('--data_path', default='/mnt/afan/scGEPOP/data/cellxgene/scb_file', type=str,
                        help='path to the data file')
    parser.add_argument('--ethnicity_path', type=str, default=None, 
                        help='path to ethnic information')
    parser.add_argument('--vocab_path', type=str, default='/mnt/afan/scGEPOP/scgepop/tokenizer/default_census_vocab.json',
                        help='path to the vocabulary file')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    # parser.add_argument('--log_dir', default='./output_dir',
    #                     help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--pad_token', type=str, default="<pad>",
                        help='value to use for padding genes')
    parser.add_argument('--pad_value', type=int, default=-2,
                        help='value to use for padding expressions')
    parser.add_argument('--valid_ratio', type=float, default=0.003,
                        help='ratio of the validation set')
    parser.add_argument('--max_seq_len', type=int, default=1536,
                        help='maximum length of the sequence')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    # parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    # parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    # parser.add_argument('--dist_url', default='env://',
    #                     help='url used to set up distributed training')
    
    parser.add_argument('--load_current_pretrained_weight', default="", type=str, help='pre-training path')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    start_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        folder_name = start_time
        if args.moe:
            folder_name += "-moe"
        if args.eth_cls:
            folder_name += "-eth"
        args.log_dir = os.path.join(args.output_dir, folder_name)
        os.makedirs(args.log_dir, exist_ok=True)
    else:
        args.log_dir = None
    
    misc.add_file_handler(logger, os.path.join(args.log_dir, "run.log"))

    logger.info(f"Running on {start_time}")
    logger.info('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    special_tokens = [args.pad_token, "<cls>", "<eoc>"]

    if args.data_path.endswith("scb_file") and args.ethnicity_path is not None:
        with open(args.ethnicity_path) as f:
            ETHNICITY_LIST = [line.rstrip('\n') for line in f]
        raw_dataset_list = []
        vocab = GeneVocab.from_file(Path(args.vocab_path))
        for idx, ethnicity in enumerate(ETHNICITY_LIST):
            ethnicity_data_path = Path(args.data_path) / ethnicity
            cls_prefix_datatable = (
                ethnicity_data_path / "all_counts" / "cls_prefix_data.parquet"
            )
            if not cls_prefix_datatable.exists():
                logger.warning(f"{cls_prefix_datatable} does not exist, skip.")
                continue
            cache_dir = ethnicity_data_path / "cache"
            ethnicity_dataset = load_dataset(
                "parquet",
                data_files=str(cls_prefix_datatable),
                split="train",
                cache_dir=str(cache_dir),
            )
            ethnicity_dataset = ethnicity_dataset.add_column('ethnicity', [ENTHNICITY_DICT[ethnicity][1]] * len(ethnicity_dataset))
            logger.info(f"Loaded {ethnicity} examples from {cls_prefix_datatable}, {len(ethnicity_dataset)} cells")
            raw_dataset_list.append(ethnicity_dataset)
        print("merging dataset...")
        raw_dataset = concatenate_datasets(raw_dataset_list)
        print("done merging dataset")
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)

    elif args.data_path.endswith("scb_file"):
        ETHNICITY_LIST = [
            'Ethiopian',
            'Japanese',
            'Finnish'
        ]
        raw_dataset_list = []
        vocab = GeneVocab.from_file(Path(args.vocab_path))
        for idx, ethnicity in enumerate(ETHNICITY_LIST):
            ethnicity_data_path = Path(args.data_path) / ethnicity
            cls_prefix_datatable = (
                ethnicity_data_path / "all_counts" / "cls_prefix_data.parquet"
            )
            if not cls_prefix_datatable.exists():
                logger.warning(f"{cls_prefix_datatable} does not exist, skip.")
                continue
            cache_dir = ethnicity_data_path / "cache"
            ethnicity_dataset = load_dataset(
                "parquet",
                data_files=str(cls_prefix_datatable),
                split="train",
                cache_dir=str(cache_dir),
            )
            ethnicity_dataset = ethnicity_dataset.add_column('ethnicity', [ENTHNICITY_DICT[ethnicity][1]] * len(ethnicity_dataset))
            logger.info(f"Loaded {ethnicity} examples from {cls_prefix_datatable}, {len(ethnicity_dataset)} cells")
            raw_dataset_list.append(ethnicity_dataset)
        print("merging dataset...")
        raw_dataset = concatenate_datasets(raw_dataset_list)
        print("done merging dataset")
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)

    with open(os.path.join(args.log_dir, "vocab.json"), "w") as f:
        json.dump(
            {token: index for token, index in vocab.get_stoi().items()},
            f,
            indent=2,
        )

    eth_cls_num = len(raw_dataset.unique("ethnicity"))

    raw_dataset = raw_dataset.with_format("torch")

    # split train and validation,
    raw_dataset = raw_dataset.train_test_split(
        test_size=args.valid_ratio, shuffle=True
    )
    dataset_train = raw_dataset["train"]
    dataset_val = raw_dataset["test"]
    logger.info(f"train set number of samples: {len(dataset_train)}")
    logger.info(f"valid set number of samples: {len(dataset_val)}")

    collator = DataCollator(
        pad_token=vocab[args.pad_token],
        pad_value=args.pad_value,
        max_length=args.max_seq_len,
        sampling=True,
    )

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        sampler=sampler_train,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        prefetch_factor=4,
    )

    criterion = nn.CrossEntropyLoss()
    
    # define the model
    model = models_sc.__dict__[args.model](vocab=vocab, moe=args.moe, cls_eth=args.eth_cls, eth_cls_num=eth_cls_num, criterion=criterion, norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    logger.info("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    logger.info("actual lr: %.2e" % args.lr)

    logger.info("accumulate grad iterations: %d" % args.accum_iter)
    logger.info("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler, logger,
            log_writer=log_writer,
            args=args
        )
        if args.log_dir and (epoch % 2 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.log_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            logger.info(log_stats)
            # with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            #     f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
