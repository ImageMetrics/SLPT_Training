import argparse

from Config.default_cal import _C as cfg
from Config.default_cal import update_config

from utils import create_logger
from utils import save_checkpoint
from model import Sparse_alignment_network_refine
from Dataloader.WFLW_loader import WFLWCal_Dataset
from backbone import Alignment_Loss
from utils import get_optimizer
from tools.train_function import train_cal_refine
from tools.validate_function import validate_cal_refine

from tensorboardX import SummaryWriter

import torch
import pprint
import os

import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Train Sparse Facial Network')

    # philly
    parser.add_argument('--modelDir', help='model directory', type=str, default='./Checkpoint')
    parser.add_argument('--logDir', help='log directory', type=str, default='./log')
    parser.add_argument('--dataDir', help='data directory', type=str, default='./')
    parser.add_argument('--target', help='targeted branch (alignmengt, emotion or pose)',
                        type=str, default='alignment')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default=None)

    args = parser.parse_args()

    return args


def main_function():
    args = parse_args()
    update_config(cfg, args)
    logger, final_output_dir, tb_log_dir = create_logger(cfg, cfg.TARGET)
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # for development
    import socket
    if socket.gethostname() == 'IMLAP64':
        cfg.defrost()
        cfg.WORKERS = 0
        cfg.TRAIN.BATCH_SIZE_PER_GPU = 2

    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    if cfg.DATASET.DATASET == 'HEADCAMCAL':
        model = Sparse_alignment_network_refine(cfg.HEADCAMCAL.NUM_POINT, cfg.MODEL.OUT_DIM,
                                         cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
                                         cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
                                         cfg.TRANSFORMER.FEED_DIM, cfg.HEADCAMCAL.INITIAL_PATH, cfg)
    else:
        raise ValueError('Wrong Dataset')

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    loss_function = Alignment_Loss(cfg).cuda()
    from backbone.Loss import Consistency_Loss
    consistency_loss_function = Consistency_Loss().cuda()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if cfg.DATASET.DATASET == 'HEADCAMCAL':
        train_dataset = WFLWCal_Dataset(
            cfg, cfg.HEADCAMCAL.ROOT, True,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]),
            annotation_file = os.path.join(cfg.HEADCAMCAL.ROOT,
                                           'HEADCAMCAL_annotations', 'list_85pt_rect_attr_train_test',
                                           'list_85pt_rect_attr_train.txt'),
            calibration_annotation_file = os.path.join(cfg.HEADCAMCAL.ROOT,
                                           'HEADCAMCAL_annotations', 'list_85pt_rect_attr_train_test',
                                           'list_85pt_rect_attr_calibration_train.txt'),
            wflw_config = cfg.HEADCAMCAL,
        )

        valid_dataset = WFLWCal_Dataset(
            cfg, cfg.HEADCAMCAL.ROOT, False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]),
            annotation_file = os.path.join(cfg.HEADCAMCAL.ROOT,
                                       'HEADCAMCAL_annotations', 'list_85pt_rect_attr_train_test',
                                       'list_85pt_rect_attr_test.txt'),
            calibration_annotation_file = os.path.join(cfg.HEADCAMCAL.ROOT,
                                           'HEADCAMCAL_annotations', 'list_85pt_rect_attr_train_test',
                                           'list_85pt_rect_attr_calibration_test.txt'),
            wflw_config=cfg.HEADCAMCAL,
        )

    else:
        raise ValueError('Wrong Dataset')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=cfg.WORKERS > 0,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=cfg.WORKERS > 0,
    )

    best_perf = 100.0
    # best_model = False
    last_epoch = -1

    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']

        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    for epoch in range(begin_epoch, begin_epoch + cfg.TRAIN.NUM_EPOCH):
        train_cal_refine(cfg, train_loader, model, loss_function, consistency_loss_function, optimizer, epoch,
              final_output_dir, writer_dict)
        perf_indicator = validate_cal_refine(
            cfg, valid_loader, model, loss_function, consistency_loss_function, final_output_dir, writer_dict
        )

        if perf_indicator <= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

        lr_scheduler.step()

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main_function()
