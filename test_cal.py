import argparse

from Config.default_cal import _C as cfg
from Config.default_cal import update_config

from utils import create_logger
from model import Sparse_alignment_network_cal
from Dataloader.WFLW_loader import WFLWCal_Dataset
from utils import AverageMeter


import torch
import cv2
import numpy as np
import pprint
import os

import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Train Sparse Facial Network')

    # philly
    parser.add_argument('--dataDir', help='data directory', type=str, default='./')
    parser.add_argument('--checkpoint', help='checkpoint file', type=str, default='./WFLW_6_layer.pth')

    args = parser.parse_args()

    return args

def calcuate_loss(name, pred, gt, trans):

    pred = (pred - trans[:, 2]) @ np.linalg.inv(trans[:, 0:2].T)

    if name == 'HEADCAMCAL':
        norm = np.linalg.norm(gt[28, :] - gt[26, :])
    else:
        raise ValueError('Wrong Dataset')

    error_real = np.mean(np.linalg.norm((pred - gt), axis=1) / norm)

    return error_real

def main_function():
    args = parse_args()
    update_config(cfg, args)

    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    if cfg.DATASET.DATASET == 'HEADCAMCAL':
        model = Sparse_alignment_network_cal(cfg.HEADCAMCAL.NUM_POINT, cfg.MODEL.OUT_DIM,
                                         cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
                                         cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
                                         cfg.TRANSFORMER.FEED_DIM, cfg.HEADCAMCAL.INITIAL_PATH, cfg)
    else:
        raise ValueError('Wrong Dataset')

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if cfg.DATASET.DATASET == 'HEADCAMCAL':
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

    # 验证数据迭代器
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    checkpoint = torch.load(args.checkpoint)

    model.module.load_state_dict(checkpoint)

    error_list = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, input_cal, meta, meta_cal) in enumerate(valid_loader):
            Annotated_Points = meta['Annotated_Points'].numpy()[0]
            Trans = meta['trans'].numpy()[0]
            calibration_points = meta_cal['Points'].cuda().float()

            outputs_initial = model(input.cuda(), input_cal.cuda(), calibration_points)

            output = outputs_initial[2][0, -1, :, :].cpu().numpy()

            error = calcuate_loss(cfg.DATASET.DATASET, output * cfg.MODEL.IMG_SIZE, Annotated_Points, Trans)
            error_list.update(error, input.size(0))

            msg = 'Epoch: [{0}/{1}]\t' \
                  'NME: {error:.3f}%\t'.format(
                i, len(valid_loader), error=error_list.avg * 100.0)

            print(msg)

        print("finished")
        print("Mean Error: {:.3f}".format(error_list.avg * 100.0))


if __name__ == '__main__':
    main_function()

