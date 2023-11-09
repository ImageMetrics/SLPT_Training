import time
import logging
import os

import numpy as np
import torch

from torch.nn import functional as F
from utils import AverageMeter

logger = logging.getLogger(__name__)


def train(config, train_loader, model, loss_function, optimizer, epoch, output_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    NME_stage1 = AverageMeter()
    NME_stage2 = AverageMeter()
    NME_stage3 = AverageMeter()
    loss_average = AverageMeter()

    model.train()

    end = time.time()

    for i, data in enumerate(train_loader):
        if config.DATASET.DATASET == 'HEADCAMCAL':
            input, _, meta, _ = data
        else:
            input, meta = data

        data_time.update(time.time() - end)
        ground_truth = meta['Points'].cuda().float()
        landmarks = model(input)

        R_loss_1 = loss_function(landmarks[0], ground_truth)
        R_loss_2 = loss_function(landmarks[1], ground_truth)
        R_loss_3 = loss_function(landmarks[2], ground_truth)

        loss = 0.2 * R_loss_1 + 0.3 * R_loss_2 + 0.5 * R_loss_3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        NME_stage1.update(R_loss_1.item(), input.size(0))
        NME_stage2.update(R_loss_2.item(), input.size(0))
        NME_stage3.update(R_loss_3.item(), input.size(0))

        loss_average.update(loss.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'NME_stage1 {NME_stage1.val:.5f} ({NME_stage1.avg:.5f})\t' \
                  'NME_stage2 {NME_stage2.val:.5f} ({NME_stage2.avg:.5f})\t' \
                  'NME_stage3 {NME_stage3.val:.5f} ({NME_stage3.avg:.5f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=input.size(0) / batch_time.val,
                data_time=data_time, loss=loss_average, NME_stage1=NME_stage1,
                NME_stage2=NME_stage2, NME_stage3=NME_stage3)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', loss_average.val, global_steps)
            writer.add_scalar('NME1', NME_stage1.val, global_steps)
            writer.add_scalar('NME2', NME_stage2.val, global_steps)
            writer.add_scalar('NME3', NME_stage3.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            # prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)


def train_cal(config, train_loader, model, loss_function, consistency_loss_function,
              optimizer, epoch, output_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    NME_stage1 = AverageMeter()
    NME_stage2 = AverageMeter()
    NME_stage3 = AverageMeter()
    consistency_stage1 = AverageMeter()
    consistency_stage2 = AverageMeter()
    consistency_stage3 = AverageMeter()
    loss_average = AverageMeter()

    model.train()

    end = time.time()

    for i, (input, input_cal, meta, meta_cal) in enumerate(train_loader):
        data_time.update(time.time() - end)
        ground_truth = meta['Points'].cuda().float()
        calibration_points = meta_cal['Points'].cuda().float()
        landmarks = model(input, input_cal, calibration_points)

        R_loss_1 = loss_function(landmarks[0], ground_truth)
        R_loss_2 = loss_function(landmarks[1], ground_truth)
        R_loss_3 = loss_function(landmarks[2], ground_truth)

        loss = 0.2 * R_loss_1 + 0.3 * R_loss_2 + 0.5 * R_loss_3

        # feature_map = model.module.backbone(input.cuda())
        # calibration_feature_map = model.module.backbone(input_cal.cuda())
        #
        # consistency_loss_1 = consistency_loss_function(landmarks[0], ground_truth, feature_map,
        #                          calibration_feature_map, calibration_points, model.module, stage=1)
        # consistency_loss_2 = consistency_loss_function(landmarks[1], ground_truth, feature_map,
        #                          calibration_feature_map, calibration_points, model.module, stage=2)
        # consistency_loss_3 = consistency_loss_function(landmarks[2], ground_truth, feature_map,
        #                          calibration_feature_map, calibration_points, model.module, stage=3)
        #
        # loss += 100 * (0.2 * consistency_loss_1 + 0.3 * consistency_loss_2 + 0.5 * consistency_loss_3)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        NME_stage1.update(R_loss_1.item(), input.size(0))
        NME_stage2.update(R_loss_2.item(), input.size(0))
        NME_stage3.update(R_loss_3.item(), input.size(0))

        # consistency_stage1.update(consistency_loss_1.item(), input.size(0))
        # consistency_stage2.update(consistency_loss_2.item(), input.size(0))
        # consistency_stage3.update(consistency_loss_3.item(), input.size(0))

        loss_average.update(loss.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'NME_stage1 {NME_stage1.val:.5f} ({NME_stage1.avg:.5f})\t' \
                  'NME_stage2 {NME_stage2.val:.5f} ({NME_stage2.avg:.5f})\t' \
                  'NME_stage3 {NME_stage3.val:.5f} ({NME_stage3.avg:.5f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=input.size(0) / batch_time.val,
                data_time=data_time, loss=loss_average, NME_stage1=NME_stage1,
                NME_stage2=NME_stage2, NME_stage3=NME_stage3)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', loss_average.val, global_steps)
            writer.add_scalar('NME1', NME_stage1.val, global_steps)
            writer.add_scalar('NME2', NME_stage2.val, global_steps)
            writer.add_scalar('NME3', NME_stage3.val, global_steps)
            # writer.add_scalar('consistency1', consistency_stage1.val, global_steps)
            # writer.add_scalar('consistency2', consistency_stage2.val, global_steps)
            # writer.add_scalar('consistency3', consistency_stage3.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            # prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)


def train_cal_refine(config, train_loader, model, loss_function, consistency_loss_function,
              optimizer, epoch, output_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    NME_stage1 = AverageMeter()
    NME_stage2 = AverageMeter()
    NME_stage3 = AverageMeter()
    consistency_stage1 = AverageMeter()
    consistency_stage2 = AverageMeter()
    consistency_stage3 = AverageMeter()
    loss_average = AverageMeter()

    model.train()

    end = time.time()

    for i, (input, input_cal, meta, meta_cal) in enumerate(train_loader):
        data_time.update(time.time() - end)
        ground_truth = meta['Points'].cuda().float()
        calibration_points = meta_cal['Points'].cuda().float()
        start_points = meta['StartPoints'].cuda().float()
        landmarks = model(input, start_points, input_cal, calibration_points)

        R_loss_1 = loss_function(landmarks[0], ground_truth)
        R_loss_2 = loss_function(landmarks[1], ground_truth)
        R_loss_3 = loss_function(landmarks[2], ground_truth)

        loss = 0.2 * R_loss_1 + 0.3 * R_loss_2 + 0.5 * R_loss_3

        feature_map = model.module.backbone(input.cuda())
        calibration_feature_map = model.module.backbone(input_cal.cuda())

        consistency_loss_1 = consistency_loss_function(landmarks[0], ground_truth, feature_map,
                                 calibration_feature_map, calibration_points, model.module, stage=1)
        consistency_loss_2 = consistency_loss_function(landmarks[1], ground_truth, feature_map,
                                 calibration_feature_map, calibration_points, model.module, stage=2)
        consistency_loss_3 = consistency_loss_function(landmarks[2], ground_truth, feature_map,
                                 calibration_feature_map, calibration_points, model.module, stage=3)

        loss += 1e3 * (0.2 * consistency_loss_1 + 0.3 * consistency_loss_2 + 0.5 * consistency_loss_3)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        NME_stage1.update(R_loss_1.item(), input.size(0))
        NME_stage2.update(R_loss_2.item(), input.size(0))
        NME_stage3.update(R_loss_3.item(), input.size(0))

        consistency_stage1.update(consistency_loss_1.item(), input.size(0))
        consistency_stage2.update(consistency_loss_2.item(), input.size(0))
        consistency_stage3.update(consistency_loss_3.item(), input.size(0))

        loss_average.update(loss.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'NME_stage1 {NME_stage1.val:.5f} ({NME_stage1.avg:.5f})\t' \
                  'NME_stage2 {NME_stage2.val:.5f} ({NME_stage2.avg:.5f})\t' \
                  'NME_stage3 {NME_stage3.val:.5f} ({NME_stage3.avg:.5f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=input.size(0) / batch_time.val,
                data_time=data_time, loss=loss_average, NME_stage1=NME_stage1,
                NME_stage2=NME_stage2, NME_stage3=NME_stage3)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', loss_average.val, global_steps)
            writer.add_scalar('NME1', NME_stage1.val, global_steps)
            writer.add_scalar('NME2', NME_stage2.val, global_steps)
            writer.add_scalar('NME3', NME_stage3.val, global_steps)
            writer.add_scalar('consistency1', consistency_stage1.val, global_steps)
            writer.add_scalar('consistency2', consistency_stage2.val, global_steps)
            writer.add_scalar('consistency3', consistency_stage3.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            # prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
