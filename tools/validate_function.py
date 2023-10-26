import numpy as np
import time
import torch
import logging
import os

from utils import AverageMeter
from backbone import Backbone

logger = logging.getLogger(__name__)


def validate(config, val_loader, model, loss_function, output_dir, writer_dict=None,
             edge_criterion=None):
    batch_time = AverageMeter()
    loss_average = AverageMeter()
    NME_stage1 = AverageMeter()
    NME_stage2 = AverageMeter()
    NME_stage3 = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            if config.DATASET.DATASET == 'HEADCAMCAL':
                input, _, meta, _ = data
            else:
                input, meta = data

            landmarks = model(input)
            ground_truth = meta['Points'].cuda().float()

            R_loss1 = loss_function(landmarks[0][:, config.TRANSFORMER.NUM_DECODER-1:config.TRANSFORMER.NUM_DECODER, :, :].detach(),
                                    ground_truth)
            R_loss2 = loss_function(landmarks[1][:, config.TRANSFORMER.NUM_DECODER-1:config.TRANSFORMER.NUM_DECODER, :, :].detach(),
                                    ground_truth)
            R_loss3 = loss_function(landmarks[2][:, config.TRANSFORMER.NUM_DECODER-1:config.TRANSFORMER.NUM_DECODER, :, :].detach(),
                                    ground_truth)

            NME_stage1.update(R_loss1.item(), input.size(0))
            NME_stage2.update(R_loss2.item(), input.size(0))
            NME_stage3.update(R_loss3.item(), input.size(0))


            loss = 0.2 * R_loss1 + 0.3 * R_loss2 + 0.5 * R_loss3

            loss_average.update(loss.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_FREQ == 0:
                msg = 'Epoch: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                      'NME_stage1 {NME_stage1.val:.5f} ({NME_stage1.avg:.5f})\t' \
                      'NME_stage2 {NME_stage2.val:.5f} ({NME_stage2.avg:.5f})\t' \
                      'NME_stage3 {NME_stage3.val:.5f} ({NME_stage3.avg:.5f})\t'.format(
                    i, len(val_loader), batch_time=batch_time,
                    speed=input.size(0) / batch_time.val,
                    loss=loss_average, NME_stage1=NME_stage1, NME_stage2=NME_stage2,
                    NME_stage3=NME_stage3)
                logger.info(msg)

                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('validate_loss', loss_average.val, global_steps)
                writer.add_scalar('validate_NME1', NME_stage1.val, global_steps)
                writer.add_scalar('validate_NME2', NME_stage2.val, global_steps)
                writer.add_scalar('validate_NME3', NME_stage3.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1
        msg = 'Stage1: ({NME_stage1.avg:.5f})\t' \
              'Stage2: ({NME_stage2.avg:.5f})\t' \
              'Stage3: ({NME_stage3.avg:.5f})\t'.format(
            NME_stage1=NME_stage1, NME_stage2=NME_stage2, NME_stage3=NME_stage3)
        logger.info(msg)
        return NME_stage3.avg


def validate_cal(config, val_loader, model, loss_function, consistency_loss_function,
                 output_dir, writer_dict=None, edge_criterion=None):
    batch_time = AverageMeter()
    loss_average = AverageMeter()
    NME_stage1 = AverageMeter()
    NME_stage2 = AverageMeter()
    NME_stage3 = AverageMeter()
    consistency_stage1 = AverageMeter()
    consistency_stage2 = AverageMeter()
    consistency_stage3 = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, input_cal, meta, meta_cal) in enumerate(val_loader):
            calibration_landmarks = meta_cal['Points'].cuda().float()
            landmarks = model(input, input_cal, calibration_landmarks)
            ground_truth = meta['Points'].cuda().float()

            R_loss1 = loss_function(landmarks[0][:, config.TRANSFORMER.NUM_DECODER-1:config.TRANSFORMER.NUM_DECODER, :, :].detach(),
                                    ground_truth)
            R_loss2 = loss_function(landmarks[1][:, config.TRANSFORMER.NUM_DECODER-1:config.TRANSFORMER.NUM_DECODER, :, :].detach(),
                                    ground_truth)
            R_loss3 = loss_function(landmarks[2][:, config.TRANSFORMER.NUM_DECODER-1:config.TRANSFORMER.NUM_DECODER, :, :].detach(),
                                    ground_truth)
            loss = 0.2 * R_loss1 + 0.3 * R_loss2 + 0.5 * R_loss3

            # feature_map = model.module.backbone(input.cuda())
            # calibration_feature_map = model.module.backbone(input_cal.cuda())
            #
            # consistency_loss_1 = consistency_loss_function(landmarks[0], ground_truth, feature_map,
            #                                                calibration_feature_map, calibration_landmarks, model.module,
            #                                                stage=1)
            # consistency_loss_2 = consistency_loss_function(landmarks[1], ground_truth, feature_map,
            #                                                calibration_feature_map, calibration_landmarks, model.module,
            #                                                stage=2)
            # consistency_loss_3 = consistency_loss_function(landmarks[2], ground_truth, feature_map,
            #                                                calibration_feature_map, calibration_landmarks, model.module,
            #                                                stage=3)
            #
            # loss += 100 * (0.2 * consistency_loss_1 + 0.3 * consistency_loss_2 + 0.5 * consistency_loss_3)

            NME_stage1.update(R_loss1.item(), input.size(0))
            NME_stage2.update(R_loss2.item(), input.size(0))
            NME_stage3.update(R_loss3.item(), input.size(0))

            # consistency_stage1.update(consistency_loss_1.item(), input.size(0))
            # consistency_stage2.update(consistency_loss_2.item(), input.size(0))
            # consistency_stage3.update(consistency_loss_3.item(), input.size(0))

            loss_average.update(loss.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_FREQ == 0:
                msg = 'Epoch: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                      'NME_stage1 {NME_stage1.val:.5f} ({NME_stage1.avg:.5f})\t' \
                      'NME_stage2 {NME_stage2.val:.5f} ({NME_stage2.avg:.5f})\t' \
                      'NME_stage3 {NME_stage3.val:.5f} ({NME_stage3.avg:.5f})\t'.format(
                    i, len(val_loader), batch_time=batch_time,
                    speed=input.size(0) / batch_time.val,
                    loss=loss_average, NME_stage1=NME_stage1, NME_stage2=NME_stage2,
                    NME_stage3=NME_stage3)
                logger.info(msg)

                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('validate_loss', loss_average.val, global_steps)
                writer.add_scalar('validate_NME1', NME_stage1.val, global_steps)
                writer.add_scalar('validate_NME2', NME_stage2.val, global_steps)
                writer.add_scalar('validate_NME3', NME_stage3.val, global_steps)
                # writer.add_scalar('validate_consistency1', consistency_stage1.val, global_steps)
                # writer.add_scalar('validate_consistency2', consistency_stage2.val, global_steps)
                # writer.add_scalar('validate_consistency3', consistency_stage3.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1
        msg = 'Stage1: ({NME_stage1.avg:.5f})\t' \
              'Stage2: ({NME_stage2.avg:.5f})\t' \
              'Stage3: ({NME_stage3.avg:.5f})\t'.format(
            NME_stage1=NME_stage1, NME_stage2=NME_stage2, NME_stage3=NME_stage3)
        logger.info(msg)
        return NME_stage3.avg
