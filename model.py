import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Transformer import Transformer
from Transformer import interpolation_layer
from Transformer import _get_clones
from Transformer import get_roi
from backbone import get_face_alignment_net
from backbone import Get_Hourglass
from backbone import build_position_encoding
from backbone import Backbone, feature_fusion
from backbone import conv_1x1_bn
from utils import decode_preds, get_initial_pred


class Sparse_alignment_network(nn.Module):
    def __init__(self, num_point, d_model, trainable,
                 return_interm_layers, dilation, nhead,  feedforward_dim,
                 initial_path, cfg):
        super(Sparse_alignment_network, self).__init__()
        self.num_point = num_point
        self.d_model = d_model
        self.trainable = trainable
        self.return_interm_layers = return_interm_layers
        self.dilation = dilation
        self.nhead = nhead
        self.feedforward_dim = feedforward_dim
        self.initial_path = initial_path
        self.heatmap_size = cfg.MODEL.HEATMAP
        self.Sample_num = cfg.MODEL.SAMPLE_NUM

        self.initial_points = torch.from_numpy(np.load(initial_path)['init_face'] / 256.0).view(1, num_point, 2).float()
        self.initial_points.requires_grad = False

        # ROI_creator
        self.ROI_1 = get_roi(self.Sample_num, 8.0, 64)
        self.ROI_2 = get_roi(self.Sample_num, 4.0, 64)
        self.ROI_3 = get_roi(self.Sample_num, 2.0, 64)

        self.interpolation = interpolation_layer()

        # feature_extractor
        self.feature_extractor = nn.Conv2d(d_model, d_model, kernel_size=self.Sample_num, bias=False)

        self.feature_norm = nn.LayerNorm(d_model)

        # Transformer
        self.Transformer = Transformer(num_point, d_model, nhead, cfg.TRANSFORMER.NUM_DECODER,
                                       feedforward_dim, dropout=0.1)

        self.out_layer = nn.Linear(d_model, 2)

        self._reset_parameters()

        # backbone
        self.backbone = get_face_alignment_net(cfg)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, image):
        bs = image.size(0)

        output_list = []

        feature_map = self.backbone(image)

        initial_landmarks = self.initial_points.repeat(bs, 1, 1).to(image.device)

        # stage_1
        ROI_anchor_1, bbox_size_1, start_anchor_1 = self.ROI_1(initial_landmarks.detach())
        ROI_anchor_1 = ROI_anchor_1.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
        ROI_feature_1 = self.interpolation(feature_map, ROI_anchor_1.detach()).view(bs, self.num_point, self.Sample_num,
                                                                            self.Sample_num, self.d_model)
        ROI_feature_1 = ROI_feature_1.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                     self.d_model).permute(0, 3, 2, 1)

        transformer_feature_1 = self.feature_extractor(ROI_feature_1).view(bs, self.num_point, self.d_model)

        offset_1 = self.Transformer(transformer_feature_1)
        offset_1 = self.out_layer(offset_1)

        landmarks_1 = start_anchor_1.unsqueeze(1) + bbox_size_1.unsqueeze(1) * offset_1
        output_list.append(landmarks_1)

        # stage_2
        ROI_anchor_2, bbox_size_2, start_anchor_2 = self.ROI_2(landmarks_1[:, -1, :, :].detach())
        ROI_anchor_2 = ROI_anchor_2.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
        ROI_feature_2 = self.interpolation(feature_map, ROI_anchor_2.detach()).view(bs, self.num_point, self.Sample_num,
                                                                                 self.Sample_num, self.d_model)
        ROI_feature_2 = ROI_feature_2.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                           self.d_model).permute(0, 3, 2, 1)

        transformer_feature_2 = self.feature_extractor(ROI_feature_2).view(bs, self.num_point, self.d_model)

        offset_2 = self.Transformer(transformer_feature_2)
        offset_2 = self.out_layer(offset_2)

        landmarks_2 = start_anchor_2.unsqueeze(1) + bbox_size_2.unsqueeze(1) * offset_2
        output_list.append(landmarks_2)

        # stage_3
        ROI_anchor_3, bbox_size_3, start_anchor_3 = self.ROI_3(landmarks_2[:, -1, :, :].detach())
        ROI_anchor_3 = ROI_anchor_3.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
        ROI_feature_3= self.interpolation(feature_map, ROI_anchor_3.detach()).view(bs, self.num_point, self.Sample_num,
                                                                                   self.Sample_num, self.d_model)
        ROI_feature_3 = ROI_feature_3.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                           self.d_model).permute(0, 3, 2, 1)

        transformer_feature_3 = self.feature_extractor(ROI_feature_3).view(bs, self.num_point, self.d_model)

        offset_3 = self.Transformer(transformer_feature_3)
        offset_3 = self.out_layer(offset_3)

        landmarks_3 = start_anchor_3.unsqueeze(1) + bbox_size_3.unsqueeze(1) * offset_3
        output_list.append(landmarks_3)

        return output_list


class Sparse_alignment_network_cal(Sparse_alignment_network):
    def __init__(self, *args):
        super().__init__(*args)

        self.cal_method = 'concat' # can be 'concat', 'subtract'

        # modifications for concatenated calibration
        if self.cal_method == 'concat':
            self.feature_extractor = nn.Conv2d(self.d_model * 2, self.d_model, kernel_size=self.Sample_num, bias=False)

    def forward(self, image, cal_image, cal_landmarks):

        bs = image.size(0)

        output_list = []

        feature_map = self.backbone(image)
        calibration_feature_map = self.backbone(cal_image)

        # cal features
        ROI_anchor_cal_1, bbox_size_cal_1, start_anchor_cal_1 = self.ROI_1(cal_landmarks.detach())
        ROI_anchor_cal_1 = ROI_anchor_cal_1.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
        ROI_feature_cal_1 = self.interpolation(calibration_feature_map, ROI_anchor_cal_1.detach()).view(bs, self.num_point, self.Sample_num,
                                                                            self.Sample_num, self.d_model)
        ROI_feature_cal_1 = ROI_feature_cal_1.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                     self.d_model).permute(0, 3, 2, 1)

        ROI_anchor_cal_2, bbox_size_cal_2, start_anchor_cal_2 = self.ROI_2(cal_landmarks.detach())
        ROI_anchor_cal_2 = ROI_anchor_cal_2.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
        ROI_feature_cal_2 = self.interpolation(calibration_feature_map, ROI_anchor_cal_2.detach()).view(bs, self.num_point, self.Sample_num,
                                                                            self.Sample_num, self.d_model)
        ROI_feature_cal_2 = ROI_feature_cal_2.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                     self.d_model).permute(0, 3, 2, 1)

        ROI_anchor_cal_3, bbox_size_cal_3, start_anchor_cal_3 = self.ROI_3(cal_landmarks.detach())
        ROI_anchor_cal_3 = ROI_anchor_cal_3.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
        ROI_feature_cal_3 = self.interpolation(calibration_feature_map, ROI_anchor_cal_3.detach()).view(bs, self.num_point, self.Sample_num,
                                                                            self.Sample_num, self.d_model)
        ROI_feature_cal_3 = ROI_feature_cal_3.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                     self.d_model).permute(0, 3, 2, 1)

        initial_landmarks = self.initial_points.repeat(bs, 1, 1).to(image.device)

        # stage_1
        ROI_anchor_1, bbox_size_1, start_anchor_1 = self.ROI_1(initial_landmarks.detach())
        ROI_anchor_1 = ROI_anchor_1.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
        ROI_feature_1 = self.interpolation(feature_map, ROI_anchor_1.detach()).view(bs, self.num_point, self.Sample_num,
                                                                            self.Sample_num, self.d_model)
        ROI_feature_1 = ROI_feature_1.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                     self.d_model).permute(0, 3, 2, 1)

        if self.cal_method == 'subtract':
            # subtracted calibration
            transformer_feature_1 = self.feature_extractor(ROI_feature_1-ROI_feature_cal_1).view(bs, self.num_point, self.d_model)
        elif self.cal_method == 'concat':
            # concatenated calibration
            transformer_feature_1 = self.feature_extractor(
                torch.cat((ROI_feature_1,ROI_feature_cal_1), dim=1)
            ).view(bs, self.num_point, self.d_model)

        offset_1 = self.Transformer(transformer_feature_1)
        offset_1 = self.out_layer(offset_1)

        landmarks_1 = start_anchor_1.unsqueeze(1) + bbox_size_1.unsqueeze(1) * offset_1
        output_list.append(landmarks_1)

        # stage_2
        ROI_anchor_2, bbox_size_2, start_anchor_2 = self.ROI_2(landmarks_1[:, -1, :, :].detach())
        ROI_anchor_2 = ROI_anchor_2.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
        ROI_feature_2 = self.interpolation(feature_map, ROI_anchor_2.detach()).view(bs, self.num_point, self.Sample_num,
                                                                                 self.Sample_num, self.d_model)
        ROI_feature_2 = ROI_feature_2.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                           self.d_model).permute(0, 3, 2, 1)

        if self.cal_method == 'subtract':
            # subtracted calibration
            transformer_feature_2 = self.feature_extractor(ROI_feature_2 - ROI_feature_cal_2).view(bs, self.num_point,
                                                                                                   self.d_model)
        elif self.cal_method == 'concat':
            # concatenated calibration
            transformer_feature_2 = self.feature_extractor(
                torch.cat((ROI_feature_2,ROI_feature_cal_2), dim=1)
            ).view(bs, self.num_point, self.d_model)

        offset_2 = self.Transformer(transformer_feature_2)
        offset_2 = self.out_layer(offset_2)

        landmarks_2 = start_anchor_2.unsqueeze(1) + bbox_size_2.unsqueeze(1) * offset_2
        output_list.append(landmarks_2)

        # stage_3
        ROI_anchor_3, bbox_size_3, start_anchor_3 = self.ROI_3(landmarks_2[:, -1, :, :].detach())
        ROI_anchor_3 = ROI_anchor_3.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
        ROI_feature_3= self.interpolation(feature_map, ROI_anchor_3.detach()).view(bs, self.num_point, self.Sample_num,
                                                                                   self.Sample_num, self.d_model)
        ROI_feature_3 = ROI_feature_3.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                           self.d_model).permute(0, 3, 2, 1)

        if self.cal_method == 'subtract':
            # subtracted calibration
            transformer_feature_3 = self.feature_extractor(ROI_feature_3 - ROI_feature_cal_3).view(bs, self.num_point,
                                                                                                   self.d_model)
        elif self.cal_method == 'concat':
            # concatenated calibration
            transformer_feature_3 = self.feature_extractor(
                torch.cat((ROI_feature_3,ROI_feature_cal_3), dim=1)
            ).view(bs, self.num_point, self.d_model)

        offset_3 = self.Transformer(transformer_feature_3)
        offset_3 = self.out_layer(offset_3)

        landmarks_3 = start_anchor_3.unsqueeze(1) + bbox_size_3.unsqueeze(1) * offset_3
        output_list.append(landmarks_3)

        return output_list
