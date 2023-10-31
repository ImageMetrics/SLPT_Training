import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Transformer import Transformer
from Transformer import interpolation_layer
from Transformer import get_roi
from backbone import get_face_alignment_net



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


from Transformer.Transformer import TransformerCal


class Sparse_alignment_network_cal(Sparse_alignment_network):
    def __init__(self, num_point, d_model, trainable,
                 return_interm_layers, dilation, nhead,  feedforward_dim,
                 initial_path, cfg):
        super().__init__(num_point, d_model, trainable,
                 return_interm_layers, dilation, nhead,  feedforward_dim,
                 initial_path, cfg)

        # Transformer
        self.Transformer = TransformerCal(num_point, d_model, nhead, cfg.TRANSFORMER.NUM_DECODER,
                                       feedforward_dim, dropout=0.1)

        self.feature_extractor_cal = nn.Conv2d(d_model, d_model, kernel_size=self.Sample_num, bias=False)

        self._reset_parameters()

        # backbone
        self.backbone = get_face_alignment_net(cfg)

    def forward(self, image, cal_image, cal_landmarks):

        bs = image.size(0)

        output_list = []

        feature_map = self.backbone(image)
        calibration_feature_map = self.backbone(cal_image)

        # cal features
        ROI_feature_cal_1, _, _ = self.get_image_features(calibration_feature_map, cal_landmarks, stage=1)
        ROI_feature_cal_2, _, _ = self.get_image_features(calibration_feature_map, cal_landmarks, stage=2)
        ROI_feature_cal_3, _, _ = self.get_image_features(calibration_feature_map, cal_landmarks, stage=3)

        initial_landmarks = self.initial_points.repeat(bs, 1, 1).to(image.device)

        # stage_1
        ROI_feature_1, bbox_size_1, start_anchor_1 = \
            self.get_image_features(feature_map, initial_landmarks, stage=1)

        transformer_feature_1 = self.feature_extractor(ROI_feature_1).view(bs, self.num_point, self.d_model)
        transformer_feature_cal_1 = self.feature_extractor_cal(ROI_feature_cal_1).view(bs, self.num_point, self.d_model)

        offset_1 = self.Transformer(transformer_feature_1, transformer_feature_cal_1)
        offset_1 = self.out_layer(offset_1)

        landmarks_1 = start_anchor_1.unsqueeze(1) + bbox_size_1.unsqueeze(1) * offset_1
        output_list.append(landmarks_1)

        # stage_2
        ROI_feature_2, bbox_size_2, start_anchor_2 = \
            self.get_image_features(feature_map, landmarks_1[:, -1, :, :], stage=2)

        transformer_feature_2 = self.feature_extractor(ROI_feature_2).view(bs, self.num_point, self.d_model)
        transformer_feature_cal_2 = self.feature_extractor_cal(ROI_feature_cal_2).view(bs, self.num_point, self.d_model)

        offset_2 = self.Transformer(transformer_feature_2, transformer_feature_cal_2)
        offset_2 = self.out_layer(offset_2)

        landmarks_2 = start_anchor_2.unsqueeze(1) + bbox_size_2.unsqueeze(1) * offset_2
        output_list.append(landmarks_2)

        # stage_3
        ROI_feature_3, bbox_size_3, start_anchor_3 = \
            self.get_image_features(feature_map, landmarks_2[:, -1, :, :], stage=3)

        transformer_feature_3 = self.feature_extractor(ROI_feature_3).view(bs, self.num_point, self.d_model)
        transformer_feature_cal_3 = self.feature_extractor_cal(ROI_feature_cal_3).view(bs, self.num_point, self.d_model)

        offset_3 = self.Transformer(transformer_feature_3, transformer_feature_cal_3)
        offset_3 = self.out_layer(offset_3)

        landmarks_3 = start_anchor_3.unsqueeze(1) + bbox_size_3.unsqueeze(1) * offset_3
        output_list.append(landmarks_3)

        return output_list

    def get_image_features(self, feature_map, landmarks, stage=1):
        bs = feature_map.size(0)

        # features
        if stage == 1:
            ROI_anchor_1, bbox_size_1, start_anchor_1 = self.ROI_1(landmarks.detach())
            ROI_anchor_1 = ROI_anchor_1.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
            ROI_feature_1 = self.interpolation(feature_map, ROI_anchor_1.detach()).view(bs, self.num_point, self.Sample_num,
                                                                                self.Sample_num, self.d_model)
            ROI_feature_1 = ROI_feature_1.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                         self.d_model).permute(0, 3, 2, 1)
            return ROI_feature_1, bbox_size_1, start_anchor_1

        elif stage == 2:
            ROI_anchor_2, bbox_size_2, start_anchor_2 = self.ROI_2(landmarks.detach())
            ROI_anchor_2 = ROI_anchor_2.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
            ROI_feature_2 = self.interpolation(feature_map, ROI_anchor_2.detach()).view(bs, self.num_point, self.Sample_num,
                                                                                self.Sample_num, self.d_model)
            ROI_feature_2 = ROI_feature_2.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                         self.d_model).permute(0, 3, 2, 1)
            return ROI_feature_2, bbox_size_2, start_anchor_2
        else:
            ROI_anchor_3, bbox_size_3, start_anchor_3 = self.ROI_3(landmarks.detach())
            ROI_anchor_3 = ROI_anchor_3.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
            ROI_feature_3 = self.interpolation(feature_map, ROI_anchor_3.detach()).view(bs, self.num_point, self.Sample_num,
                                                                                self.Sample_num, self.d_model)
            ROI_feature_3 = ROI_feature_3.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                         self.d_model).permute(0, 3, 2, 1)

            return ROI_feature_3, bbox_size_3, start_anchor_3


class Sparse_alignment_network_refine(Sparse_alignment_network_cal):
    def __init__(self, *args):
        super().__init__(*args)

        # ROI_creator - make these a lot tighter
        self.ROI_1 = get_roi(self.Sample_num, 4.0, 64)
        self.ROI_2 = get_roi(self.Sample_num, 2.0, 64)
        self.ROI_3 = get_roi(self.Sample_num, 1.0, 64)

    def forward(self, image, start_landmarks, cal_image, cal_landmarks):

        bs = image.size(0)

        output_list = []

        feature_map = self.backbone(image)
        calibration_feature_map = self.backbone(cal_image)

        # cal features
        ROI_feature_cal_1, _, _ = self.get_image_features(calibration_feature_map, cal_landmarks, stage=1)
        ROI_feature_cal_2, _, _ = self.get_image_features(calibration_feature_map, cal_landmarks, stage=2)
        ROI_feature_cal_3, _, _ = self.get_image_features(calibration_feature_map, cal_landmarks, stage=3)

        initial_landmarks = start_landmarks.to(image.device)

        # stage_1
        ROI_feature_1, bbox_size_1, start_anchor_1 = \
            self.get_image_features(feature_map, initial_landmarks, stage=1)

        transformer_feature_1 = self.feature_extractor(ROI_feature_1).view(bs, self.num_point, self.d_model)
        transformer_feature_cal_1 = self.feature_extractor_cal(ROI_feature_cal_1).view(bs, self.num_point, self.d_model)

        offset_1 = self.Transformer(transformer_feature_1, transformer_feature_cal_1)
        offset_1 = self.out_layer(offset_1)

        landmarks_1 = start_anchor_1.unsqueeze(1) + bbox_size_1.unsqueeze(1) * offset_1
        output_list.append(landmarks_1)

        # stage_2
        ROI_feature_2, bbox_size_2, start_anchor_2 = \
            self.get_image_features(feature_map, landmarks_1[:, -1, :, :], stage=2)

        transformer_feature_2 = self.feature_extractor(ROI_feature_2).view(bs, self.num_point, self.d_model)
        transformer_feature_cal_2 = self.feature_extractor_cal(ROI_feature_cal_2).view(bs, self.num_point, self.d_model)

        offset_2 = self.Transformer(transformer_feature_2, transformer_feature_cal_2)
        offset_2 = self.out_layer(offset_2)

        landmarks_2 = start_anchor_2.unsqueeze(1) + bbox_size_2.unsqueeze(1) * offset_2
        output_list.append(landmarks_2)

        # stage_3
        ROI_feature_3, bbox_size_3, start_anchor_3 = \
            self.get_image_features(feature_map, landmarks_2[:, -1, :, :], stage=3)

        transformer_feature_3 = self.feature_extractor(ROI_feature_3).view(bs, self.num_point, self.d_model)
        transformer_feature_cal_3 = self.feature_extractor_cal(ROI_feature_cal_3).view(bs, self.num_point, self.d_model)

        offset_3 = self.Transformer(transformer_feature_3, transformer_feature_cal_3)
        offset_3 = self.out_layer(offset_3)

        landmarks_3 = start_anchor_3.unsqueeze(1) + bbox_size_3.unsqueeze(1) * offset_3
        output_list.append(landmarks_3)

        return output_list

    def get_image_features(self, feature_map, landmarks, stage=1):
        bs = feature_map.size(0)

        # features
        if stage == 1:
            ROI_anchor_1, bbox_size_1, start_anchor_1 = self.ROI_1(landmarks.detach())
            ROI_anchor_1 = ROI_anchor_1.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
            ROI_feature_1 = self.interpolation(feature_map, ROI_anchor_1.detach()).view(bs, self.num_point, self.Sample_num,
                                                                                self.Sample_num, self.d_model)
            ROI_feature_1 = ROI_feature_1.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                         self.d_model).permute(0, 3, 2, 1)
            return ROI_feature_1, bbox_size_1, start_anchor_1

        elif stage == 2:
            ROI_anchor_2, bbox_size_2, start_anchor_2 = self.ROI_2(landmarks.detach())
            ROI_anchor_2 = ROI_anchor_2.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
            ROI_feature_2 = self.interpolation(feature_map, ROI_anchor_2.detach()).view(bs, self.num_point, self.Sample_num,
                                                                                self.Sample_num, self.d_model)
            ROI_feature_2 = ROI_feature_2.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                         self.d_model).permute(0, 3, 2, 1)
            return ROI_feature_2, bbox_size_2, start_anchor_2
        else:
            ROI_anchor_3, bbox_size_3, start_anchor_3 = self.ROI_3(landmarks.detach())
            ROI_anchor_3 = ROI_anchor_3.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
            ROI_feature_3 = self.interpolation(feature_map, ROI_anchor_3.detach()).view(bs, self.num_point, self.Sample_num,
                                                                                self.Sample_num, self.d_model)
            ROI_feature_3 = ROI_feature_3.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                         self.d_model).permute(0, 3, 2, 1)

            return ROI_feature_3, bbox_size_3, start_anchor_3
