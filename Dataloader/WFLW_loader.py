import copy
import random
import logging
import json

import cv2
import numpy
import torch
import numpy as np
import os
import utils

from torch.utils.data import Dataset

from utils import generate_target


logger = logging.getLogger(__name__)


class WFLW_Dataset(Dataset):
    def __init__(self, cfg, root, is_train,
                 transform=None,
                 annotation_file=None,
                 wflw_config=None):
        self.Image_size = cfg.MODEL.IMG_SIZE
        self.is_train = is_train
        self.root = root

        if wflw_config is None:
            wflw_config = cfg.WFLW

        self.number_landmarks = wflw_config.NUM_POINT
        self.flip_index = np.genfromtxt(os.path.join(self.root, "Mirror.txt"),
                                        dtype=int, delimiter=',')

        self.Fraction = wflw_config.FRACTION
        self.Translation_Factor = wflw_config.TRANSLATION
        self.Rotation_Factor = wflw_config.ROTATION
        self.Scale_Factor = wflw_config.SCALE
        self.Occlusion_Mean = wflw_config.OCCLUSION_MEAN
        self.Occlusion_Std = wflw_config.OCCLUSION_STD
        self.Flip = wflw_config.FLIP
        self.Occlusion = wflw_config.OCCLUSION
        self.Transfer = wflw_config.CHANNEL_TRANSFER

        self.Heatmap_size = cfg.MODEL.HEATMAP

        self.Data_Format = wflw_config.DATA_FORMAT
        self.image_dir = wflw_config.IMAGE_DIR

        self.Transform = transform

        if annotation_file is None:
            if is_train:
                self.annotation_file = os.path.join(root, 'WFLW_annotations', 'list_98pt_rect_attr_train_test',
                                                    'list_98pt_rect_attr_train.txt')
            else:
                self.annotation_file = os.path.join(root, 'WFLW_annotations', 'list_98pt_rect_attr_train_test',
                                                    'list_98pt_rect_attr_test.txt')
        else:
            self.annotation_file = annotation_file

        self.database = self.get_file_information()

    def get_file_information(self):
        Data_base = []

        with open(self.annotation_file) as f:
            info_list = f.read().splitlines()
            f.close()

        for temp_info in info_list:
            temp_point = []
            temp_info = temp_info.split(' ')
            for i in range(2 * self.number_landmarks):
                temp_point.append(float(temp_info[i]))
            point_coord = np.array(temp_point, dtype=np.float).reshape(self.number_landmarks, 2)
            max_index = np.max(point_coord, axis=0)
            min_index = np.min(point_coord, axis=0)
            temp_box = np.array([min_index[0], min_index[1], max_index[0] - min_index[0],
                                 max_index[1] - min_index[1]])
            temp_name = os.path.join(self.root, self.image_dir, temp_info[-1])
            temp_name = temp_name.replace('\\', os.sep)
            Data_base.append({'Img': temp_name,
                              'bbox': temp_box,
                              'point': point_coord})

        return Data_base

    def Image_Flip(self, Img, GT):
        Mirror_GT = []
        width = Img.shape[1]
        for i in self.flip_index:
            Mirror_GT.append([width - 1 - GT[i][0], GT[i][1]])
        Img = cv2.flip(Img, 1)
        return Img, numpy.array(Mirror_GT)

    def Channel_Transfer(self, Img, Flag):
        if Flag == 1:
            Img = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
            Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        return Img

    def Create_Occlusion(self, Img):
        Occlusion_width = int(self.Image_size * np.random.normal(self.Occlusion_Mean, self.Occlusion_Std))
        Occlusion_high = int(self.Image_size * np.random.normal(self.Occlusion_Mean, self.Occlusion_Std))
        Occlusion_x = np.random.randint(0, self.Image_size - Occlusion_width)
        Occlusion_y = np.random.randint(0, self.Image_size - Occlusion_high)

        Img[Occlusion_y:Occlusion_y + Occlusion_high, Occlusion_x:Occlusion_x + Occlusion_width, 0] = \
            np.random.randint(0, 256)
        Img[Occlusion_y:Occlusion_y + Occlusion_high, Occlusion_x:Occlusion_x + Occlusion_width, 1] = \
            np.random.randint(0, 256)
        Img[Occlusion_y:Occlusion_y + Occlusion_high, Occlusion_x:Occlusion_x + Occlusion_width, 2] = \
            np.random.randint(0, 256)

        return Img

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        db_slic = copy.deepcopy(self.database[idx])
        Img_path = db_slic['Img']
        BBox = db_slic['bbox']
        Points = db_slic['point']
        Annotated_Points = Points.copy()

        Img = cv2.imread(Img_path)

        Img_shape = Img.shape
        Img = cv2.cvtColor(Img, cv2.COLOR_RGB2BGR)
        if len(Img_shape) < 3:
            Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        else:
            if Img_shape[2] == 4:
                Img = cv2.cvtColor(Img, cv2.COLOR_RGBA2RGB)
            elif Img_shape[2] == 1:
                Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)

        if self.is_train == True:
            Rotation_Factor = self.Rotation_Factor * np.pi / 180.0
            Scale_Factor = self.Scale_Factor
            Translation_X_Factor = self.Translation_Factor
            Translation_Y_Factor = self.Translation_Factor

            angle = np.clip(np.random.normal(0, Rotation_Factor), -2 * Rotation_Factor, 2 * Rotation_Factor)
            Scale = np.clip(np.random.normal(self.Fraction, Scale_Factor), self.Fraction - Scale_Factor, self.Fraction + Scale_Factor)

            Translation_X = np.clip(np.random.normal(0, Translation_X_Factor), -Translation_X_Factor, Translation_X_Factor)
            Translation_Y = np.clip(np.random.normal(0, Translation_Y_Factor), -Translation_Y_Factor, Translation_Y_Factor)

            trans = utils.get_transforms(BBox, Scale, angle, self.Image_size, shift_factor=[Translation_X, Translation_Y])

            input = cv2.warpAffine(Img, trans, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)

            for i in range(self.number_landmarks):
                Points[i,0:2] = utils.affine_transform(Points[i,0:2], trans)

            if self.Flip is True:
                Flip_Flag = np.random.randint(0, 2)
                if Flip_Flag == 1:
                    input, Points = self.Image_Flip(input, Points)

            if self.Transfer is True:
                Transfer_Flag = np.random.randint(0, 5)
                input = self.Channel_Transfer(input, Transfer_Flag)

            if self.Occlusion is True:
                Occlusion_Flag = np.random.randint(0, 2)
                if Occlusion_Flag == 1:
                    input = self.Create_Occlusion(input)

            if self.Transform is not None:
                input = self.Transform(input)



            meta = {'Img_path': Img_path,
                    'Points': Points / (self.Image_size),
                    'BBox': BBox,
                    'trans': trans,
                    'Scale': Scale,
                    'angle': angle,
                    'Translation': [Translation_X, Translation_Y]}

            return input, meta

        else:
            trans = utils.get_transforms(BBox, self.Fraction, 0.0, self.Image_size, shift_factor=[0.0, 0.0])

            input = cv2.warpAffine(Img, trans, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)

            for i in range(self.number_landmarks):
                Points[i, 0:2] = utils.affine_transform(Points[i, 0:2], trans)

            meta = {
                "Annotated_Points": Annotated_Points,
                'Img_path': Img_path,
                'Points': Points / (self.Image_size),
                'BBox': BBox,
                'trans': trans,
                'Scale': self.Fraction,
                'angle': 0.0,
                'Translation': [0.0, 0.0],
            }

            # target = np.zeros((self.number_landmarks, self.Heatmap_size, self.Heatmap_size))
            # tpts = Points / (self.Image_size - 1) * (self.Heatmap_size - 1)
            # for i in range(self.number_landmarks):
            #     if tpts[i, 1] > 0:
            #         target[i] = generate_target(target[i], tpts[i], self.sigma)

            if self.Transform is not None:
                input = self.Transform(input)

            return input, meta


class WFLWCal_Dataset(WFLW_Dataset):
    def __init__(self,
                 *args,
                 calibration_annotation_file=None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.calibration_annotation_file = calibration_annotation_file

        self.calibration_database = self.get_calibration_file_information()

    def get_file_information(self):
        Data_base = []

        with open(self.annotation_file) as f:
            info_list = f.read().splitlines()
            f.close()

        for temp_info in info_list:
            temp_point = []
            temp_info = temp_info.split(' ')
            for i in range(2 * self.number_landmarks):
                temp_point.append(float(temp_info[i]))
            point_coord = np.array(temp_point, dtype=np.float).reshape(self.number_landmarks, 2)
            max_index = np.max(point_coord, axis=0)
            min_index = np.min(point_coord, axis=0)
            temp_box = np.array([min_index[0], min_index[1], max_index[0] - min_index[0],
                                 max_index[1] - min_index[1]])
            temp_name = os.path.join(self.root, self.image_dir, temp_info[-1])
            temp_name = temp_name.replace('\\', os.sep)
            temp_point = []
            for i in range(2 * self.number_landmarks, 4 * self.number_landmarks):
                temp_point.append(float(temp_info[i]))
            start_coord = np.array(temp_point, dtype=np.float).reshape(self.number_landmarks, 2)
            Data_base.append({'Img': temp_name,
                              'bbox': temp_box,
                              'point': point_coord,
                              'start': start_coord,})
        return Data_base

    def get_calibration_file_information(self):
        Data_base = []

        with open(self.calibration_annotation_file) as f:
            info_list = f.read().splitlines()
            f.close()

        for temp_info in info_list:
            temp_point = []
            temp_info = temp_info.split(' ')
            for i in range(2 * self.number_landmarks):
                temp_point.append(float(temp_info[i]))
            point_coord = np.array(temp_point, dtype=np.float).reshape(self.number_landmarks, 2)
            max_index = np.max(point_coord, axis=0)
            min_index = np.min(point_coord, axis=0)
            temp_box = np.array([min_index[0], min_index[1], max_index[0] - min_index[0],
                                 max_index[1] - min_index[1]])
            temp_name = os.path.join(self.root, self.image_dir, temp_info[-1])
            temp_name = temp_name.replace('\\', os.sep)
            Data_base.append({'Img': temp_name,
                              'bbox': temp_box,
                              'point': point_coord,})

        return Data_base

    def __getitem__(self, idx):
        input, meta, Flip_Flag = self._get_item_internal(idx, self.database, self.is_train)
        input_cal, meta_cal, _ = self._get_item_internal(idx, self.calibration_database, False,
                                                      Flip_Flag=Flip_Flag)

        return input, input_cal, meta, meta_cal

    def _get_item_internal(self, idx, database, is_train,
                           Flip_Flag=None):
        db_slic = copy.deepcopy(database[idx])
        Img_path = db_slic['Img']
        BBox = db_slic['bbox']
        Points = db_slic['point']
        if 'start' in db_slic:
            StartPoints = db_slic['start']
        else:
            StartPoints = None
        Annotated_Points = Points.copy()

        Img = cv2.imread(Img_path)

        Img_shape = Img.shape
        Img = cv2.cvtColor(Img, cv2.COLOR_RGB2BGR)
        if len(Img_shape) < 3:
            Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        else:
            if Img_shape[2] == 4:
                Img = cv2.cvtColor(Img, cv2.COLOR_RGBA2RGB)
            elif Img_shape[2] == 1:
                Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)

        if is_train == True:
            Rotation_Factor = self.Rotation_Factor * np.pi / 180.0
            Scale_Factor = self.Scale_Factor
            Translation_X_Factor = self.Translation_Factor
            Translation_Y_Factor = self.Translation_Factor

            angle = np.clip(np.random.normal(0, Rotation_Factor), -2 * Rotation_Factor, 2 * Rotation_Factor)
            Scale = np.clip(np.random.normal(self.Fraction, Scale_Factor), self.Fraction - Scale_Factor, self.Fraction + Scale_Factor)

            Translation_X = np.clip(np.random.normal(0, Translation_X_Factor), -Translation_X_Factor, Translation_X_Factor)
            Translation_Y = np.clip(np.random.normal(0, Translation_Y_Factor), -Translation_Y_Factor, Translation_Y_Factor)

            trans = utils.get_transforms(BBox, Scale, angle, self.Image_size, shift_factor=[Translation_X, Translation_Y])

            input = cv2.warpAffine(Img, trans, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)

            for i in range(self.number_landmarks):
                Points[i,0:2] = utils.affine_transform(Points[i,0:2], trans)
                if StartPoints is not None:
                    StartPoints[i,0:2] = utils.affine_transform(StartPoints[i,0:2], trans)

            Flip_Flag = 0
            if self.Flip is True:
                Flip_Flag = np.random.randint(0, 2)
                if Flip_Flag == 1:
                    input, Points, StartPoints = self.Image_Flip(input, Points, StartPoints)

            if self.Transfer is True:
                Transfer_Flag = np.random.randint(0, 5)
                input = self.Channel_Transfer(input, Transfer_Flag)

            if self.Occlusion is True:
                Occlusion_Flag = np.random.randint(0, 2)
                if Occlusion_Flag == 1:
                    input = self.Create_Occlusion(input)

            if self.Transform is not None:
                input = self.Transform(input)

            meta = {'Img_path': Img_path,
                    'Points': Points / self.Image_size,
                    'BBox': BBox,
                    'trans': trans,
                    'Scale': Scale,
                    'angle': angle,
                    'Translation': [Translation_X, Translation_Y]}
            if StartPoints is not None:
                meta['StartPoints'] = StartPoints / self.Image_size

        else:
            trans = utils.get_transforms(BBox, self.Fraction, 0.0, self.Image_size, shift_factor=[0.0, 0.0])

            input = cv2.warpAffine(Img, trans, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)

            for i in range(self.number_landmarks):
                Points[i, 0:2] = utils.affine_transform(Points[i, 0:2], trans)
                if StartPoints is not None:
                    StartPoints[i, 0:2] = utils.affine_transform(StartPoints[i, 0:2], trans)

            if self.Flip is True and Flip_Flag == 1:
                input, Points, StartPoints = self.Image_Flip(input, Points, StartPoints)

            meta = {
                "Annotated_Points": Annotated_Points,
                'Img_path': Img_path,
                'Points': Points / self.Image_size,
                'BBox': BBox,
                'trans': trans,
                'Scale': self.Fraction,
                'angle': 0.0,
                'Translation': [0.0, 0.0],
            }
            if StartPoints is not None:
                meta['StartPoints'] = StartPoints / self.Image_size

            # target = np.zeros((self.number_landmarks, self.Heatmap_size, self.Heatmap_size))
            # tpts = Points / (self.Image_size - 1) * (self.Heatmap_size - 1)
            # for i in range(self.number_landmarks):
            #     if tpts[i, 1] > 0:
            #         target[i] = generate_target(target[i], tpts[i], self.sigma)

            if self.Transform is not None:
                input = self.Transform(input)

        return input, meta, Flip_Flag

    def Image_Flip(self, Img, GT, ST):
        Mirror_GT = []
        Mirror_ST = []
        width = Img.shape[1]
        for i in self.flip_index:
            Mirror_GT.append([width - 1 - GT[i][0], GT[i][1]])
            if ST is not None:
                Mirror_ST.append([width - 1 - ST[i][0], ST[i][1]])
        Img = cv2.flip(Img, 1)
        if ST is not None:
            Mirror_ST = numpy.array(Mirror_ST)
        else:
            Mirror_ST = None

        return Img, numpy.array(Mirror_GT), Mirror_ST
