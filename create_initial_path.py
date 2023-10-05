import argparse

from Config import cfg
from Config import update_config
from Dataloader import WFLW_Dataset, W300_Dataset

import os

import torchvision.transforms as transforms
import numpy as np


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

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if cfg.DATASET.DATASET == '300W':
        train_dataset = W300_Dataset(
            cfg, cfg.W300.ROOT, True,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )

    elif cfg.DATASET.DATASET == 'WFLW':
        train_dataset = WFLW_Dataset(
            cfg, cfg.WFLW.ROOT, True,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )

    elif cfg.DATASET.DATASET == 'HEADCAM':
        train_dataset = WFLW_Dataset(
            cfg, cfg.HEADCAM.ROOT, True,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]),
            annotation_file=os.path.join(cfg.HEADCAM.ROOT,
                                         'HEADCAM_annotations', 'list_85pt_rect_attr_train_test',
                                         'list_85pt_rect_attr_train.txt'),
            wflw_config=cfg.HEADCAM,
        )

    else:
        raise ValueError('Wrong Dataset')

    init_face = []
    for db_rec in train_dataset.database:
        # get bound box
        bbox = db_rec['bbox']

        # scale up by average factor
        scale_fac = train_dataset.Fraction
        center_point = bbox[0:2] + (bbox[2:4] / 2)
        wh_scaled = bbox[2:4] * scale_fac
        bbox_scaled = bbox
        bbox_scaled[0:2] = center_point - (wh_scaled * 0.5)
        bbox_scaled[2:4] = wh_scaled

        # now align points within this box and scale by 256
        landmarks = db_rec['point']

        landmarks_norm = ((landmarks - bbox_scaled[0:2]) / bbox_scaled[2:4]) * 256
        init_face.append(landmarks_norm)

    init_face = np.array(init_face)
    init_face = np.mean(init_face, axis=0)

    import matplotlib.pyplot as plt
    plt.scatter(init_face[:,0], init_face[:,1])
    plt.scatter(init_face[[26, 28],0], init_face[[26, 28],1], c='r')  # eye points
    plt.show()

    file_name = fr'X:\git\SLPT_Training\Config\init_{init_face.shape[0]}.npz'
    np.savez(file_name, init_face=init_face)

    # comparison_file_name = r'X:\git\SLPT_Training\Config\init_98.npz'
    # init_face = np.load(comparison_file_name)['init_face']
    #
    # plt.scatter(init_face[:,0], init_face[:,1])
    # plt.show()

    pass


if __name__ == '__main__':
    main_function()
