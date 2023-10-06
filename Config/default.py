#--------------------------------------
#Author: Jiahao.Xia
#--------------------------------------

from yacs.config import CfgNode as CN

import os

_C = CN()
_C.GPUS = (0, )
_C.WORKERS = 0
_C.PIN_MEMORY = True
_C.AUTO_RESUME = True
_C.PRINT_FREQ = 10

_C.DATASET = CN()
_C.DATASET.CHANNEL = 3
_C.DATASET.DATASET = 'HEADCAM'

_C.DEBUG = CN()
_C.DEBUG.DEBUG = True
_C.DEBUG.SAVE_BATCH_IMAGES_GT = True
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = True
_C.DEBUG.SAVE_HEATMAPS_GT = True
_C.DEBUG.SAVE_HEATMAPS_PRED = True

_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 1

_C.MODEL = CN()
_C.MODEL.NAME = "Sparase_alignment"
_C.MODEL.IMG_SIZE = 256
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = None
_C.MODEL.POINT_NUM = 85
_C.MODEL.OUT_DIM = 256
_C.MODEL.NUM_JOINTS = 85
_C.MODEL.ITER_NUM = 1
_C.MODEL.BACKBONE = 'resnet50'
_C.MODEL.TRAINABLE = True
_C.MODEL.INTER_LAYER = True
_C.MODEL.DILATION = False
_C.MODEL.EMBEDDING = 'v2'
_C.MODEL.PRETRAINED = "./Config/hrnetv2_w18_imagenet_pretrained.pth"
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.HEATMAP = 64
_C.MODEL.SAMPLE_NUM = 7

_C.TRANSFORMER = CN()
_C.TRANSFORMER.NHEAD = 8
_C.TRANSFORMER.NUM_DECODER = 6
_C.TRANSFORMER.FEED_DIM = 1024

_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

_C.LOSS = CN()
_C.LOSS.USE_TARGET_WEIGHT = True

_C.TRAIN = CN()
_C.TRAIN.TRAIN = True
_C.TRAIN.SHUFFLE = True
_C.TRAIN.LR = 0.001
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [120, 140]
_C.TRAIN.OPTIMIZER = "adam"
_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.NUM_EPOCH = 150

_C.TEST = CN()
_C.TEST.POST_PROCESS = True
_C.TEST.BATCH_SIZE_PER_GPU = 32

_C.HYPERPARAMETERS = CN()

_C.WFLW = CN()
_C.WFLW.ROOT = './Data/WFLW'
_C.WFLW.NUM_POINT = 98
_C.WFLW.FRACTION = 1.20
_C.WFLW.SCALE = 0.05
_C.WFLW.ROTATION = 15
_C.WFLW.TRANSLATION = 0.05
_C.WFLW.OCCLUSION_MEAN = 0.20
_C.WFLW.OCCLUSION_STD = 0.08
_C.WFLW.DATA_FORMAT = "RGB"
_C.WFLW.FLIP = True
_C.WFLW.CHANNEL_TRANSFER = True
_C.WFLW.OCCLUSION = True
_C.WFLW.INITIAL_PATH = './Config/init_98.npz'

_C.W300 = CN()
_C.W300.ROOT = './Data/300W'
_C.W300.NUM_POINT = 68
_C.W300.FRACTION = 1.20
_C.W300.SCALE = 0.05
_C.W300.ROTATION = 15
_C.W300.TRANSLATION = 0.05
_C.W300.OCCLUSION_MEAN = 0.20
_C.W300.OCCLUSION_STD = 0.08
_C.W300.DATA_FORMAT = "RGB"
_C.W300.FLIP = True
_C.W300.CHANNEL_TRANSFER = True
_C.W300.OCCLUSION = True
_C.W300.INITIAL_PATH = './Config/init_68.npz'

_C.HEADCAM = CN()
_C.HEADCAM.ROOT = './Data/HEADCAM'
_C.HEADCAM.NUM_POINT = 85
_C.HEADCAM.FRACTION = 1.20
_C.HEADCAM.SCALE = 0.05
_C.HEADCAM.ROTATION = 15
_C.HEADCAM.TRANSLATION = 0.05
_C.HEADCAM.OCCLUSION_MEAN = 0.20
_C.HEADCAM.OCCLUSION_STD = 0.08
_C.HEADCAM.DATA_FORMAT = "RGB"
_C.HEADCAM.FLIP = True
_C.HEADCAM.CHANNEL_TRANSFER = False
_C.HEADCAM.OCCLUSION = False
_C.HEADCAM.INITIAL_PATH = './Config/init_85.npz'

# High-Resoluion Net
_C.MODEL.EXTRA = CN()
_C.MODEL.EXTRA.PRETRAINED_LAYERS = ['*']
_C.MODEL.EXTRA.STEM_INPLANES = 64
_C.MODEL.EXTRA.FINAL_CONV_KERNEL = 1
_C.MODEL.EXTRA.WITH_HEAD = True

_C.MODEL.EXTRA.STAGE2 = CN()
_C.MODEL.EXTRA.STAGE2.NUM_MODULES = 1
_C.MODEL.EXTRA.STAGE2.NUM_BRANCHES = 2
_C.MODEL.EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
_C.MODEL.EXTRA.STAGE2.NUM_CHANNELS = [18, 36]
_C.MODEL.EXTRA.STAGE2.BLOCK = 'BASIC'
_C.MODEL.EXTRA.STAGE2.FUSE_METHOD = 'SUM'

_C.MODEL.EXTRA.STAGE3 = CN()
_C.MODEL.EXTRA.STAGE3.NUM_MODULES = 1
_C.MODEL.EXTRA.STAGE3.NUM_BRANCHES = 3
_C.MODEL.EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
_C.MODEL.EXTRA.STAGE3.NUM_CHANNELS = [18, 36, 72]
_C.MODEL.EXTRA.STAGE3.BLOCK = 'BASIC'
_C.MODEL.EXTRA.STAGE3.FUSE_METHOD = 'SUM'

_C.MODEL.EXTRA.STAGE4 = CN()
_C.MODEL.EXTRA.STAGE4.NUM_MODULES = 1
_C.MODEL.EXTRA.STAGE4.NUM_BRANCHES = 4
_C.MODEL.EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
_C.MODEL.EXTRA.STAGE4.NUM_CHANNELS = [18, 36, 72, 144]
_C.MODEL.EXTRA.STAGE4.BLOCK = 'BASIC'
_C.MODEL.EXTRA.STAGE4.FUSE_METHOD = 'SUM'

def update_config(cfg, args):
    cfg.defrost()

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    if args.dataDir:
        cfg.DATA_DIR = args.dataDir
        cfg.WFLW.ROOT = os.path.join(cfg.DATA_DIR, 'WFLW')
        cfg.HEADCAM.ROOT = os.path.join(cfg.DATA_DIR, 'HEADCAM')

    if args.target:
        cfg.TARGET = args.target


    if cfg.MODEL.PRETRAINED is not None:
        cfg.MODEL.PRETRAINED = os.path.join(
            cfg.DATA_DIR, cfg.MODEL.PRETRAINED
        )
    else:
        cfg.MODEL.PRETRAINED = None

    cfg.freeze()
