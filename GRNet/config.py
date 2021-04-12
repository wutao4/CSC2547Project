# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 17:18:04
# @Email:  cshzxie@gmail.com

from easydict import EasyDict as edict

__C                                              = edict()
cfg                                              = __C

#
# Dataset Config
#
__C.DATASETS                                     = edict()
__C.DATASETS.COMPLETION3D                        = edict()
__C.DATASETS.COMPLETION3D.CATEGORY_FILE_PATH     = './datasets/Completion3D.json'
__C.DATASETS.COMPLETION3D.PARTIAL_POINTS_PATH    = '/home/SENSETIME/xiehaozhe/Datasets/Completion3D/%s/partial/%s/%s.h5'
__C.DATASETS.COMPLETION3D.COMPLETE_POINTS_PATH   = '/home/SENSETIME/xiehaozhe/Datasets/Completion3D/%s/gt/%s/%s.h5'
__C.DATASETS.SHAPENET                            = edict()
__C.DATASETS.SHAPENET.CATEGORY_FILE_PATH         = './datasets/ShapeNet.json'
__C.DATASETS.SHAPENET.N_RENDERINGS               = 8
__C.DATASETS.SHAPENET.N_POINTS                   = 16384
__C.DATASETS.SHAPENET.PARTIAL_POINTS_PATH        = '/home/SENSETIME/xiehaozhe/Datasets/ShapeNet/ShapeNetCompletion/%s/partial/%s/%s/%02d.pcd'
__C.DATASETS.SHAPENET.COMPLETE_POINTS_PATH       = '/home/SENSETIME/xiehaozhe/Datasets/ShapeNet/ShapeNetCompletion/%s/complete/%s/%s.pcd'
__C.DATASETS.KITTI                               = edict()
__C.DATASETS.KITTI.CATEGORY_FILE_PATH            = './datasets/KITTI.json'
__C.DATASETS.KITTI.PARTIAL_POINTS_PATH           = '/home/SENSETIME/xiehaozhe/Datasets/KITTI/cars/%s.pcd'
__C.DATASETS.KITTI.BOUNDING_BOX_FILE_PATH        = '/home/SENSETIME/xiehaozhe/Datasets/KITTI/bboxes/%s.txt'
__C.DATASETS.CLEARGRASP                          = edict()
__C.DATASETS.CLEARGRASP.CATEGORY_FILE_PATH       = './datasets/ClearGrasp.json'
# __C.DATASETS.CLEARGRASP.CATEGORY_FILE_PATH       = './datasets/ClearGrasp-test.json'
__C.DATASETS.CLEARGRASP.PARTIAL_POINTS_PATH      = './datasets/cleargrasp-dataset-pcd/%s/transparent/%s.pcd'
__C.DATASETS.CLEARGRASP.COMPLETE_POINTS_PATH     = './datasets/cleargrasp-dataset-pcd/%s/opaque/%s.pcd'
__C.DATASETS.FRANKASCAN                          = edict()
__C.DATASETS.FRANKASCAN.CATEGORY_FILE_PATH       = './datasets/FrankaScan.json'
# __C.DATASETS.FRANKASCAN.CATEGORY_FILE_PATH       = './datasets/FrankaScan-test.json'
__C.DATASETS.FRANKASCAN.POINTS_DIR_PATH          = './datasets/frankascan/%s/%s/'
__C.DATASETS.FRANKASCAN.PARTIAL_POINTS_PATH      = './datasets/frankascan/%s/%s/depth2pcd_%s.pcd'
__C.DATASETS.FRANKASCAN.COMPLETE_POINTS_PATH     = './datasets/frankascan/%s/%s/depth2pcd_GT_%s.pcd'

#
# Dataset
#
__C.DATASET                                      = edict()
# Dataset Options: Completion3D, ShapeNet, ShapeNetCars, KITTI, ClearGrasp, FrankaScan
__C.DATASET.TRAIN_DATASET                        = 'FrankaScan'
__C.DATASET.TEST_DATASET                         = 'FrankaScan'

#
# Constants
#
__C.CONST                                        = edict()
__C.CONST.DEVICE                                 = '0'
__C.CONST.NUM_WORKERS                            = 8
__C.CONST.N_INPUT_POINTS                         = 2048
__C.CONST.BIG_N_INPUT_POINTS                     = 16384

#
# Directories
#
__C.DIR                                          = edict()
__C.DIR.OUT_PATH                                 = './output'

#
# Memcached
#
__C.MEMCACHED                                    = edict()
__C.MEMCACHED.ENABLED                            = False
__C.MEMCACHED.LIBRARY_PATH                       = '/mnt/lustre/share/pymc/py3'
__C.MEMCACHED.SERVER_CONFIG                      = '/mnt/lustre/share/memcached_client/server_list.conf'
__C.MEMCACHED.CLIENT_CONFIG                      = '/mnt/lustre/share/memcached_client/client.conf'

#
# Network
#
__C.NETWORK                                      = edict()
__C.NETWORK.N_SAMPLING_POINTS                    = 2048
__C.NETWORK.GRIDDING_LOSS_SCALES                 = [128]
__C.NETWORK.GRIDDING_LOSS_ALPHAS                 = [0.1]

#
# Train
#
__C.TRAIN                                        = edict()
__C.TRAIN.BATCH_SIZE                             = 32
__C.TRAIN.N_EPOCHS                               = 500
__C.TRAIN.SAVE_FREQ                              = 25
# __C.TRAIN.LEARNING_RATE                          = 5e-5 0.5877
# __C.TRAIN.LEARNING_RATE                          = 2e-4 0.6069
# __C.TRAIN.LEARNING_RATE                          = 5e-4 0.6391-0.6424 / epoch400 ~0.640 / epoch300 ~0.630
# __C.TRAIN.LEARNING_RATE                          = 1e-3 0.6339
__C.TRAIN.LEARNING_RATE                          = 1e-4
__C.TRAIN.LR_MILESTONES                          = [50]
__C.TRAIN.GAMMA                                  = .5
__C.TRAIN.BETAS                                  = (.9, .999)
__C.TRAIN.WEIGHT_DECAY                           = 0

#
# Test
#
__C.TEST                                         = edict()
__C.TEST.METRIC_NAME                             = 'ChamferDistance'
