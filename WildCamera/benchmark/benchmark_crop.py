from __future__ import print_function, division
import os, sys, inspect, copy
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
sys.path.insert(0, project_root)

import warnings
warnings.filterwarnings('ignore')
import torch
import argparse
from loguru import logger
from pprint import pprint

from WildCamera.newcrfs.newcrf_incidencefield import NEWCRFIF
from WildCamera.evaluation.evaluate_crop import EvaluateCrop

parser = argparse.ArgumentParser(description='NeWCRFs PyTorch implementation.', fromfile_prefix_chars='@')
parser.add_argument('--load_ckpt',                 type=str,   help='path of ckpt')
parser.add_argument('--data_path',                 type=str,   help='path to the data', default='data/MonoCalib')

def main_worker(args):
    args.gpu = 0
    pprint(vars(args))

    model = NEWCRFIF(version='large07', pretrained=None)
    model.load_state_dict(torch.load(args.load_ckpt, map_location="cpu"), strict=True)
    model.eval()
    model.cuda()
    logger.info("Load Model from %s" % args.load_ckpt)

    evaluate_crop = EvaluateCrop()
    evaluate_crop.evaluate(model, args, group=None)

if __name__ == '__main__':
    args = parser.parse_args()

    args.world_size = 1
    args.augscale = 2.0
    args.input_height = 480
    args.input_width = 640
    args.eval_workers = 4
    args.l1_th = 0.02

    args.datasets_train = [
        'KITTI',
        'NYUv2',
        'ARKitScenes',
        'Waymo',
        'RGBD',
        'ScanNet',
        'MVS',
    ]
    args.datasets_eval = [
        'KITTI',
        'NYUv2',
        'ARKitScenes',
        'Waymo',
        'RGBD',
        'ScanNet',
        'MVS',
    ]

    args.load_ckpt = os.path.join(project_root, 'model_zoo/Release', 'wild_camera_all.pth')
    main_worker(args)
