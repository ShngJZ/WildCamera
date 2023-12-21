import sys, os, time, inspect, tabulate
import tqdm
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import transforms, ops

from loguru import logger

from tools.tools import to_cuda, DistributedSamplerNoEvenlyDivisible
from tools.visualization import tensor2rgb
from tools.calibrator import MonocularCalibrator
from WildCamera.datasets.IncdDataset import IncdDataset

class EvaluateCrop:
    def __init__(self):
        return

    def evaluate(self, model, args, group=None):
        measurements_all = dict()

        for dataaset in args.datasets_eval:

            augmentation = True

            val_dataset = IncdDataset(
                data_root=args.data_path,
                datasets_included=[dataaset],
                ht=args.input_height,
                wt=args.input_width,
                augmentation=augmentation,
                shuffleseed=None,
                split='test',
                augscale=args.augscale,
                no_change_prob=0.5,
                transformcategory='transform_crop'
            )
            measurements = evaluate_crop(model, val_dataset, args, group)
            measurements_all[dataaset] = measurements

        if args.gpu == 0:
            errors_all_mean = [['Datasets', 'miou', 'acc']]
            for key in measurements_all.keys():
                measurements = measurements_all[key]
                errors_all_mean.append(
                    [
                        key,
                        measurements['miou'],
                        measurements['acc'],
                    ]
                )
            logger.info(
                "\n" +
                tabulate.tabulate(errors_all_mean, headers='firstrow', tablefmt='fancy_grid', numalign="center", floatfmt=".3f"))

        torch.cuda.synchronize()
        return


def acquire_bbox(K, h, w):
    pts1 = torch.from_numpy(np.array([[0.5, 0.5, 1]])).float().cuda().T
    pts2 = torch.from_numpy(np.array([[w - 0.5, 0.5, 1]])).float().cuda().T
    pts3 = torch.from_numpy(np.array([[w - 0.5, h - 0.5, 1]])).float().cuda().T
    pts4 = torch.from_numpy(np.array([[0.5, h - 0.5, 1]])).float().cuda().T

    pts1 = K @ pts1
    pts2 = K @ pts2
    pts3 = K @ pts3
    pts4 = K @ pts4

    ptss = torch.cat([pts1, pts2, pts3, pts4])
    ptss = ptss.squeeze().cpu().numpy()[:, 0:2]

    ptss_ = [
        min(ptss[:, 0]),
        min(ptss[:, 1]),
        max(ptss[:, 0]),
        max(ptss[:, 1]),
    ]
    ptss_ = np.array(ptss_)
    ptss_ = torch.from_numpy(ptss_).float().cuda()
    return ptss_

def bbox_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)
    return aucs

@torch.no_grad()
def evaluate_crop(model, val_dataset, args, group=None):
    """ In Validation, random sample N Images to mimic a real test set situation  """
    monocalibrator = MonocularCalibrator(l1_th=args.l1_th)
    model.eval()
    if group is not None:
        sampler = DistributedSamplerNoEvenlyDivisible(val_dataset, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=1, sampler=sampler, pin_memory=False, shuffle=False, num_workers=args.eval_workers, drop_last=False)
    else:
        val_loader = DataLoader(val_dataset, batch_size=1, pin_memory=False, shuffle=False, num_workers=args.eval_workers, drop_last=False)

    measurements = torch.zeros(len(val_dataset), 2).cuda(device=args.gpu)
    measurements_detection = torch.zeros(len(val_dataset)).cuda(device=args.gpu)
    world_size = args.world_size

    th = 0.2

    for val_id, data_blob in enumerate(tqdm.tqdm(val_loader, disable=False)):
        sample_batched = to_cuda(data_blob)
        rgb, K, scaleM, T, K_raw, rgb_raw = sample_batched['rgb'], sample_batched['K'], sample_batched['scaleM'], sample_batched['T'], sample_batched['K_raw'], sample_batched['rgb_raw']

        incidence = model(rgb)
        Kgt, Kest = torch.clone(K).squeeze(0), monocalibrator.calibrate_camera_4DoF(incidence, RANSAC_trial=2048)

        Kest_wo_change = torch.inverse(scaleM) @ Kest
        fx_fy = (Kest_wo_change[0, 0, 0] / Kest_wo_change[0, 1, 1]).item()
        fx_fy = max(fx_fy, 1 / fx_fy)
        fx_fy = abs(fx_fy - 1)
        diffx = abs(Kest_wo_change[0, 0, 2].item() - sample_batched['size_wo_change'][1].item() / 2)
        diffx = diffx / sample_batched['size_wo_change'][1].item()
        diffy = abs(Kest_wo_change[0, 1, 2].item() - sample_batched['size_wo_change'][0].item() / 2)
        diffy = diffy / sample_batched['size_wo_change'][0].item()
        diffb = max(diffx, diffy)
        genuen = (fx_fy < th) and (diffb < th)

        genuen_gt = ((T[0].cpu() - torch.eye(3)).abs().max() < 1e-2).item()

        if genuen == genuen_gt:
            measurements_detection[val_id * world_size + args.gpu] = 1

        b, _, h, w = rgb.shape
        bbox_est = acquire_bbox(K_raw @ torch.inverse(Kest), h, w)
        bbox_gt = acquire_bbox(K_raw @ torch.inverse(Kgt), h, w)

        iou = ops.box_iou(bbox_gt.unsqueeze(0), bbox_est.unsqueeze(0))
        measurements[val_id * world_size + args.gpu, 0] = iou.squeeze().item()
        measurements[val_id * world_size + args.gpu, 1] = 1 - float(genuen_gt)

    if group is not None:
        dist.all_reduce(tensor=measurements, op=dist.ReduceOp.SUM, group=group)

    measurements = measurements.cpu().numpy()
    miou = np.sum(measurements[:, 0] * measurements[:, 1]) / np.sum(measurements[:, 1])
    measurements = {
        'miou': miou,
        'acc': measurements_detection.mean().item()
    }
    return measurements