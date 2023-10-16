# Put most Common Functions Here
import os, cv2
import torch
import torch.nn.functional as F
import torch.distributed as dist
import math
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

from torch.utils.data import Sampler
from torchvision import transforms

# -- # Visualization
def tensor2disp(tensor, vmax=0.18, percentile=None, viewind=0):
    cm = plt.get_cmap('magma')
    tnp = tensor[viewind, 0, :, :].detach().cpu().numpy()
    if percentile is not None:
        if np.sum(tnp > 0) > 100:
            vmax = np.percentile(tnp[tnp > 0], 95)
        else:
            vmax = 1.0
    tnp = tnp / vmax
    tnp = (cm(tnp) * 255).astype(np.uint8)
    return Image.fromarray(tnp[:, :, 0:3])

def tensor2grad(gradtensor, percentile=95, pos_bar=0, neg_bar=0, viewind=0):
    cm = plt.get_cmap('bwr')
    gradnumpy = gradtensor.detach().cpu().numpy()[viewind, 0, :, :]

    selector_pos = gradnumpy > 0
    if np.sum(selector_pos) > 1:
        if pos_bar <= 0:
            pos_bar = np.percentile(gradnumpy[selector_pos], percentile)
        gradnumpy[selector_pos] = gradnumpy[selector_pos] / pos_bar / 2

    selector_neg = gradnumpy < 0
    if np.sum(selector_neg) > 1:
        if neg_bar >= 0:
            neg_bar = -np.percentile(-gradnumpy[selector_neg], percentile)
        gradnumpy[selector_neg] = -gradnumpy[selector_neg] / neg_bar / 2

    disp_grad_numpy = gradnumpy + 0.5
    colorMap = cm(disp_grad_numpy)[:, :, 0:3]
    return Image.fromarray((colorMap * 255).astype(np.uint8))

def tensor2rgb(tensor, viewind=0):
    tnp = tensor.detach().cpu().permute([0, 2, 3, 1]).contiguous()[viewind, :, :, :].numpy()
    if np.max(tnp) <= 2:
        tnp = tnp * 255
    tnp = np.clip(tnp, a_min=0, a_max=255).astype(np.uint8)
    return Image.fromarray(tnp)
