import math
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch.utils.data import Sampler

# -- # Common Functions
class InputPadder:
    """ Pads images such that dimensions are divisible by ds """
    def __init__(self, dims, mode='leftend', ds=32):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // ds) + 1) * ds - self.ht) % ds
        pad_wd = (((self.wd // ds) + 1) * ds - self.wd) % ds
        if mode == 'leftend':
            self._pad = [0, pad_wd, 0, pad_ht]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]

        self.mode = mode

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def coords_gridN(batch, ht, wd, device):
    coords = torch.meshgrid(
        (
            torch.linspace(-1 + 1 / ht, 1 - 1 / ht, ht, device=device),
            torch.linspace(-1 + 1 / wd, 1 - 1 / wd, wd, device=device),
        )
    )

    coords = torch.stack((coords[1], coords[0]), dim=0)[
        None
    ].repeat(batch, 1, 1, 1)
    return coords

def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    return batch

def rename_ckpt(ckpt):
    renamed_ckpt = dict()
    for k in ckpt.keys():
        if 'module.' in k:
            renamed_ckpt[k.replace('module.', '')] = torch.clone(ckpt[k])
        else:
            renamed_ckpt[k] = torch.clone(ckpt[k])
    return renamed_ckpt

def resample_rgb(rgb, scaleM, batch, ht, wd, device):
    coords = coords_gridN(batch, ht, wd, device)
    x, y = torch.split(coords, 1, dim=1)
    x = (x + 1) / 2 * wd
    y = (y + 1) / 2 * ht

    scaleM = scaleM.squeeze()

    x = x * scaleM[0, 0] + scaleM[0, 2]
    y = y * scaleM[1, 1] + scaleM[1, 2]

    _, _, orgh, orgw = rgb.shape
    x = x / orgw * 2 - 1.0
    y = y / orgh * 2 - 1.0

    coords = torch.stack([x.squeeze(1), y.squeeze(1)], dim=3)
    rgb_resized = torch.nn.functional.grid_sample(rgb, coords, mode='bilinear', align_corners=True)

    return rgb_resized

def intrinsic2incidence(K, b, h, w, device):
    coords = coords_gridN(b, h, w, device)

    x, y = torch.split(coords, 1, dim=1)
    x = (x + 1) / 2.0 * w
    y = (y + 1) / 2.0 * h

    pts3d = torch.cat([x, y, torch.ones_like(x)], dim=1)
    pts3d = rearrange(pts3d, 'b d h w -> b h w d')
    pts3d = pts3d.unsqueeze(dim=4)

    K_ex = K.view([b, 1, 1, 3, 3])
    pts3d = torch.linalg.inv(K_ex) @ pts3d
    pts3d = torch.nn.functional.normalize(pts3d, dim=3)
    return pts3d

def apply_augmentation(rgb, K, seed=None, augscale=2.0, no_change_prob=0.0):
    _, h, w = rgb.shape

    if seed is not None:
        np.random.seed(seed)

    if np.random.uniform(0, 1) < no_change_prob:
        extension_rx, extension_ry = 1.0, 1.0
    else:
        extension_rx, extension_ry = np.random.uniform(1, augscale), np.random.uniform(1, augscale)

    hs, ws = int(np.ceil(h * extension_ry)), int(np.ceil(w * extension_rx))

    stx = float(np.random.randint(0, int(ws - w + 1), 1).item() + 0.5)
    edx = float(stx + w - 1)
    sty = float(np.random.randint(0, int(hs - h + 1), 1).item() + 0.5)
    edy = float(sty + h - 1)

    stx = stx / ws * w
    edx = edx / ws * w

    sty = sty / hs * h
    edy = edy / hs * h

    ptslt, ptslt_ = np.array([stx, sty, 1]), np.array([0.5, 0.5, 1])
    ptsrt, ptsrt_ = np.array([edx, sty, 1]), np.array([w-0.5, 0.5, 1])
    ptslb, ptslb_ = np.array([stx, edy, 1]), np.array([0.5, h-0.5, 1])
    ptsrb, ptsrb_ = np.array([edx, edy, 1]), np.array([w-0.5, h-0.5, 1])

    pts1 = np.stack([ptslt, ptsrt, ptslb, ptsrb], axis=1)
    pts2 = np.stack([ptslt_, ptsrt_, ptslb_, ptsrb_], axis=1)

    T_num = pts1 @ pts2.T @ np.linalg.inv(pts2 @ pts2.T)
    T = np.eye(3)
    T[0, 0] = T_num[0, 0]
    T[0, 2] = T_num[0, 2]
    T[1, 1] = T_num[1, 1]
    T[1, 2] = T_num[1, 2]
    T = torch.from_numpy(T).float()

    K_trans = torch.inverse(T) @ K

    b = 1
    _, h, w = rgb.shape
    device = rgb.device
    rgb_trans = resample_rgb(rgb.unsqueeze(0), T, b, h, w, device).squeeze(0)
    return rgb_trans, K_trans, T



class IncidenceLoss(nn.Module):
    def __init__(self, loss='cosine'):
        super(IncidenceLoss, self).__init__()
        self.loss = loss
        self.smoothl1 = torch.nn.SmoothL1Loss(beta=0.2)

    def forward(self, incidence, K):
        b, _, h, w = incidence.shape
        device = incidence.device

        incidence_gt = intrinsic2incidence(K, b, h, w, device)
        incidence_gt = incidence_gt.squeeze(4)
        incidence_gt = rearrange(incidence_gt, 'b h w d -> b d h w')

        if self.loss == 'cosine':
            loss = 1 - torch.cosine_similarity(incidence, incidence_gt, dim=1)
        elif self.loss == 'absolute':
            loss = self.smoothl1(incidence, incidence_gt)

        loss = loss.mean()
        return loss


class DistributedSamplerNoEvenlyDivisible(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        num_samples = int(math.floor(len(self.dataset) * 1.0 / self.num_replicas))
        rest = len(self.dataset) - num_samples * self.num_replicas
        if self.rank < rest:
            num_samples += 1
        self.num_samples = num_samples
        self.total_size = len(dataset)
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch