import os
import torch
import numpy as np

from WildCamera.datasets.MegaDepth import MegaDepth
from WildCamera.datasets.GSV import GSV
from WildCamera.datasets.GenericDataset import GenericDataset

from torch.utils.data import Subset

def IncdDataset(
        data_root,
        datasets_included,
        ht=384, wt=512,
        augmentation=False,
        shuffleseed=None,
        split='train',
        dataset_favour_long=0.0,
        augscale=2.0,
        no_change_prob=0.1,
        coloraugmentation=False,
        coloraugmentation_scale=0.0
) -> None:
    datasets = dict()

    for datasetname in datasets_included:
        if datasetname == 'GSV':
            datasets[datasetname] = GSV(
                os.path.join(data_root, 'GSV'),
                ht=ht, wt=wt,
                shuffleseed=shuffleseed,
                split=split
            )
        elif datasetname == 'MegaDepth':
            datasets[datasetname] = MegaDepth(
                os.path.join(data_root, 'MegaDepth'),
                ht=ht, wt=wt,
                shuffleseed=shuffleseed,
                split=split,
            )
        elif datasetname == 'BIWIRGBDID':
            datasets[datasetname] = GenericDataset(
                os.path.join(data_root, 'BIWIRGBDID'),
                ht=ht, wt=wt,
                augmentation=False,
                shuffleseed=shuffleseed,
                split=split,
                datasetname=datasetname,
                coloraugmentation=coloraugmentation,
                coloraugmentation_scale=coloraugmentation_scale
            )
        elif datasetname == 'CAD120':
            datasets[datasetname] = GenericDataset(
                os.path.join(data_root, 'CAD120'),
                ht=ht, wt=wt,
                augmentation=False,
                shuffleseed=shuffleseed,
                split=split,
                datasetname=datasetname,
                coloraugmentation=coloraugmentation,
                coloraugmentation_scale=coloraugmentation_scale
            )
        elif (datasetname == 'Objectron') or (datasetname == 'MVImgNet'):
            augscale_obj = 1 + (augscale - 1) / 5
            datasets[datasetname] = GenericDataset(
                os.path.join(data_root, datasetname),
                ht=ht, wt=wt,
                augmentation=True,
                shuffleseed=shuffleseed,
                split=split,
                datasetname=datasetname,
                augscale=augscale_obj,
                no_change_prob=no_change_prob,
                coloraugmentation=coloraugmentation,
                coloraugmentation_scale=coloraugmentation_scale
            )
        else:
            datasets[datasetname] = GenericDataset(
                os.path.join(data_root, datasetname),
                ht=ht, wt=wt,
                augmentation=augmentation,
                shuffleseed=shuffleseed,
                split=split,
                datasetname=datasetname,
                augscale=augscale,
                no_change_prob=no_change_prob,
                coloraugmentation=coloraugmentation,
                coloraugmentation_scale=coloraugmentation_scale
            )

    if split == 'train':
        min_len = np.array([len(datasets[key]) for key in datasets.keys()]).min()

        datasets_sub = dict()
        for key in datasets.keys():
            dataset = datasets[key]
            subsample_len = np.floor((1 - dataset_favour_long) * min_len + dataset_favour_long * len(dataset))
            subsample_len = int(subsample_len)

            indices = torch.randperm(len(dataset))
            subsampled_dataset = Subset(dataset, indices[0:subsample_len])
            datasets_sub[key] = subsampled_dataset

        incdataset = torch.utils.data.ConcatDataset([datasets_sub[key] for key in datasets_sub.keys()])
    else:
        incdataset = torch.utils.data.ConcatDataset([datasets[key] for key in datasets.keys()])

    return incdataset
