import os, io, glob, natsort, random, copy
import h5py
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

class GSV:
    def __init__(self, data_root, ht=384, wt=512, shuffleseed=None, split='train') -> None:
        split_path = os.path.join('splits', 'gsv_{}.txt'.format(split))

        with open(split_path) as file:
            data_names = [line.rstrip() for line in file]

        if split == 'train':
            if shuffleseed is not None:
                random.seed(shuffleseed)
            data_names = copy.deepcopy(data_names) + copy.deepcopy(data_names)
            random.shuffle(data_names)

        self.data_root = data_root
        self.wt, self.ht = wt, ht
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.tensor = transforms.ToTensor()
        self.data_names = data_names

    def __len__(self):
        return len(self.data_names)

    def load_im(self, im_ref):
        im = Image.open(im_ref)
        return im

    def __getitem__(self, idx):
        # read intrinsics of original size
        scene_name, entry_name = self.data_names[idx].split(' ')
        h5pypath = os.path.join(self.data_root, '{}.hdf5'.format(scene_name))

        with h5py.File(h5pypath, 'r') as hf:
            K_color = np.array(hf['intrinsic'][entry_name])

            # Load positive pair data
            rgb = self.load_im(io.BytesIO(np.array(hf['color'][entry_name])))
            w, h = rgb.size
            rgb = rgb.resize((self.wt, self.ht))

            scaleM = np.eye(3)
            scaleM[0, 0] = self.wt / w
            scaleM[1, 1] = self.ht / h
            aspect_ratio_restoration = (scaleM[1, 1] / scaleM[0, 0]).item()

            K = torch.from_numpy(scaleM @ K_color).float()

            fovy = 2 * np.arctan(h / K_color[1, 1] / 2)
            fovy = fovy.item()

        # Recompute camera intrinsic matrix due to the resize
        rgb = self.normalize(self.tensor(rgb))

        data_dict = {
            'K': K,
            'rgb': rgb,
            'K_raw': torch.clone(K),
            'rgb_raw': torch.clone(rgb),
            'aspect_ratio_restoration': aspect_ratio_restoration,
            'fovy': fovy,
            # Only Required in Crop Evaluation and Visualization in Training
            'T': torch.eye(3, dtype=torch.float32),
            'scaleM': scaleM.astype(np.float32),
            'size_wo_change': (h, w),
        }

        return data_dict