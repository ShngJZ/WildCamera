from __future__ import print_function, division
import warnings, os
warnings.filterwarnings('ignore')

from WildCamera.newcrfs.newcrf_incidencefield import NEWCRFIF

import torch
from PIL import Image

project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

if __name__ == '__main__':
    # NeWCRFs model
    model = NEWCRFIF(version='large07', pretrained=None)
    model.eval()
    model.cuda()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    ckpt_path = os.path.join(os.path.dirname(script_dir), 'model_zoo/Release', 'wild_camera_all.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)

    images_folder = os.path.join(project_dir, 'asset', 'images-from-github-wt-intrinsic')
    info_path = os.path.join(images_folder, 'intrinsic_gt.txt')
    with open(info_path) as file:
        infos = [line.rstrip() for line in file]

    for idx, info in enumerate(infos):
        imgname, focalgt, source = info.split(' ')

        images_path = os.path.join(images_folder, imgname)
        intrinsic, _ = model.inference(Image.open(images_path), wtassumption=False)
        focal = intrinsic[0, 0].item()

        print("Image Name: %s, Est Focal %.1f, Gt Focal %.1f, Source - %s" % (imgname, focal, float(focalgt), source))