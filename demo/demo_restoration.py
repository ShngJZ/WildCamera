import os, sys, inspect
project_root = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, project_root)

import PIL.Image as Image
import torch
import numpy as np

from WildCamera.newcrfs.newcrf_incidencefield import NEWCRFIF
from tools.tools import resample_rgb
from tools.visualization import tensor2rgb

def restore_image(image, intrinsic, fixcrop=True):
    # Adjust Intrinsic with Crop and Resize
    w, h = image.size
    wt, ht = image.size

    # Fix Aspect Ratio, Avoid Image Reduced
    resizeM = np.eye(3)
    if intrinsic[0, 0] > intrinsic[1, 1]:
        r = intrinsic[0, 0] / intrinsic[1, 1]
        resizeM[1, 1] = r
        wt, ht = wt, ht * r
    else:
        r = intrinsic[1, 1] / intrinsic[0, 0]
        resizeM[0, 0] = r
        wt, ht = wt * r, ht
    wt, ht = int(np.ceil(wt).item()), int(np.ceil(ht).item())

    # Fix Crop
    cropM = np.eye(3)
    if fixcrop:
        intrinsic_ = resizeM @ intrinsic
        padding_lr, padding_ud = intrinsic_[0, 2] - wt / 2, intrinsic_[1, 2] - ht / 2
        if padding_lr < 0:
            cropM[0, 2] = -padding_lr
        if padding_ud < 0:
            cropM[1, 2] = -padding_ud

        wt, ht = int(np.ceil(wt + np.abs(padding_lr)).item()), int(np.ceil(ht + np.abs(padding_ud)).item())

    resample_matrix = np.linalg.inv(cropM @ resizeM)
    totensor = transforms.ToTensor()
    image_restore = resample_rgb(
        totensor(image).unsqueeze(0),
        torch.from_numpy(resample_matrix).float().view([1, 3, 3]),
        batch=1, ht=ht, wd=wt, device=torch.device("cpu")
    )

    return tensor2rgb(image_restore, viewind=0)


@torch.no_grad()
def main():
    model = NEWCRFIF(version='large07', pretrained=None)
    model.eval()
    model.cuda()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    ckpt_path = os.path.join(os.path.dirname(script_dir), 'model_zoo/Release', 'wild_camera_all.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)

    img_folder = os.path.join(project_root, 'asset', 'image_restoration')
    img_names = [
        'lenna_distorted_cropright.jpg',
        'lenna_distorted_cropleft.jpg',
    ]
    for img_name in img_names:
        img_path = os.path.join(img_folder, img_name)
        img = Image.open(img_path)

        export_root = os.path.join(img_folder, '{}_restored.jpg'.format(img_name.split('.')[0]))
        intrinsic, _ = model.inference(img, wtassumption=False)
        model.restore_image(img, intrinsic).save(export_root)

if __name__ == '__main__':
    main()
