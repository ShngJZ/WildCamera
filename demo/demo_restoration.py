import os, sys, inspect
project_root = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, project_root)

import PIL.Image as Image
import torch
from WildCamera.newcrfs.newcrf_incidencefield import NEWCRFIF

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
