from __future__ import print_function, division
import os, copy, tqdm, glob, natsort
import warnings
warnings.filterwarnings('ignore')

from WildCamera.newcrfs.newcrf_incidencefield import NEWCRFIF

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def draw_focalbar(rgbaug, fest, minfocal, maxfocal):
    tmp = copy.deepcopy(rgbaug)

    font_size = 28
    font = ImageFont.truetype("arial.ttf", size=font_size)

    rgbaug = copy.deepcopy(tmp)
    w, h = rgbaug.size
    rgbaug.putalpha(255)

    paddingr = 0.1
    barsty = h * paddingr
    baredy = h * (1 - paddingr)
    barx = w * 0.9
    horizonbarnum = 7 * 5
    horizonbarlen = 0.01

    white = Image.new('RGBA', rgbaug.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(white)
    draw.rectangle(
        [
            (w * 0.9 - w * horizonbarlen - 15 / 640 * w, h * paddingr - 15 / 480 * h),
            w * 0.9 + w * horizonbarlen + 35 / 640 * w, h * (1 - paddingr) + 15 / 480 * h
        ],
        fill=(255, 255, 255, 128)
    )
    rgbaug = Image.alpha_composite(rgbaug, white)

    draw = ImageDraw.Draw(rgbaug)
    draw.line((barx, barsty, barx, baredy), fill=(0, 0, 0, 255), width=5)

    for i in range(horizonbarnum + 1):
        r = i / horizonbarnum
        bary = h * paddingr * r + h * (1 - paddingr) * (1 - r)
        barstx = w * 0.9 - w * horizonbarlen
        baredx = w * 0.9 + w * horizonbarlen
        draw.line((barstx, bary, baredx, bary), fill=(0, 0, 0, 255), width=int(3))

    horizonbarnum = 7
    horizonbarlen = 0.02

    for i in range(horizonbarnum + 1):
        r = i / horizonbarnum
        bary = h * paddingr * r + h * (1 - paddingr) * (1 - r)
        barstx = w * 0.9 - w * horizonbarlen
        baredx = w * 0.9 + w * horizonbarlen
        draw.line((barstx, bary, baredx, bary), fill=(0, 0, 0, 255), width=3)

        textpadding = 30 / 640 * w
        draw.text((barstx + textpadding, bary - 5 / 480 * h), str(int(maxfocal * r + minfocal * (1 - r))), fill=(0, 0, 0, 255), font=font)

    r = (fest - minfocal) / (maxfocal - minfocal)
    bary = h * paddingr * r + h * (1 - paddingr) * (1 - r)
    barstx = w * 0.9 - w * horizonbarlen
    baredx = w * 0.9 + w * horizonbarlen
    draw.line((barstx, bary, baredx, bary), fill=(255, 165, 0, 255), width=5)
    return rgbaug

def smooth_width(fests_precomputed, idx, width=3):
    # Smooth as an average of Three Frames
    stidx = idx - int((width - 1) / 2)
    edidx = idx + int((width - 1) / 2)

    if stidx < 0:
        stidx = 0

    if edidx >= len(fests_precomputed):
        edidx = len(fests_precomputed) - 1

    return np.mean(fests_precomputed[stidx:edidx+1])


@torch.no_grad()
def draw_dollyzoom(model):
    """ In Validation, random sample N Images to mimic a real test set situation  """
    model.eval()

    dollyzoom_folder = os.path.join(project_dir, 'asset', 'dollyzoom')
    output_dollyzoom = os.path.join(project_dir, 'output', 'dollyzoom')
    dollyzoomvideos = [
        'dz1',
        'dz2',
        'dz3'
    ]

    for dollyzoomvideo in dollyzoomvideos:
        dollyzoomimg_folder = os.path.join(dollyzoom_folder, dollyzoomvideo)
        jpgs = glob.glob(os.path.join(dollyzoomimg_folder, '*.jpg'))
        jpgs = natsort.natsorted(jpgs)

        fests = list()
        for jpg in tqdm.tqdm(jpgs):
            intrinsic, _ = model.inference(Image.open(jpg), wtassumption=False)
            fest = intrinsic[0, 0]
            fests.append(fest)

        fests = np.array(fests)
        minfocal = fests.min()
        maxfocal = fests.max()

        focalpadding = 0.1 * minfocal

        jpgs_focalbar = list()
        for idx, jpg in enumerate(jpgs):
            fest = smooth_width(fests, idx, width=3)
            jpgs_focalbar.append(draw_focalbar(Image.open(jpg), fest, minfocal=int(minfocal - focalpadding), maxfocal=int(maxfocal + focalpadding)))

        output_folder = os.path.join(output_dollyzoom, dollyzoomvideo)
        os.makedirs(output_folder, exist_ok=True)
        for idx, jpgimgage in enumerate(jpgs_focalbar):
            jpg_path = os.path.join(output_folder, '{}.png'.format(str(idx)))
            jpgimgage.save(jpg_path)

    return

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