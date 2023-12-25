from __future__ import print_function, division
import os, sys, inspect, time, copy, warnings
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
sys.path.insert(0, project_root)
warnings.filterwarnings('ignore')

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset
from torchvision import transforms

import argparse
import numpy as np
from loguru import logger
from einops import rearrange
from pprint import pprint

from WildCamera.newcrfs.newcrf_incidencefield import NEWCRFIF
from WildCamera.datasets.IncdDataset import IncdDataset
from tools.tools import to_cuda, intrinsic2incidence, resample_rgb, DistributedSamplerNoEvenlyDivisible, IncidenceLoss
from tools.visualization import tensor2disp, tensor2rgb

parser = argparse.ArgumentParser(description='NeWCRFs PyTorch implementation.', fromfile_prefix_chars='@')

parser.add_argument('--experiment_name',           type=str,   help='experiment name', default='multi-modality newcrfs')
parser.add_argument('--experiment_set',            type=str,   choices=['gsv', 'in_the_wild'], required=True)
parser.add_argument('--saving_location',           type=str,   help='saving location', default=None)
parser.add_argument('--encoder',                   type=str,   help='type of encoder, base07, large07', default='large07')
parser.add_argument('--pretrain',                  type=str,   help='path of pretrained encoder', default='model_zoo/swin_transformer/swin_large_patch4_window7_224_22k.pth')
parser.add_argument('--load_ckpt',                 type=str,   help='path of ckpt', default=None)
parser.add_argument('--evaluation_only',           action="store_true")
parser.add_argument('--l1_th',                     type=int,  help='RANSAC threshold', default=0.02)

# Dataset
parser.add_argument('--data_path',                 type=str,   help='path to the data', default='data/MonoCalib')
parser.add_argument('--input_height',              type=int,   help='input height', default=480)
parser.add_argument('--input_width',               type=int,   help='input width',  default=640)

# Training
parser.add_argument('--batch_size',                type=int,   help='batch size', default=16)
parser.add_argument('--loss',                      type=str,   default='cosine', help='You can also choees l1 loss')
parser.add_argument('--steps_per_epoch',           type=int,   help='frequency for evaluation', default=1000)
parser.add_argument('--termination_epoch',         type=int,   help='epoch to stop training', default=25)

# Training Misc
parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--train_workers',             type=int,   help='dataloader workers', default=32)
parser.add_argument('--eval_workers',              type=int,   help='dataloader workers', default=2)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-5)
parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)

# Augmentation
parser.add_argument('--dataset_favour_long',       type=float, default=0.1, help='whether in training will see more samples from large dataset')
parser.add_argument('--augscale',                  type=float, default=2.0, help='The scale of Augmentation')
parser.add_argument('--no_change_prob',            type=float, default=0.1, help='The probability of seeing original image')
parser.add_argument('--coloraugmentation',         action="store_true")
parser.add_argument('--coloraugmentation_scale',   type=float, default=0.0)

# Multi-gpu training
parser.add_argument('--gpu',                       type=int,  help='GPU id to use.', default=None)
parser.add_argument('--dist_url',                  type=str,  help='url used to set up distributed training', default='tcp://127.0.0.1:1235')
parser.add_argument('--dist_backend',              type=str,  help='distributed backend', default='nccl')

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    group = dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.gpu)

    # NeWCRFs model
    model = NEWCRFIF(version=args.encoder, pretrained=args.pretrain)
    model.train()

    if args.load_ckpt is not None:
        model.load_state_dict(torch.load(args.load_ckpt, map_location="cpu"), strict=True)
        model.eval()
        logger.info("Load Model from %s" % args.load_ckpt)

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.train_workers = int(args.train_workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
    else:
        model = torch.nn.DataParallel(model)
        model.cuda()

    if args.distributed:
        print("== Model Initialized on GPU: {}".format(args.gpu))
    else:
        print("== Model Initialized")

    # Training parameters
    optimizer = torch.optim.Adam([{'params': model.module.parameters()}], lr=args.learning_rate)

    cudnn.benchmark = True

    incidence_criterion = IncidenceLoss(loss=args.loss)

    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.1 * args.learning_rate

    steps_per_epoch = args.steps_per_epoch
    num_total_steps = 250000
    epoch = int(num_total_steps / steps_per_epoch)

    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

    if args.gpu == 0:
        checkpoint_dir = os.path.join(args.saving_location, 'model_zoo', args.experiment_name)
        writer = SummaryWriter(checkpoint_dir, flush_secs=30)
        logger.add(os.path.join(checkpoint_dir, "{}.log".format(args.experiment_name)))
    else:
        writer = None
        logger.remove()
        logger.add(sys.stderr, level="ERROR")

    if args.experiment_set == 'gsv':
        from WildCamera.evaluation.evaluate_fov import EvaluateFov
        evaluator = EvaluateFov()
    elif args.experiment_set == 'in_the_wild':
        from WildCamera.evaluation.evaluate_intrinsic import EvaluateIntrinsic
        evaluator = EvaluateIntrinsic()
    else:
        raise NotImplementedError()

    if args.evaluation_only:
        evaluator.evaluate(
            model,
            args,
            steps=0,
            writer=writer,
            group=group,
            wtassumption=False,
        )
        evaluator.evaluate(
            model,
            args,
            steps=0,
            writer=writer,
            group=group,
            wtassumption=True,
        )
        return

    for n_0 in range(epoch):

        if n_0 > args.termination_epoch:
            break

        incdataset = IncdDataset(
            data_root=args.data_path,
            datasets_included=args.datasets_train,
            ht=args.input_height,
            wt=args.input_width,
            augmentation=True,
            split='train',
            shuffleseed=int(n_0 + 1),
            dataset_favour_long=args.dataset_favour_long,
            augscale=args.augscale,
            no_change_prob=args.no_change_prob,
            coloraugmentation=args.coloraugmentation,
            coloraugmentation_scale=args.coloraugmentation_scale
        )
        sampler = DistributedSamplerNoEvenlyDivisible(incdataset, shuffle=True)
        dataloader = torch.utils.data.DataLoader(
            incdataset,
            sampler=sampler,
            batch_size=int(args.batch_size),
            num_workers=int(args.train_workers)
        )
        assert len(dataloader) > steps_per_epoch
        dataloader = iter(dataloader)
        sampler.set_epoch(n_0)
        for global_step in range(int(n_0 * steps_per_epoch), int((n_0 + 1) * steps_per_epoch)):
            sample_batched = next(dataloader)
            sample_batched = to_cuda(sample_batched)

            rgb, K = sample_batched['rgb'], sample_batched['K']

            model.train()
            incidence = model(rgb)
            optimizer.zero_grad()

            # Loss For Normal
            loss = incidence_criterion(incidence, K)
            loss.backward()

            for param_group in optimizer.param_groups:
                current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                param_group['lr'] = current_lr

            optimizer.step()

            if np.mod(global_step, 1000) == 0 and args.gpu == 0:
                b = 1
                _, _, h, w = rgb.shape

                rgb = inv_normalize(rgb)
                vls1 = tensor2rgb(rgb, viewind=0)

                device = rgb.device
                incidence_gt = intrinsic2incidence(K[0:1], b, h, w, device)
                incidence_gt = rearrange(incidence_gt.squeeze(dim=4), 'b h w d -> b d h w')
                vls3 = tensor2rgb((incidence + 1) / 2, viewind=0)
                vls4 = tensor2rgb((incidence_gt + 1) / 2, viewind=0)

                vls = np.concatenate([np.array(vls1), np.array(vls3), np.array(vls4)], axis=0)
                writer.add_image('visualization', (torch.from_numpy(vls).float() / 255).permute([2, 0, 1]), global_step)

            if writer is not None and args.gpu == 0:
                writer.add_scalar('loss/loss_incidence', loss.item(), global_step)
                if np.mod(global_step, 500) == 0:
                    logger.info('Step %d, Epoch %d, loss %.3f' % (global_step, n_0, loss.item()))

        evaluator.evaluate(
            model,
            args,
            steps=global_step,
            writer=writer,
            group=group,
            wtassumption=False,
        )

        evaluator.evaluate(
            model,
            args,
            steps=global_step,
            writer=writer,
            group=group,
            wtassumption=True,
        )

def main():
    args = parser.parse_args()
    torch.cuda.empty_cache()
    args.world_size = torch.cuda.device_count()
    args.distributed = True
    args.dist_url = 'tcp://127.0.0.1:' + str(np.random.randint(2000, 3000, 1).item())

    if args.saving_location is None:
        args.saving_location = project_root

    if args.experiment_set == 'gsv':
        args.datasets_train = [
            'GSV'
        ]
        args.datasets_eval = [
            'GSV'
        ]
    elif args.experiment_set == 'in_the_wild':
        args.datasets_train = [
            'Nuscenes',
            'KITTI',
            'Cityscapes',
            'NYUv2',
            'ARKitScenes',
            'MegaDepth',
            'SUN3D',
            'MVImgNet',
            'Objectron'
        ]
        args.datasets_eval = [
            'Nuscenes',
            'KITTI',
            'Cityscapes',
            'NYUv2',
            'ARKitScenes',
            'MegaDepth',
            'SUN3D',
            'MVImgNet',
            'Objectron',
            'Waymo',
            'BIWIRGBDID',
            'RGBD',
            'ScanNet',
            'MVS',
            'Scenes11'
        ]

    else:
        raise NotImplementedError()

    pprint(vars(args))
    mp.spawn(main_worker, nprocs=args.world_size, args=(args.world_size, args))

if __name__ == '__main__':
    main()
