import os, sys, inspect, tqdm, copy, pickle
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
sys.path.insert(0, project_root)

import numpy as np
import torch
from tabulate import tabulate
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
from WildCamera.evaluation.evaluate_pose import compute_pose_error, pose_auc, estimate_pose, compute_relative_pose
from WildCamera.newcrfs.newcrf_incidencefield import NEWCRFIF

class ScanNetBenchmark:
    def __init__(self, data_root="data/scannet") -> None:
        self.data_root = data_root

    def benchmark(self, camera_calibrator, use_calibrated_intrinsic=False):
        with torch.no_grad():
            data_root = self.data_root
            tmp = np.load(os.path.join(data_root, "test.npz"))
            pairs, rel_pose = tmp["name"], tmp["rel_pose"]
            tot_e_t, tot_e_R, tot_e_pose = [], [], []
            np.random.seed(0)
            pair_inds = np.random.choice(
                range(len(pairs)), size=len(pairs), replace=False
            )
            for cnt, pairind in enumerate(tqdm(pair_inds)):
                scene = pairs[pairind]
                scene_name = f"scene0{scene[0]}_00"
                im1_path = os.path.join(
                        self.data_root,
                        "scans_test",
                        scene_name,
                        "color",
                        f"{scene[2]}.jpg",
                    )
                im1 = Image.open(im1_path)
                im2_path = os.path.join(
                        self.data_root,
                        "scans_test",
                        scene_name,
                        "color",
                        f"{scene[3]}.jpg",
                    )
                im2 = Image.open(im2_path)
                T_gt = rel_pose[pairind].reshape(3, 4)
                R, t = T_gt[:3, :3], T_gt[:3, 3]
                K = np.stack(
                    [
                        np.array([float(i) for i in r.split()])
                        for r in open(
                            os.path.join(
                                self.data_root,
                                "scans_test",
                                scene_name,
                                "intrinsic",
                                "intrinsic_color.txt",
                            ),
                            "r",
                        )
                        .read()
                        .split("\n")
                        if r
                    ]
                )
                w1, h1 = im1.size
                w2, h2 = im2.size

                if not use_calibrated_intrinsic:
                    K1 = K.copy()
                    K2 = K.copy()
                else:
                    K1_, _ = camera_calibrator.inference(im1, wtassumption=True)
                    K1_ = K1_.astype(np.float64)
                    K2_, _ = camera_calibrator.inference(im2, wtassumption=True)
                    K2_ = K2_.astype(np.float64)

                    K1 = np.eye(4)
                    K1[0:3, 0:3] = K1_

                    K2 = np.eye(4)
                    K2[0:3, 0:3] = K2_

                scale1 = 480 / min(w1, h1)
                scale2 = 480 / min(w2, h2)
                w1, h1 = scale1 * w1, scale1 * h1
                w2, h2 = scale2 * w2, scale2 * h2
                K1 = K1 * scale1
                K2 = K2 * scale2

                corres_fold = os.path.join(self.data_root, 'ScanNetCorrespondence')
                sv_path = os.path.join(corres_fold, '{}.pkl'.format(str(pairind)))
                with open(sv_path, 'rb') as f:
                    sparse_matches = pickle.load(f)

                kpts1 = sparse_matches[:, :2]
                kpts2 = sparse_matches[:, 2:]

                for _ in range(5):
                    shuffling = np.random.permutation(np.arange(len(kpts1)))
                    kpts1 = kpts1[shuffling]
                    kpts2 = kpts2[shuffling]
                    try:
                        norm_threshold = 0.8 / (
                        np.mean(np.abs(K1[:2, :2])) + np.mean(np.abs(K2[:2, :2])))
                        R_est, t_est, mask = estimate_pose(
                            kpts1,
                            kpts2,
                            K1,
                            K2,
                            norm_threshold,
                            conf=0.99999,
                        )
                        T1_to_2_est = np.concatenate((R_est, t_est), axis=-1)  #
                        e_t, e_R = compute_pose_error(T1_to_2_est, R, t)
                        e_pose = max(e_t, e_R)
                    except Exception as _:
                        e_t, e_R = 90, 90
                        e_pose = max(e_t, e_R)
                    tot_e_t.append(e_t)
                    tot_e_R.append(e_R)
                    tot_e_pose.append(e_pose)

            tot_e_pose = np.array(tot_e_pose)
            thresholds = [5, 10, 20]
            auc = pose_auc(tot_e_pose, thresholds)
            acc_5 = (tot_e_pose < 5).mean()
            acc_10 = (tot_e_pose < 10).mean()
            acc_15 = (tot_e_pose < 15).mean()
            acc_20 = (tot_e_pose < 20).mean()
            map_5 = acc_5
            map_10 = np.mean([acc_5, acc_10])
            map_20 = np.mean([acc_5, acc_10, acc_15, acc_20])

            result = {
                "auc_5": auc[0],
                "auc_10": auc[1],
                "auc_20": auc[2],
                "map_5": map_5,
                "map_10": map_10,
                "map_20": map_20,
            }

            for k in result.keys():
                result[k] = result[k] * 100

            return result

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument('--load_ckpt', type=str, help='path of ckpt')
    args, _ = parser.parse_known_args()

    scannet_benchmark = ScanNetBenchmark(args.data_path)

    args.load_ckpt = os.path.join(project_root, 'model_zoo', 'Release', 'wild_camera_all.pth')
    camera_calibrator = NEWCRFIF(version='large07', pretrained=None)
    camera_calibrator.load_state_dict(torch.load(args.load_ckpt, map_location="cpu"), strict=True)
    camera_calibrator.eval()
    camera_calibrator.cuda()

    result = scannet_benchmark.benchmark(camera_calibrator, use_calibrated_intrinsic=True)
    print(tabulate(result.items(), headers=['Metric', 'Scores'], tablefmt='fancy_grid', floatfmt=".2f", numalign="left"))