# Put most Common Functions Here
import cv2
import torch
import numpy as np
from einops.einops import rearrange

def compute_intrinsic_measure(Kest, Kgt, h, w):
    fxest, fyest, bxest, byest = Kest[0, 0], Kest[1, 1], Kest[0, 2], Kest[1, 2]
    fxgt, fygt, bxgt, bygt = Kgt[0, 0], Kgt[1, 1], Kgt[0, 2], Kgt[1, 2]

    error_fx = ((fxest - fxgt) / fxgt).abs().item()
    error_fy = ((fyest - fygt) / fygt).abs().item()

    error_f = max(
        error_fx,
        error_fy,
    )

    error_bx = (bxest - bxgt).abs().item() / w * 2
    error_by = (byest - bygt).abs().item() / h * 2
    error_b = max(
        error_bx,
        error_by
    )

    return error_fx, error_fy, error_f, error_bx, error_by, error_b