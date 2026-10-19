"""Segmentation model factory (video-project analog of project1/seg/model.py).

Deliberately identical to project1's factory: the leakage/unlearning apparatus
never needed to know about the image backbone's internals, only NUM_CLASSES. Same
holds for the video baseline -- DeepLabV3-R50 run per-frame is the B0 baseline;
the temporal propagation module (ConvGRU over a short window) is added in
video_model.py once the per-frame baseline is validated (see POSITIONING.md,
step 3: establish the floor before building on top of it).
"""
import torch
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50

from seg.data import NUM_CLASSES


def get_seg_model(num_classes=NUM_CLASSES, pretrained_backbone=True,
                  device="cuda", arch="deeplabv3r50"):
    if arch == "deeplabv3r50":
        weights_backbone = "IMAGENET1K_V1" if pretrained_backbone else None
        model = deeplabv3_resnet50(weights=None, weights_backbone=weights_backbone,
                                   num_classes=num_classes, aux_loss=True)
    else:
        raise ValueError(f"unknown arch {arch!r}")
    return model.to(device)


def seg_forward(model, x):
    """Return main logits [N,C,H,W]. `x` is a single frame [N,3,H,W] for the
    per-frame B0 baseline; video_model.seg_forward_clip handles clip inputs."""
    return model(x)["out"]
