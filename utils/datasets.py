from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .io import read_image, read_pfm


@dataclass(frozen=True)
class StereoSample:
    name: str
    left: Path
    right: Path
    disparity: Path
    non_occluded: Path | None = None


MASK_RULES = {
    "eth3d": "0 < disparity < 1024; mask0nocc == 255",
    "middlebury": "0 < disparity < 1024; mask0nocc == 255",
    "kitti2012": "0 < disparity < 192; disp_occ ground truth",
    "kitti2015": "0 < disparity < 192; disp_occ_0 ground truth",
}


def discover_samples(dataset, root, resolution="Q"):
    """Root is ETH3D/, Middlebury/, KITTI_2012/, or KITTI_2015/ respectively."""
    root = Path(root)
    if dataset not in MASK_RULES:
        raise ValueError(f"unsupported dataset: {dataset}")
    if resolution not in ("F", "H", "Q"):
        raise ValueError("Middlebury resolution must be F, H, or Q")
    samples = []
    if dataset == "eth3d":
        for scene in sorted((root / "two_view_training").glob("*")):
            if scene.is_dir():
                gt = root / "two_view_training_gt" / scene.name
                samples.append(
                    StereoSample(
                        scene.name,
                        scene / "im0.png",
                        scene / "im1.png",
                        gt / "disp0GT.pfm",
                        gt / "mask0nocc.png",
                    )
                )
    elif dataset == "middlebury":
        for scene in sorted((root / "MiddEval3" / f"training{resolution}").glob("*")):
            if scene.is_dir():
                samples.append(
                    StereoSample(
                        scene.name,
                        scene / "im0.png",
                        scene / "im1.png",
                        scene / "disp0GT.pfm",
                        scene / "mask0nocc.png",
                    )
                )
    else:
        left_dir, right_dir, gt_dir = (
            ("colored_0", "colored_1", "disp_occ")
            if dataset == "kitti2012"
            else ("image_2", "image_3", "disp_occ_0")
        )
        training = root / "training"
        # Compare all three ID sets so a missing left image cannot silently shrink evaluation.
        groups = [
            {p.name: p for p in (training / directory).glob("*_10.png")}
            for directory in (left_dir, right_dir, gt_dir)
        ]
        if not (groups[0].keys() == groups[1].keys() == groups[2].keys()):
            raise ValueError(
                f"left/right/ground-truth sample IDs differ under {training}"
            )
        samples = [
            StereoSample(name.removesuffix(".png"), *(group[name] for group in groups))
            for name in sorted(groups[0])
        ]
    if not samples:
        raise ValueError(f"no {dataset} samples found under {root}")
    for sample in samples:
        for path in (sample.left, sample.right, sample.disparity, sample.non_occluded):
            if path is not None and not path.is_file():
                raise FileNotFoundError(path)
    return samples


def load_sample(sample, dataset):
    left, right = read_image(sample.left), read_image(sample.right)
    if dataset.startswith("kitti"):
        with Image.open(sample.disparity) as image:
            disparity = np.array(image, dtype=np.float32) / 256.0
    else:
        disparity = read_pfm(sample.disparity)
    valid = (disparity > 0) & (disparity < 1024)
    if dataset.startswith("kitti"):
        valid &= np.abs(disparity) < 192
    if sample.non_occluded is not None:
        with Image.open(sample.non_occluded) as image:
            non_occluded = np.array(image.convert("L")) == 255
        if non_occluded.shape != disparity.shape:
            raise ValueError(f"mask shape mismatch: {sample.name}")
        valid &= non_occluded
    if left.shape != right.shape or tuple(left.shape[-2:]) != disparity.shape:
        raise ValueError(f"image/disparity shape mismatch: {sample.name}")
    if not valid.any():
        raise ValueError(f"no valid evaluation pixels: {sample.name}")
    return left, right, torch.from_numpy(disparity), torch.from_numpy(valid)
