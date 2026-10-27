"""RGB images and floating-point disparity files."""

from pathlib import Path

import numpy as np
from PIL import Image
import torch


def read_image(path):
    """Read RGB CHW float32 in [0, 255]; grayscale is repeated across RGB."""
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1)


def read_pfm(path):
    """Read a single-channel PFM disparity, retaining invalid values for masking."""
    with Path(path).open("rb") as stream:
        if stream.readline().strip() != b"Pf":
            raise ValueError(f"expected single-channel PFM: {path}")
        width, height = map(int, stream.readline().split())
        scale = float(stream.readline())
        if width <= 0 or height <= 0 or not np.isfinite(scale) or scale == 0:
            raise ValueError(f"invalid PFM header: {path}")
        values = np.fromfile(stream, dtype="<f4" if scale < 0 else ">f4")
    if values.size != height * width:
        raise ValueError(f"invalid PFM data size: {path}")
    # Dataset PFMs use scale only to signal endianness, as in the reference reader.
    return np.flipud(values.reshape(height, width)).astype(np.float32).copy()


def write_pfm(path, disparity):
    array = np.asarray(disparity, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("PFM disparity must be a 2D array")
    with Path(path).open("wb") as stream:
        stream.write(f"Pf\n{array.shape[1]} {array.shape[0]}\n-1\n".encode("ascii"))
        stream.write(np.flipud(array).astype("<f4").tobytes())


def save_disparity(directory, disparity):
    """Save lossless NPY/PFM plus a per-image normalized grayscale preview."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    array = np.asarray(disparity, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("disparity must be a finite 2D array")
    np.save(directory / "disparity.npy", array)
    write_pfm(directory / "disparity.pfm", array)
    lo, hi = float(array.min()), float(array.max())
    normalized = (array - lo) / (hi - lo) if hi > lo else np.zeros_like(array)
    Image.fromarray(np.round(normalized * 255).astype(np.uint8)).save(
        directory / "disparity.png"
    )
