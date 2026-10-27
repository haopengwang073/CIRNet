"""Strict loading and image-size handling shared by the public entry points."""

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from core import CIRNet


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(value="auto"):
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if value == "auto"
        else torch.device(value)
    )
    if device.type not in ("cpu", "cuda"):
        raise ValueError("supported devices are cpu and cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def load_model(checkpoint, device="auto", iterations=4):
    """Load a released plain state_dict strictly, without downloading backbone weights."""
    model = CIRNet(iterations=iterations)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError(
            "checkpoint keys do not match the released CIRNet architecture"
        )
    for key, value in state.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != expected[key].shape
            or value.dtype != expected[key].dtype
        ):
            raise ValueError(f"incompatible checkpoint tensor: {key}")
    model.load_state_dict(state, strict=True)
    return model.to(resolve_device(device)).eval()


class InputPadder:
    """Symmetric replicate padding to multiples of 32, matching evaluation preprocessing."""

    def __init__(self, shape):
        height, width = shape[-2:]
        ph, pw = -height % 32, -width % 32
        self.padding = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
        self.height, self.width = height, width

    def pad(self, *images):
        return [F.pad(image, self.padding, mode="replicate") for image in images]

    def unpad(self, image):
        left, _, top, _ = self.padding
        return image[..., top : top + self.height, left : left + self.width]


@torch.inference_mode()
def predict(model, left, right):
    """Predict at the original resolution from equal-sized RGB CHW images."""
    if (
        left.ndim != 3
        or left.shape != right.shape
        or left.shape[0] != 3
        or min(left.shape[-2:]) < 1
    ):
        raise ValueError("left and right must have equal non-empty RGB CHW shapes")
    device = next(model.parameters()).device
    left, right = left[None].to(device=device, dtype=torch.float32), right[None].to(
        device=device, dtype=torch.float32
    )
    padder = InputPadder(left.shape)
    result = padder.unpad(model(*padder.pad(left, right)))[0, 0].float().cpu()
    if not torch.isfinite(result).all():
        raise ValueError("prediction contains non-finite disparities")
    return result
