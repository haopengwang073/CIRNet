import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import timm
import torch

from utils.runtime import load_model
from utils.datasets import MASK_RULES, discover_samples, load_sample
from utils.metrics import image_metrics
from utils.runtime import predict, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=list(MASK_RULES))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--resolution",
        choices=["F", "H", "Q"],
        default="Q",
        help="Middlebury resolution",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON results file")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iterations", type=int, default=4)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = load_model(args.checkpoint, args.device, args.iterations)
    device = next(model.parameters()).device

    samples = discover_samples(args.dataset, args.root, args.resolution)

    results = []
    for index, sample in enumerate(samples, 1):
        left, right, ground_truth, mask = load_sample(sample, args.dataset)
        prediction = predict(model, left, right)
        results.append(
            {
                "sample": sample.name,
                "height": left.shape[-2],
                "width": left.shape[-1],
                "valid_pixels": mask.sum().item(),
                **image_metrics(prediction, ground_truth, mask),
            }
        )
        print(
            f'[{index}/{len(samples)}] {sample.name}: EPE={results[-1]["EPE"]:.6f}',
            flush=True,
        )

    summary = {
        key: float(np.mean([row[key] for row in results]))
        for key in ("EPE", "D1", "Bad1", "Bad2", "Bad3")
    }

    report = {
        "dataset": args.dataset,
        "resolution": args.resolution if args.dataset == "middlebury" else "native",
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": sha256(args.checkpoint),
        "iterations": args.iterations,
        "precision": "float32",
        "mask_rule": MASK_RULES[args.dataset],
        "aggregation": "arithmetic mean of per-image metrics; rates in percent; EPE in pixels",
        "protocol": "local evaluation; not online benchmark scores or TensorRT latency",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "timm": timm.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor()
            ),
            "cudnn_deterministic": True,
            "tf32": False,
        },
        "count": len(results),
        "metrics": summary,
        "images": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
