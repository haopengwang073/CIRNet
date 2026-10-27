import argparse
from pathlib import Path

from utils.runtime import load_model
from utils.io import read_image, save_disparity
from utils.runtime import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--iterations", type=int, default=4)
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device, args.iterations)
    disparity = predict(model, read_image(args.left), read_image(args.right))

    save_disparity(args.output, disparity.numpy())
    print(f"Saved disparity.npy, disparity.pfm, and disparity.png to {args.output}")


if __name__ == "__main__":
    main()
