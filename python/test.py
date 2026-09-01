import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CHECKPOINT_ROOT
from model import CNNNet
from utils import get_dataloader, torch_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained HE-Net CNN on CIFAR-10")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--filters", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    param = {
        "output_size": 10,
        "batch_size": args.batch_size,
        "training": "fp32",
        "dataset_name": "CIFAR_10",
        "dir": str(CHECKPOINT_ROOT / "CIFAR_10"),
    }

    _, _, test_loader = get_dataloader(param)

    model = CNNNet(10, filters=args.filters)
    ckpt_dir = Path(param["dir"]) / param["training"]
    ckpt_path = ckpt_dir / f"{param['dataset_name']}_{param['training']}_state_dict.pt"
    check_point = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(check_point)
    model.eval()

    acc = torch_inference(model, data=test_loader, device=args.device)
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
