import argparse
import sys
import warnings
from pathlib import Path

import torch
from torchsummary import summary

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CHECKPOINT_ROOT
from model import CNNNet
from utils import get_dataloader, train, torch_inference

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HE-Net CNN on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--filters", type=int, default=128)
    parser.add_argument("--seed", type=int, default=727)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_name = "CIFAR_10"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device Type: {device}")

    param = {
        "output_size": 10,
        "batch_size": args.batch_size,
        "training": "fp32",
        "dataset_name": dataset_name,
        "criterion": torch.nn.CrossEntropyLoss(),
        "accuracy_test": [],
        "accuracy_train": [],
        "loss_test_history": [],
        "loss_train_history": [],
        "dir": str(CHECKPOINT_ROOT / dataset_name),
        "seed": args.seed,
        "lr": args.lr,
        "epochs": args.epochs,
        "gamma": 0.1,
        "milestones": [0.5 * args.epochs, 0.7 * args.epochs],
    }

    torch.manual_seed(args.seed)
    train_loader, valid_loader, test_loader = get_dataloader(param)

    cnn_net = CNNNet(param["output_size"], filters=args.filters).to(device)
    summary(cnn_net, (3, 32, 32), device=device)

    cnn_net = train(cnn_net, train_loader, valid_loader, param, device=device)
    acc = torch_inference(cnn_net, data=test_loader, device=device)
    print(f"With {dataset_name}: top-1 accuracy = {acc * 100:2.3f}%")


if __name__ == "__main__":
    main()
