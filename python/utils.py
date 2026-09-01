import pickle as pkl
import sys
from pathlib import Path
from typing import Dict, Tuple

from config import DATA_DIR

import numpy as np
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm


def get_dataloader(param: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    valid_size = 0.2
    num_workers = 0

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_dir = Path(param.get("data_dir", DATA_DIR))
    train_data = datasets.CIFAR10(str(data_dir), train=True,
                                  download=True, transform=transform)
    test_data = datasets.CIFAR10(str(data_dir), train=False,
                                 download=True, transform=transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=param["batch_size"],
                                               sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=param["batch_size"],
                                               sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=param["batch_size"],
                                              num_workers=num_workers)

    return train_loader, valid_loader, test_loader


def train(
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        param: Dict,
        step: int = 1,
        device: str = "cpu") -> nn.Module:
    model = model.to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=param["lr"])
    optimizer = torch.optim.SGD(model.parameters(), lr=param["lr"], momentum=0.9, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=param["milestones"], gamma=param["gamma"]
    )

    best_acc = 0.0
    # To avoid breaking up the tqdm bar
    with tqdm(total=param["epochs"], file=sys.stdout) as pbar:

        for i in range(param["epochs"]):
            # Train the model
            model.train()
            loss_batch_train, accuracy_batch_train = [], []

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                # x = torch.mean(x, dim=1, keepdim=True)

                optimizer.zero_grad()
                yhat = model(x)
                loss_train = param["criterion"](yhat, y)
                loss_train.backward()
                optimizer.step()

                loss_batch_train.append(loss_train.item())
                accuracy_batch_train.extend((yhat.argmax(1) == y).cpu().float().tolist())

            if scheduler:
                scheduler.step()

            param["accuracy_train"].append(np.mean(accuracy_batch_train))
            param["loss_train_history"].append(np.mean(loss_batch_train))

            # Evaluation during training:
            # Disable autograd engine (no backpropagation)
            # To reduce memory usage and to speed up computations
            with torch.no_grad():
                # Notify batchnormalization & dropout layers to work in eval mode
                model.eval()
                loss_batch_test, accuracy_batch_test = [], []
                for x, y in valid_loader:
                    x, y = x.to(device), y.to(device)
                    # x = torch.mean(x, dim=1, keepdim=True)
                    yhat = model(x)
                    loss_test = param["criterion"](yhat, y)
                    loss_batch_test.append(loss_test.item())
                    accuracy_batch_test.extend((yhat.argmax(1) == y).cpu().float().tolist())

                test_acc = np.mean(accuracy_batch_test)
                param["accuracy_test"].append(test_acc)
                param["loss_test_history"].append(np.mean(loss_batch_test))

                # save best model
                if best_acc < test_acc:
                    best_acc = test_acc
                    # Save the state_dict
                    dir = Path(".") / param["dir"] / param["training"]
                    dir.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        model.state_dict(), f"{dir}/{param['dataset_name']}_{param['training']}_state_dict.pt")

                    with open(f"{dir}/{param['dataset_name']}_history.pkl", "wb") as f:
                        pkl.dump(param, f)

                    for param_name in model.state_dict():
                        file_path = f"{dir}/{param_name}.txt"
                        data = model.state_dict()[param_name].cpu().numpy().flatten()
                        np.savetxt(file_path, data, fmt="%f", delimiter=",", newline=",")


            if i % step == 0:
                pbar.write(
                    f"Epoch {i:2}: Train loss = {param['loss_train_history'][-1]:.4f} "
                    f"VS Test loss = {param['loss_test_history'][-1]:.4f} - "
                    f"Accuracy train: {param['accuracy_train'][-1]:.4f} "
                    f"VS Accuracy test: {param['accuracy_test'][-1]:.4f}"
                )
                pbar.update(step)

    # for name, param in model.named_parameters():
    #     file_path = f"{dir}/{name}.txt"
    #     # torch.save(param.data, file_path)
    #     np.savetxt(file_path, param.data.flatten(), fmt="%f", delimiter=",", newline=",")

    torch.cuda.empty_cache()

    return model


def torch_inference(
        model: nn.Module,
        data: DataLoader,
        device: str = "cpu",
        verbose: bool = False,
) -> float:
    num_correct = 0
    num_examples = 0
    model = model.to(device)

    with torch.no_grad():
        model.eval()
        for x, y in tqdm(data, disable=verbose is False):
            x, y = x.to(device), y.to(device)
            # x = torch.mean(x, dim=1, keepdim=True)
            yhat = model(x)
            _, pred = torch.max(yhat, 1)
            num_correct += pred.eq(y.data.view_as(pred)).sum()
            num_examples += pred.size(0)

    return float(num_correct) / float(num_examples)
