import argparse
import torch
import torchvision
from torch.utils.data import TensorDataset
import wandb

# --------------------
# Args
# --------------------
parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
parser.add_argument('--dataset', type=str, default='fashionmnist',
                    choices=['fashionmnist', 'kmnist', 'cifar10', 'svhn'],
                    help='Which torchvision dataset to use')
parser.add_argument('--train_size', type=float, default=0.8,
                    help='Proportion of training set to use for train vs val')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
print(f"Using dataset: {args.dataset}")

# --------------------
# Helpers
# --------------------
def _split_train_val(x, y, train_size=0.8, generator_seed=42):
    n = x.shape[0]
    n_train = int(n * train_size)
    # Use a stable random split instead of slicing the front of the dataset
    g = torch.Generator().manual_seed(generator_seed)
    perm = torch.randperm(n, generator=g)
    idx_train = perm[:n_train]
    idx_val = perm[n_train:]
    return x[idx_train], y[idx_train], x[idx_val], y[idx_val]

def _to_torch_tensor(arr):
    # Convert numpy arrays to torch tensors without copying when possible
    if isinstance(arr, torch.Tensor):
        return arr
    return torch.from_numpy(arr)

# --------------------
# Loader (supports multiple torchvision datasets)
# --------------------
def load(dataset_name='fashionmnist', train_size=0.8):
    root = './data'

    if dataset_name == 'fashionmnist':
        train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True)
        test  = torchvision.datasets.FashionMNIST(root=root, train=False, download=True)
        x_train, y_train = train.data, train.targets
        x_test,  y_test  = test.data,  test.targets

    elif dataset_name == 'kmnist':
        train = torchvision.datasets.KMNIST(root=root, train=True, download=True)
        test  = torchvision.datasets.KMNIST(root=root, train=False, download=True)
        x_train, y_train = train.data, train.targets
        x_test,  y_test  = test.data,  test.targets

    elif dataset_name == 'cifar10':
        train = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
        test  = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
        # CIFAR10 .data is numpy (N, 32, 32, 3); targets is a list
        x_train, y_train = _to_torch_tensor(train.data), _to_torch_tensor(torch.tensor(train.targets))
        x_test,  y_test  = _to_torch_tensor(test.data),  _to_torch_tensor(torch.tensor(test.targets))
        # Reorder to (N, 32, 32, 3) -> we keep HWC as-is since we’re saving raw; model code can permute later.

    elif dataset_name == 'svhn':
        # SVHN uses split="train"/"test", labels in .labels (numpy), data in .data (numpy, HWC)
        train = torchvision.datasets.SVHN(root=root, split='train', download=True)
        test  = torchvision.datasets.SVHN(root=root, split='test',  download=True)
        x_train, y_train = _to_torch_tensor(train.data), _to_torch_tensor(train.labels)
        x_test,  y_test  = _to_torch_tensor(test.data),  _to_torch_tensor(test.labels)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Ensure y are 1D long tensors
    y_train = y_train.to(dtype=torch.long).view(-1)
    y_test  = y_test.to(dtype=torch.long).view(-1)

    # Split train into train/val
    x_tr, y_tr, x_val, y_val = _split_train_val(x_train, y_train, train_size=train_size)

    training_set   = TensorDataset(x_tr,  y_tr)
    validation_set = TensorDataset(x_val, y_val)
    test_set       = TensorDataset(x_test, y_test)
    return [training_set, validation_set, test_set]

# --------------------
# W&B Logging
# --------------------
def load_and_log():
    with wandb.init(
        project="MLOps-Pycon2023",
        name=f"{args.dataset.upper()} Raw Split | ExecId-{args.IdExecution}",
        job_type="load-data",
        config={
            "dataset": args.dataset,
            "train_size": args.train_size
        }
    ) as run:
        datasets = load(dataset_name=args.dataset, train_size=args.train_size)
        names = ["training", "validation", "test"]

        raw_data = wandb.Artifact(
            f"{args.dataset}-raw", type="dataset",
            description=f"raw {args.dataset} dataset, split into train/val/test",
            metadata={
                "source": f"torchvision.datasets.{args.dataset.upper()}",
                "sizes": [len(ds) for ds in datasets],
                "shapes": {
                    "training_x": tuple(datasets[0].tensors[0].shape),
                    "validation_x": tuple(datasets[1].tensors[0].shape),
                    "test_x": tuple(datasets[2].tensors[0].shape),
                }
            })

        for name, data in zip(names, datasets):
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        run.log_artifact(raw_data)

if __name__ == "__main__":
    load_and_log()

