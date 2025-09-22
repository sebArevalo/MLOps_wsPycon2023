# src/data/load.py
import argparse
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
import wandb
from sklearn.datasets import fetch_lfw_pairs

# --------------------
# Argsss
# --------------------
parser = argparse.ArgumentParser()
parser.add_argument("--IdExecution", type=str, default="console")
parser.add_argument("--train_size", type=float, default=0.8,
                    help="Proportion of TRAIN subset used for train vs val.")
parser.add_argument("--color", type=str, default="false", choices=["true", "false"],
                    help="Load color images (true) or grayscale (false).")
parser.add_argument("--resize", type=float, default=1.0,
                    help="Resize factor for LFW pairs (1.0 keeps 62x47).")
parser.add_argument("--data_home", type=str, default="./data",
                    help="Where to cache/download LFW.")
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
print("Using dataset: LFW pairs (scikit-learn)")

# --------------------
# Helpers
# --------------------
def _split_train_val(x: torch.Tensor, y: torch.Tensor,
                     train_size: float = 0.8, seed: int = 42):
    n = x.shape[0]
    n_train = int(n * train_size)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    tr, va = perm[:n_train], perm[n_train:]
    return x[tr], y[tr], x[va], y[va]

def _to_tensor(arr) -> torch.Tensor:
    if isinstance(arr, torch.Tensor):
        return arr
    return torch.from_numpy(np.asarray(arr))

# --------------------
# Load LFW pairs
# --------------------
def load_lfw_pairs(train_size=0.8, color=False, resize=1.0, data_home="./data"):
    """
    Returns [training_set, validation_set, test_set]
    where each split is a TensorDataset of:
        x: uint8 tensor (N, 2, H, W[, C])
        y: long tensor (N,)
    """
    kwargs = dict(
        data_home=data_home,
        color=color,
        funneled=True,
        resize=resize,
        download_if_missing=True,
    )

    train_b = fetch_lfw_pairs(subset="train", **kwargs)
    test_b  = fetch_lfw_pairs(subset="test",  **kwargs)

    # sklearn versions: use .pairs if present, else .images
    x_train_np = getattr(train_b, "pairs", getattr(train_b, "images"))
    x_test_np  = getattr(test_b,  "pairs", getattr(test_b,  "images"))
    y_train_np = train_b.target
    y_test_np  = test_b.target

    # to torch (keep images as uint8; labels long)
    x_train = _to_tensor(x_train_np)              # (N, 2, H, W[, C]) uint8
    x_test  = _to_tensor(x_test_np)
    y_train = _to_tensor(y_train_np).to(torch.long).view(-1)
    y_test  = _to_tensor(y_test_np).to(torch.long).view(-1)

    # val split from official train
    x_tr, y_tr, x_val, y_val = _split_train_val(x_train, y_train, train_size=train_size)

    training_set   = TensorDataset(x_tr,  y_tr)
    validation_set = TensorDataset(x_val, y_val)
    test_set       = TensorDataset(x_test, y_test)
    return [training_set, validation_set, test_set]

# --------------------
# W&B Logging
# --------------------
def load_and_log():
    color_flag = (args.color.lower() == "true")
    with wandb.init(
        project="MLOps-Pycon2023",
        name=f"LFW-PAIRS Raw Split | ExecId-{args.IdExecution}",
        job_type="load-data",
        config={
            "dataset": "lfw_pairs",
            "train_size": args.train_size,
            "color": color_flag,
            "resize": args.resize,
        }
    ) as run:
        datasets = load_lfw_pairs(
            train_size=args.train_size,
            color=color_flag,
            resize=args.resize,
            data_home=args.data_home,
        )
        names = ["training", "validation", "test"]

        shapes_meta = {
            "training_x":   tuple(datasets[0].tensors[0].shape),
            "validation_x": tuple(datasets[1].tensors[0].shape),
            "test_x":       tuple(datasets[2].tensors[0].shape),
        }

        raw = wandb.Artifact(
            "lfw-pairs-raw", type="dataset",
            description="Raw LFW pairs split into train/val/test (val from train)",
            metadata={
                "source": "sklearn.datasets.fetch_lfw_pairs",
                "sizes": [len(ds) for ds in datasets],
                "shapes": shapes_meta,
                "label_meaning": "1 = same person, 0 = different people",
                "color": color_flag,
                "resize": args.resize,
            }
        )

        for name, data in zip(names, datasets):
            with raw.new_file(f"{name}.pt", mode="wb") as f:
                x, y = data.tensors
                torch.save((x, y), f)

        run.log_artifact(raw)
        print("Logged artifact: lfw-pairs-raw")

if __name__ == "__main__":
    load_and_log()
