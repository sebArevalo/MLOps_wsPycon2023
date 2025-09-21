# train.py
import os
import argparse
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

import wandb
from facenet_pytorch import InceptionResnetV1

from src import get_model  # <- registry factory


# ----------------------------
# CLI
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, default="testing console")
parser.add_argument('--model', type=str, default='pairhead_mlp_bn',
                    help='Model registered in src: linear, pairhead_mlp_bn, pairhead_mlp_res')
parser.add_argument('--init_model_artifact', type=str, default=None,
                    help='Initialized model artifact to start from, e.g. "pairhead_mlp_bn:latest". '
                         'Defaults to <model>:latest if omitted.')
parser.add_argument('--dataset_artifact', type=str, default='lfw-pairs-raw:latest',
                    help='W&B dataset artifact with training.pt/validation.pt/test.pt')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_log_interval', type=int, default=25)
args = parser.parse_args()

if args.init_model_artifact is None:
    args.init_model_artifact = f"{args.model}:latest"

print(f"IdExecution: {args.IdExecution}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ----------------------------
# Data reading (raw .pt pairs)
# ----------------------------
def read_split(data_dir: str, split: str) -> TensorDataset:
    """Reads <split>.pt saved as (x, y). For LFW pairs: x shape ~ (N, 2, H, W[, C]), y in {0,1}."""
    x, y = torch.load(os.path.join(data_dir, f"{split}.pt"))
    return TensorDataset(x, y)


# ----------------------------
# Embeddings + pair features
# ----------------------------
def build_embedder() -> Tuple[InceptionResnetV1, Compose]:
    """Frozen FaceNet embedder + transforms to 3x160x160 tensor in [-1,1]."""
    embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    for p in embedder.parameters():
        p.requires_grad = False
    tx = Compose([
        Resize((160, 160)),
        ToTensor(),                         # HWC uint8 -> CHW float in [0,1]
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # -> [-1, 1]
    ])
    return embedder, tx


def _to_rgb(img: torch.Tensor) -> torch.Tensor:
    """img: (H, W) or (H, W, 1) or (H, W, 3) uint8 -> (H, W, 3) uint8."""
    if img.ndim == 2:
        img = img.unsqueeze(-1).repeat(1, 1, 3)
    elif img.shape[-1] == 1:
        img = img.repeat(1, 1, 3)
    return img


@torch.no_grad()
def embed_batch(embedder, tx, pair_imgs: torch.Tensor) -> torch.Tensor:
    """
    pair_imgs: (B, 2, H, W[, C]) uint8
    returns embeddings: (B, 2, 512) float32, L2-normalized
    """
    # ensure HWC last
    if pair_imgs.ndim == 5:  # (B, 2, H, W, C)
        pass
    elif pair_imgs.ndim == 4:  # (B, 2, H, W) -> add C=1
        pair_imgs = pair_imgs.unsqueeze(-1)
    else:
        raise ValueError("Unexpected pair_imgs shape")

    B = pair_imgs.shape[0]
    e_list = []
    for i in range(2):  # two faces per pair
        imgs = pair_imgs[:, i]  # (B, H, W, C/1)
        # convert to RGB HWC uint8 per image -> CHW float via tx
        # build a batch
        batch_tensors = []
        for b in range(B):
            rgb = _to_rgb(imgs[b].cpu())
            t = tx(rgb)  # 3x160x160
            batch_tensors.append(t)
        batch = torch.stack(batch_tensors, dim=0).to(device)
        e = embedder(batch)            # (B, 512)
        e = F.normalize(e, p=2, dim=1) # L2 norm
        e_list.append(e)
    E1, E2 = e_list[0], e_list[1]      # (B,512) each
    return torch.stack([E1, E2], dim=1)  # (B, 2, 512)


def make_pair_features(E: torch.Tensor) -> torch.Tensor:
    """
    E: (B, 2, D) -> features (B, 1026) for D=512
    x = [|e1-e2|, e1*e2, cos, 1-cos]
    """
    e1, e2 = E[:, 0, :], E[:, 1, :]
    absdiff = (e1 - e2).abs()
    hadamard = e1 * e2
    cos = F.cosine_similarity(e1, e2).unsqueeze(-1)
    one_minus = 1.0 - cos
    return torch.cat([absdiff, hadamard, cos, one_minus], dim=-1)  # (B, 1026)


# ----------------------------
# Train / Eval
# ----------------------------
def train_one_epoch(model, loader, optimizer, example_ct, epoch, log_interval):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    for bidx, (xpair, y) in enumerate(loader):
        y = y.float().to(device)  # (B,)
        # compute pair features on the fly
        E = embed_batch(embedder, tx, xpair)       # (B, 2, 512)
        feats = make_pair_features(E)              # (B, 1026)
        optimizer.zero_grad()
        logits = model(feats)                      # (B, 1)
        logits = logits.squeeze(1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        example_ct += xpair.size(0)

        if bidx % log_interval == 0:
            wandb.log({"epoch": epoch, "train/loss": float(loss)}, step=example_ct)
            print(f"Train Epoch {epoch} [{bidx * len(y)}/{len(loader.dataset)}] "
                  f"({bidx/len(loader):.0%})  Loss: {loss.item():.6f}")
    return example_ct


@torch.no_grad()
def evaluate(model, loader, split_name="validation", example_ct=0, epoch=0):
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction='sum')

    total_loss = 0.0
    y_true_all, y_score_all = [], []

    for xpair, y in loader:
        y = y.float().to(device)
        E = embed_batch(embedder, tx, xpair)
        feats = make_pair_features(E)
        logits = model(feats).squeeze(1)
        loss = criterion(logits, y)
        total_loss += loss.item()

        y_true_all.append(y.detach().cpu())
        y_score_all.append(torch.sigmoid(logits).detach().cpu())

    y_true = torch.cat(y_true_all).numpy()
    y_score = torch.cat(y_score_all).numpy()
    avg_loss = total_loss / len(loader.dataset)

    # Metrics
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
        roc_auc = float(roc_auc_score(y_true, y_score))
        pr_auc = float(average_precision_score(y_true, y_score))

        # Best threshold by accuracy
        fpr, tpr, thr = roc_curve(y_true, y_score)
        accs = []
        for th in thr:
            preds = (y_score >= th).astype('int64')
            acc = (preds == y_true).mean()
            accs.append(acc)
        best_acc = float(max(accs)) if accs else 0.0
    except Exception as e:
        print("Metric computation issue:", e)
        roc_auc, pr_auc, best_acc = 0.0, 0.0, 0.0

    wandb.log({
        "epoch": epoch,
        f"{split_name}/loss": avg_loss,
        f"{split_name}/roc_auc": roc_auc,
        f"{split_name}/pr_auc": pr_auc,
        f"{split_name}/best_acc": best_acc,
    }, step=example_ct)

    print(f"[{split_name}] loss={avg_loss:.4f} roc_auc={roc_auc:.4f} "
          f"pr_auc={pr_auc:.4f} best_acc={best_acc:.4f}")

    return avg_loss, roc_auc, pr_auc, best_acc


def train_and_log(config, experiment_id='00'):
    with wandb.init(
        project="MLOps-Pycon2023",
        name=f"Train {args.model} ExecId-{args.IdExecution} Exp-{experiment_id}",
        job_type="train-model",
        config=config
    ) as run:
        cfg = wandb.config

        # ---- Data
        data_art = run.use_artifact(args.dataset_artifact)
        data_dir = data_art.download()
        train_ds = read_split(data_dir, "training")
        val_ds   = read_split(data_dir, "validation")

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        # ---- Model (initialized weights)
        init_art = run.use_artifact(args.init_model_artifact)
        init_dir = init_art.download()
        # file is "initialized_model_<model>.pth"
        init_fname = [f for f in os.listdir(init_dir) if f.startswith("initialized_model_") and f.endswith(".pth")][0]
        init_path = os.path.join(init_dir, init_fname)
        model_config = init_art.metadata or {}
        # ensure defaults if missing
        model_config.setdefault("input_shape", 1026)
        model_config.setdefault("num_classes", 1)

        model = get_model(args.model, **model_config).to(device)
        model.load_state_dict(torch.load(init_path, map_location=device))
        wandb.watch(model, log="all", log_freq=100)

        # ---- Optimizer
        optimizer_cls = getattr(torch.optim, cfg.optimizer)
        optimizer = optimizer_cls(model.parameters(), lr=cfg.lr)

        # ---- Train
        example_ct = 0
        best_val = -1.0
        best_ckpt = "trained_model_best.pth"

        for epoch in range(cfg.epochs):
            example_ct = train_one_epoch(model, train_loader, optimizer, example_ct, epoch, cfg.batch_log_interval)
            _, roc_auc, pr_auc, best_acc = evaluate(model, val_loader, "validation", example_ct, epoch)
            score = roc_auc  # pick your selection metric
            if score > best_val:
                best_val = score
                torch.save(model.state_dict(), best_ckpt)

        # ---- Log trained artifact
        trained_art = wandb.Artifact(
            "trained-model", type="model",
            description=f"Trained {args.model} on LFW pairs",
            metadata=dict(model_config)
        )
        trained_art.add_file(best_ckpt)
        run.log_artifact(trained_art)

        return best_ckpt, model_config


def evaluate_and_log(model_config, experiment_id='00'):
    with wandb.init(
        project="MLOps-Pycon2023",
        name=f"Eval {args.model} ExecId-{args.IdExecution} Exp-{experiment_id}",
        job_type="eval-model",
        config=model_config
    ) as run:
        # Data
        data_art = run.use_artifact(args.dataset_artifact)
        data_dir = data_art.download()
        test_ds = read_split(data_dir, "test")
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

        # Trained model
        trained_art = run.use_artifact("trained-model:latest")
        mdir = trained_art.download()
        mpath = os.path.join(mdir, "trained_model_best.pth")
        model = get_model(args.model, **(model_config or {"input_shape": 1026, "num_classes": 1})).to(device)
        model.load_state_dict(torch.load(mpath, map_location=device))

        # Final eval
        evaluate(model, test_loader, "test", example_ct=0, epoch=0)


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # global (frozen) embedder + transforms
    embedder, tx = build_embedder()

    train_cfg = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "batch_log_interval": args.batch_log_interval,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "model": args.model,
        "dataset_artifact": args.dataset_artifact,
        "init_model_artifact": args.init_model_artifact,
    }

    best_ckpt, model_cfg = train_and_log(train_cfg, experiment_id='A')
    evaluate_and_log(model_cfg, experiment_id='A')
