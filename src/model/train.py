# src/model/train.py
import os
import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from facenet_pytorch import InceptionResnetV1
import wandb

# import the registry (works when run as a module: python -m src.model.train)
from . import get_model

# --------------------------
# CLI
# --------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--IdExecution", type=str, default="console")
parser.add_argument("--artifact_raw", type=str, default="lfw-pairs-raw:latest")
parser.add_argument("--initialized_model_art", type=str, default=None,
                    help="W&B artifact name of the initialized model to start from. "
                         "If None, we build from config below.")
parser.add_argument("--model_name", type=str, default="pairhead_mlp_bn",
                    help="Registry model name when not loading an initialized artifact.")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--optimizer", type=str, default="Adam")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch_log_interval", type=int, default=25)
args = parser.parse_args()

print(f"IdExecution: {args.IdExecution}\n")

# --------------------------
# Device
# --------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# --------------------------
# Data helpers
# --------------------------
def read_split(dirpath: str, split: str) -> TensorDataset:
    x, y = torch.load(os.path.join(dirpath, f"{split}.pt"))
    return TensorDataset(x, y)

# --------------------------
# FaceNet embedder + pure-Torch preprocessing
# --------------------------
def build_embedder() -> InceptionResnetV1:
    m = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    for p in m.parameters():
        p.requires_grad = False
    return m

def _hwc_any_to_chw_float_m1_1(imgs_hwc: torch.Tensor) -> torch.Tensor:
    """
    imgs_hwc: (B,H,W,C[=1 or 3]) uint8/float on CPU
    returns: (B,3,160,160) float32 on device, normalized to [-1,1]
    """
    if imgs_hwc.shape[-1] == 1:
        imgs_hwc = imgs_hwc.repeat(1, 1, 1, 3)

    x = imgs_hwc
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        x = x.float()
        with torch.no_grad():
            mx = float(x.max().item()) if x.numel() else 1.0
        if mx > 1.5:
            x = x / 255.0
        x.clamp_(0.0, 1.0)

    x = x.permute(0, 3, 1, 2)                              # HWC->CHW
    x = F.interpolate(x, (160, 160), mode="bilinear", align_corners=False)
    x = (x - 0.5) / 0.5                                    # [-1,1]
    return x.to(device)

@torch.no_grad()
def embed_batch(embedder: InceptionResnetV1, xpair: torch.Tensor) -> torch.Tensor:
    """
    xpair: (B,2,H,W[,C]) in uint8 or float
    returns: (B,2,512) embeddings (L2-normed) on CPU
    """
    if xpair.ndim == 4:                 # (B,2,H,W)
        xpair = xpair.unsqueeze(-1)     # -> (B,2,H,W,1)

    imgs1 = xpair[:, 0].contiguous().cpu()   # (B,H,W,C)
    imgs2 = xpair[:, 1].contiguous().cpu()

    t1 = _hwc_any_to_chw_float_m1_1(imgs1)
    t2 = _hwc_any_to_chw_float_m1_1(imgs2)

    e1 = embedder(t1)                          # (B,512)
    e2 = embedder(t2)
    e1 = F.normalize(e1, p=2, dim=1).cpu()
    e2 = F.normalize(e2, p=2, dim=1).cpu()
    return torch.stack([e1, e2], dim=1)        # (B,2,512)

def make_pair_features(E: torch.Tensor) -> torch.Tensor:
    """
    E: (B,2,512) -> (B,1026) features: [|e1-e2|, e1*e2, cos, 1-cos]
    """
    e1, e2 = E[:, 0, :], E[:, 1, :]
    absdiff = (e1 - e2).abs()
    hadamard = e1 * e2
    cos = F.cosine_similarity(e1, e2).unsqueeze(-1)
    one_minus = 1.0 - cos
    return torch.cat([absdiff, hadamard, cos, one_minus], dim=-1).float()

# --------------------------
# Training / Eval
# --------------------------
@dataclass
class TrainCfg:
    batch_size: int
    epochs: int
    batch_log_interval: int
    optimizer: str
    lr: float

def train_one_epoch(model, loader, optimizer, example_ct, epoch, log_interval, embedder):
    model.train()
    for batch_idx, (xpair, y) in enumerate(loader):
        with torch.no_grad():
            E = embed_batch(embedder, xpair)        # (B,2,512)
            feats = make_pair_features(E)           # (B,1026)
        feats, y = feats.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(feats).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits.float(), y.float())
        loss.backward()
        optimizer.step()

        example_ct += len(y)
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx*len(y)}/{len(loader.dataset)} "
                  f"({batch_idx/len(loader):.0%})]\tLoss: {loss.item():.6f}")
            wandb.log({"epoch": epoch, "train/loss": float(loss)}, step=example_ct)
    return example_ct

@torch.no_grad()
def evaluate_epoch(model, loader, embedder):
    model.eval()
    total, correct = 0, 0
    total_loss = 0.0
    for xpair, y in loader:
        E = embed_batch(embedder, xpair)
        feats = make_pair_features(E).to(device)
        y = y.to(device)

        logits = model(feats).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits.float(), y.float(), reduction="sum")
        total_loss += float(loss.item())

        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += int((preds == y).sum())
        total += y.numel()

    avg_loss = total_loss / total
    acc = correct / total if total else 0.0
    return avg_loss, acc

# --------------------------
# Orchestration
# --------------------------
def train_and_log(cfg: TrainCfg, experiment_id: str = "A"):
    with wandb.init(
        project="MLOps-Pycon2023",
        name=f"Train {args.model_name} ExecId-{args.IdExecution} Exp-{experiment_id}",
        job_type="train-model",
        config=vars(cfg),
    ) as run:
        # 1) raw pairs artifact
        raw_art = run.use_artifact(args.artifact_raw)
        raw_dir = raw_art.download()

        train_set = read_split(raw_dir, "training")
        valid_set = read_split(raw_dir, "validation")

        train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=cfg.batch_size, shuffle=False)

        # 2) model: from initialized artifact or build fresh from registry
        if args.initialized_model_art:
            m_art = run.use_artifact(args.initialized_model_art)
            m_dir = m_art.download()
            m_path = os.path.join(m_dir, next(p for p in os.listdir(m_dir) if p.endswith(".pth")))
            m_cfg = m_art.metadata or {}
            model = get_model(m_cfg.get("model_name", args.model_name), **m_cfg).to(device)
            model.load_state_dict(torch.load(m_path, map_location=device))
        else:
            # default config if no artifact provided
            m_cfg = {"input_shape": 1026, "hidden_layer_1": 256, "hidden_layer_2": 128, "num_classes": 1}
            model = get_model(args.model_name, **m_cfg).to(device)

        # 3) optimizer + embedder
        optimizer = getattr(torch.optim, cfg.optimizer)(model.parameters(), lr=cfg.lr)
        embedder = build_embedder()

        # 4) train
        example_ct = 0
        best_val = float("inf")
        best_path = "trained_model.pth"
        for epoch in range(cfg.epochs):
            example_ct = train_one_epoch(model, train_loader, optimizer, example_ct, epoch,
                                         cfg.batch_log_interval, embedder)
            val_loss, val_acc = evaluate_epoch(model, valid_loader, embedder)
            print(f"[val] epoch {epoch} loss={val_loss:.4f} acc={val_acc:.4f}")
            wandb.log({"epoch": epoch, "validation/loss": val_loss, "validation/accuracy": val_acc},
                      step=example_ct)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), best_path)

        # 5) log trained artifact
        art = wandb.Artifact(
            "trained-model", type="model",
            description="Pair head trained on LFW embeddings",
            metadata={"model_name": args.model_name, **(m_cfg or {})},
        )
        art.add_file(best_path)
        wandb.save(best_path)
        run.log_artifact(art)

        return best_path, m_cfg

def evaluate_and_log(trained_artifact: str, experiment_id: str = "A"):
    with wandb.init(
        project="MLOps-Pycon2023",
        name=f"Eval {args.model_name} ExecId-{args.IdExecution} Exp-{experiment_id}",
        job_type="eval-model",
    ) as run:
        raw_art = run.use_artifact(args.artifact_raw)
        raw_dir = raw_art.download()
        test_set = read_split(raw_dir, "test")
        test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

        m_art = run.use_artifact(trained_artifact)
        m_dir = m_art.download()
        m_path = os.path.join(m_dir, next(p for p in os.listdir(m_dir) if p.endswith(".pth")))
        m_cfg = m_art.metadata or {}

        model = get_model(m_cfg.get("model_name", args.model_name), **m_cfg).to(device)
        model.load_state_dict(torch.load(m_path, map_location=device))

        embedder = build_embedder()
        loss, acc = evaluate_epoch(model, test_loader, embedder)
        run.summary.update({"loss": loss, "accuracy": acc})

# --------------------------
# Entrypoint
# --------------------------
if __name__ == "__main__":
    cfg = TrainCfg(
        batch_size=args.batch_size,
        epochs=args.epochs,
        batch_log_interval=args.batch_log_interval,
        optimizer=args.optimizer,
        lr=args.lr,
    )

    best_ckpt, model_cfg = train_and_log(cfg, experiment_id="A")
    # If you want an eval job here, you can call:
    # evaluate_and_log("trained-model:latest", experiment_id="A")
