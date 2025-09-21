# src/data/preprocess.py
import os
import argparse
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from facenet_pytorch import InceptionResnetV1
import wandb

# ----------------------------------------------------
# Args
# ----------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--IdExecution", type=str, default="console")
parser.add_argument("--input_artifact", type=str, default="lfw-pairs-raw:latest",
                    help="W&B artifact with training.pt/validation.pt/test.pt of raw pairs.")
parser.add_argument("--output_artifact", type=str, default="lfw-pairs-embeddings",
                    help="Name of the processed artifact to create.")
parser.add_argument("--batch_size", type=int, default=128, help="Embedding batch size.")
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ----------------------------------------------------
# I/O helpers
# ----------------------------------------------------
def read_split(data_dir: str, split: str) -> TensorDataset:
    x, y = torch.load(os.path.join(data_dir, f"{split}.pt"))
    return TensorDataset(x, y)

def save_split(artifact: wandb.Artifact, name: str, dataset: TensorDataset):
    with artifact.new_file(f"{name}.pt", mode="wb") as f:
        x, y = dataset.tensors
        torch.save((x, y), f)

# ----------------------------------------------------
# Embedding & feature utils
# ----------------------------------------------------
def build_embedder() -> Tuple[InceptionResnetV1, Compose]:
    """Frozen FaceNet and a transform to 3x160x160 in [-1,1]."""
    embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    for p in embedder.parameters():
        p.requires_grad = False
    tx = Compose([
        Resize((160, 160)),
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return embedder, tx

def _to_rgb(img: torch.Tensor) -> torch.Tensor:
    """img: (H,W) or (H,W,1) or (H,W,3) uint8 -> (H,W,3) uint8 (CPU tensor)."""
    if img.ndim == 2:
        img = img.unsqueeze(-1).repeat(1, 1, 3)
    elif img.shape[-1] == 1:
        img = img.repeat(1, 1, 3)
    return img

@torch.no_grad()
def embed_pairs(embedder, tx, pairs: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
    """
    pairs: (N, 2, H, W[, C]) uint8
    returns: embeddings (N, 2, 512) float32 (L2-normalized)
    """
    # Ensure shape (N, 2, H, W, C)
    if pairs.ndim == 4:
        pairs = pairs.unsqueeze(-1)  # add C=1
    N = pairs.shape[0]

    # Process in mini-batches to save memory
    E1_list, E2_list = [], []
    for i in range(0, N, batch_size):
        chunk = pairs[i:i + batch_size]  # (B, 2, H, W, C)
        B = chunk.shape[0]
        tens = []
        for k in range(2):
            imgs = chunk[:, k]  # (B, H, W, C)
            batch_imgs = []
            for b in range(B):
                rgb = _to_rgb(imgs[b].cpu())
                batch_imgs.append(tx(rgb))  # (3, 160, 160)
            tens.append(torch.stack(batch_imgs, dim=0).to(device))
        e1 = embedder(tens[0])  # (B, 512)
        e2 = embedder(tens[1])  # (B, 512)
        e1 = F.normalize(e1, p=2, dim=1)
        e2 = F.normalize(e2, p=2, dim=1)
        E1_list.append(e1.cpu())
        E2_list.append(e2.cpu())
    E1 = torch.cat(E1_list, dim=0)
    E2 = torch.cat(E2_list, dim=0)
    return torch.stack([E1, E2], dim=1)  # (N, 2, 512)

def make_pair_features(E: torch.Tensor) -> torch.Tensor:
    """
    E: (N, 2, 512) -> features (N, 1026)
      x = [|e1-e2|, e1*e2, cos, 1-cos]
    """
    e1, e2 = E[:, 0, :], E[:, 1, :]
    absdiff = (e1 - e2).abs()
    hadamard = e1 * e2
    cos = F.cosine_similarity(e1, e2).unsqueeze(-1)
    one_minus = 1.0 - cos
    feats = torch.cat([absdiff, hadamard, cos, one_minus], dim=-1).to(torch.float32)
    return feats  # (N, 1026)

# ----------------------------------------------------
# Main preprocess & log
# ----------------------------------------------------
def preprocess_and_log():
    with wandb.init(
        project="MLOps-Pycon2023",
        name=f"Preprocess LFW pairs -> embeddings | ExecId-{args.IdExecution}",
        job_type="preprocess-data"
    ) as run:
        # Input raw pairs
        raw_art = run.use_artifact(args.input_artifact)
        raw_dir = raw_art.download(root="./data/artifacts/")

        # Output processed artifact
        processed = wandb.Artifact(
            args.output_artifact, type="dataset",
            description="LFW pairs converted to Facenet embeddings and 1026-D pair features",
            metadata={
                "source_artifact": args.input_artifact,
                "feature_schema": "[|e1-e2|, e1*e2, cos, 1-cos]",
                "embedding_model": "facenet-pytorch InceptionResnetV1 (vggface2)",
                "feature_dim": 1026
            }
        )

        # Embed once, save features for all splits
        embedder, tx = build_embedder()

        for split in ["training", "validation", "test"]:
            ds = read_split(raw_dir, split)         # (pairs uint8, y long)
            x_raw, y = ds.tensors
            E = embed_pairs(embedder, tx, x_raw, batch_size=args.batch_size)  # (N,2,512)
            feats = make_pair_features(E)           # (N,1026) float32
            processed_split = TensorDataset(feats, y)
            save_split(processed, split, processed_split)
            print(f"Processed {split}: {len(y)} samples → features {tuple(feats.shape)}")

        run.log_artifact(processed)
        print(f"Logged artifact: {args.output_artifact}")

if __name__ == "__main__":
    preprocess_and_log()
