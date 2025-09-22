# src/data/preprocess.py
import os
import argparse
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
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
def read_split(dirpath: str, split: str) -> TensorDataset:
    x, y = torch.load(os.path.join(dirpath, f"{split}.pt"))
    return TensorDataset(x, y)

def save_split(artifact: wandb.Artifact, name: str, dataset: TensorDataset):
    with artifact.new_file(f"{name}.pt", mode="wb") as f:
        x, y = dataset.tensors
        torch.save((x, y), f)

# ----------------------------------------------------
# Embedding & feature utils
# ----------------------------------------------------
def build_embedder() -> InceptionResnetV1:
    """Frozen FaceNet embedder."""
    m = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    for p in m.parameters():
        p.requires_grad = False
    return m

def _hwc_any_to_chw_float_m1_1(imgs_hwc: torch.Tensor) -> torch.Tensor:
    """
    imgs_hwc: (B, H, W, C[=1 or 3]) tensor of dtype uint8/float32/float64 on CPU
    returns: (B, 3, 160, 160) float32 on device, normalized to [-1, 1]
    """
    # If single channel, repeat to 3
    if imgs_hwc.shape[-1] == 1:
        imgs_hwc = imgs_hwc.repeat(1, 1, 1, 3)

    x = imgs_hwc

    # Convert to float32 (handle both uint8 and float inputs)
    if x.dtype == torch.uint8:
        x = x.to(torch.float32) / 255.0
    else:
        x = x.to(torch.float32)
        # Auto-scale if likely 0..255
        with torch.no_grad():
            maxv = float(x.max().item()) if x.numel() > 0 else 1.0
        if maxv > 1.5:
            x = x / 255.0
        x.clamp_(0.0, 1.0)

    # HWC -> CHW
    x = x.permute(0, 3, 1, 2)  # (B,3,H,W)

    # Resize to FaceNet input
    x = F.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False)

    # Normalize to [-1, 1] (mean=0.5, std=0.5)
    x = (x - 0.5) / 0.5

    return x.to(device)

@torch.no_grad()
def embed_pairs(embedder: InceptionResnetV1, pairs: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
    """
    pairs: (N, 2, H, W[, C]) in uint8 or float
    returns: (N, 2, 512) float32 (L2-normalized) on CPU
    """
    # Ensure shape (N, 2, H, W, C)
    if pairs.ndim == 4:  # (N,2,H,W)
        pairs = pairs.unsqueeze(-1)  # -> (N,2,H,W,1)

    N = pairs.shape[0]
    E1_list, E2_list = [], []

    for i in range(0, N, batch_size):
        chunk = pairs[i:i + batch_size]  # (B, 2, H, W, C)
        B = chunk.shape[0]

        # Face 1 batch (CPU -> device)
        imgs1 = chunk[:, 0]                              # (B,H,W,C)
        imgs1 = imgs1.contiguous().cpu()                 # keep CPU for preprocessing
        t1 = _hwc_any_to_chw_float_m1_1(imgs1)           # (B,3,160,160)

        # Face 2 batch
        imgs2 = chunk[:, 1]
        imgs2 = imgs2.contiguous().cpu()
        t2 = _hwc_any_to_chw_float_m1_1(imgs2)

        e1 = embedder(t1)                                # (B,512)
        e2 = embedder(t2)                                # (B,512)

        e1 = F.normalize(e1, p=2, dim=1).cpu()
        e2 = F.normalize(e2, p=2, dim=1).cpu()
        E1_list.append(e1)
        E2_list.append(e2)

    E1 = torch.cat(E1_list, dim=0)
    E2 = torch.cat(E2_list, dim=0)
    return torch.stack([E1, E2], dim=1)  # (N,2,512)

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
    return torch.cat([absdiff, hadamard, cos, one_minus], dim=-1).to(torch.float32)

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

        embedder = build_embedder()

        for split in ["training", "validation", "test"]:
            ds = read_split(raw_dir, split)  # (pairs, y)
            x_raw, y = ds.tensors
            E = embed_pairs(embedder, x_raw, batch_size=args.batch_size)  # (N,2,512)
            feats = make_pair_features(E)                                 # (N,1026)
            processed_split = TensorDataset(feats, y)
            save_split(processed, split, processed_split)
            print(f"Processed {split}: {len(y)} samples → features {tuple(feats.shape)}")

        run.log_artifact(processed)
        print(f"Logged artifact: {args.output_artifact}")

if __name__ == "__main__":
    preprocess_and_log()
