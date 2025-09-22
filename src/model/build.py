# build.py
import os
import argparse
import torch
import wandb

from src.model import get_model # <- registry factory
##testtrr
parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, default="testing console")
parser.add_argument('--model', type=str, default='pairhead_mlp_bn',
                    help='Model name registered in src (__init__.py): '
                         'linear, pairhead_mlp_bn, pairhead_mlp_res')
# Common config knobs (pair-head defaults for LFW embeddings)
parser.add_argument('--input_shape', type=int, default=1026)   # 512+512+2
parser.add_argument('--hidden_layer_1', type=int, default=256)
parser.add_argument('--hidden_layer_2', type=int, default=128)
parser.add_argument('--num_classes', type=int, default=1)      # binary logit
args = parser.parse_args()

print(f"IdExecution: {args.IdExecution}")
os.makedirs("./model", exist_ok=True)

def build_model_and_log(config: dict, model: torch.nn.Module,
                        model_name: str, model_description: str):
    with wandb.init(
        project="MLOps-Pycon2023",
        name=f"initialize {model_name} ExecId-{args.IdExecution}",
        job_type="initialize-model",
        config=config
    ) as run:
        art = wandb.Artifact(
            model_name, type="model",
            description=model_description,
            metadata=dict(config)
        )
        fname = f"initialized_model_{model_name}.pth"
        path = os.path.join("./model", fname)
        torch.save(model.state_dict(), path)
        art.add_file(path)
        run.log_artifact(art)

# ---- Build the selected model from the registry
model_config = {
    "input_shape": args.input_shape,
    "hidden_layer_1": args.hidden_layer_1,
    "hidden_layer_2": args.hidden_layer_2,
    "num_classes": args.num_classes,
}
model = get_model(args.model, **model_config)

build_model_and_log(
    model_config,
    model,
    model_name=args.model,
    model_description=f"{args.model} initialized for LFW pair verification"
)
###hol