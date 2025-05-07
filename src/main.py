import argparse
import yaml
from train import trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--wandb_api_key", type=str, default=None, help="WanDB API Key")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    trainer(cfg, use_wandb=args.wandb_api_key)