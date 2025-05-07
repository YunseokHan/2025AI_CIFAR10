import argparse
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import CIFAR10Variant
from models import get_model
import torch
from torchvision import transforms


def evaluate(cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
    ])
    test_ds = CIFAR10Variant('./data', False, tf)
    loader  = torch.utils.data.DataLoader(test_ds, 256, False, num_workers=4)

    model = get_model(cfg['model']).to(device)
    model.load_state_dict(torch.load(f"./results/{cfg['model']}/{cfg['tag']}/best_model.pt", map_location=device))
    model.eval()

    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            all_pred.append(pred); all_true.append(y)
    y_pred = torch.cat(all_pred).numpy()
    y_true = torch.cat(all_true).numpy()

    # Compute confusion matrix and accuracies
    cm   = confusion_matrix(y_true, y_pred)
    accs = cm.diagonal() / cm.sum(axis=1)
    overall_acc = cm.diagonal().sum() / cm.sum()

    save_dir = Path("./results") / cfg['model'] / cfg["tag"]
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) Save per‑class & overall accuracy
    class_names = test_ds.classes              # e.g., ["airplane", ...]
    acc_df = pd.DataFrame({
        "class": class_names + ["overall"],
        "acc":   list(accs) + [overall_acc]
    })
    acc_df.to_csv(save_dir / "class_acc.csv", index=False)

    # 2) Save raw confusion matrix
    np.savetxt(save_dir / "confusion.txt", cm, fmt="%d")

    # 3) Plot and save confusion matrix heatmap
    vmax = max(cm.max(), 1)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        vmin=0,
        vmax=vmax,
        linewidths=.5,
        square=True,
        cbar_kws={"label": "Count"},
        xticklabels=class_names,
        yticklabels=class_names,
    )
    if cfg["tag"] == 'baseline':
        plt.title("Confusion Matrix : Baseline")
    if cfg["tag"] == 'input_perturb':
        plt.title("Confusion Matrix : Input Perturbation")
    if cfg["tag"] == 'random_shuffle':
        plt.title("Confusion Matrix : Random Shuffle")
    if cfg["tag"] == 'label_noise':
        plt.title("Confusion Matrix : Label Noise")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_dir / "confusion.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg)