# CIFAR‑10 Corrupted Image Classification with ResNet‑50

This repository provides a **single, reproducible pipeline** for measuring how a vanilla ResNet‑50 reacts to three canonical data–label corruptions on CIFAR‑10:

| Variant                | What changes?                                     | Config file                       |
| ---------------------- | ------------------------------------------------- | --------------------------------- |
| **Baseline**           | Nothing – clean i.i.d. data                       | `experiments/baseline.yaml`       |
| **Input Perturbation** | Blur→Sharpen transform applied to *inputs only*   | `experiments/input_perturb.yaml`  |
| **Label Noise (20 %)** | 20 % of training labels flipped to a random class | `experiments/label_noise20.yaml`  |
| **Random Shuffle**     | 100 % of training labels uniformly shuffled       | `experiments/random_shuffle.yaml` |

Training all four variants (50 epochs each) takes **≈ 70 min on a single NVIDIA A100**.

---

## 📑 Quick start

```bash
# Clone & create env (optional)
git clone https://github.com/YunseokHan/2025AI_CIFAR10.git
cd 2025AI_CIFAR10
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # torch, torchvision, pyyaml, etc.
```

### 1. Train

`train.sh` launches the four experiments with identical hyper‑parameters (Adam, lr = 1e‑4, batch 128).

```bash
bash train.sh           # ~70 min wall‑clock on A100; creates results/<tag>/
```

### 2. Evaluate & plot

`evaluate.sh` loads the best checkpoint for each tag, computes confusion matrices / per‑class accuracy, and writes summary figures.

```bash
bash evaluate.sh        # produces results/<tag>/{confusion.png,class_acc.csv}
```

Generated artefacts – including **val‑loss log**, **confusion matrices**, and **entropy statistics** – are stored under `results/` and referenced by the accompanying LaTeX report.

---

## 🗂 Repository layout

```
experiments/   # YAML configs (one per variant)
  baseline.yaml
  input_perturb.yaml
  label_noise20.yaml
  random_shuffle.yaml
src/
  datasets.py  # CIFAR10Variant loader (noise / shuffle / perturb)
  models.py    # ResNet‑50 w/ CIFAR stem tweak
  train.py     # training loop + val‑loss checkpointing
  evaluate.py  # confusion matrix, class‑acc, entropy
train.sh       # bash helper – runs 4× train.py
evaluate.sh    # bash helper – runs 4× evaluate.py
results/       # logs & figures will appear here after running scripts
report/        # CVPR‑style LaTeX paper (compile w/ latexmk)
```

---

## ⚙️ train.sh

```bash
#!/usr/bin/env bash
set -e
for cfg in experiments/*.yaml; do
  echo "=== Training $cfg ==="
  python src/train.py --config "$cfg"
done
```

## ⚙️ evaluate.sh

```bash
#!/usr/bin/env bash
set -e
for tag in baseline input_perturb label_noise20 random_shuffle; do
  echo "=== Evaluating $tag ==="
  python src/evaluate.py --tag "$tag"
done
```

---

## Reproducing paper figures

After `evaluate.sh`, run the notebook `notebooks/make_plots.ipynb` (or `analysis.py`) to regenerate:

* Accuracy‑vs‑epoch curves
* Final accuracy bar chart (Fig.
* 2×2 confusion‑matrix grid (Fig.
* Entropy table (Tab.

Paths in the notebook assume the default `results/` structure.

---

## Citation

If you find this scaffold useful, please cite the accompanying report:

```bibtex
@misc{han2025cifar10corrupt,
  title   = {Revisiting CIFAR‑10 Robustness with Unified Corruptions},
  author  = {Yunseok Han},
  year    = {2025},
  url     = {https://github.com/YunseokHan/2025AI_CIFAR10}
}
```

---

## License

MIT License.  See `LICENSE` for details.
