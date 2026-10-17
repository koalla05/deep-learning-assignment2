# Stage 2 experiments

This is the CNN part of the Cats and Dogs++ assignment. Each image has five
binary labels: `Cat`, `Dog`, `Moris`, `Motya`, and `Biatrix`. The competition
scores macro F1, so the three rare named-pet labels matter as much as the two
common labels. Our previous frozen ConvNeXt embeddings plus logistic
regression scored 0.957 in the fair local CV check and **0.92922** on the
public leaderboard. The saved split is in [`folds.csv`](../folds.csv); the
full experiment rationale is in the [Stage 2 plan](../docs/stage-2-plan.md).

We have about 20 hours to get useful results. That is elapsed time, not a
shared compute quota: the three of us can run work in parallel on Macs or
Colab. We will decide how many trials and folds fit after timing a short run
with real images and pretrained weights.

## Setup

From the repository root, use `uv` for Python and project dependencies:

```sh
uv sync --locked
uv run --locked python -c "import torch, timm; print(torch.__version__, timm.__version__)"
```

`uv` creates `.venv`; do not make another environment or install project
packages into it by hand. Select the uv-created interpreter as the notebook
kernel. If an older manually created `.venv` is already present, stop its
kernels and move it to `.venv.manual-backup` before syncing. Keep the backup
until any tools or credentials in it have been recovered; both directories
are ignored by Git.

Put the competition's `train.csv`, `test.csv`, `train/`, and `test/` under the
ignored `data/` directory. Keep `folds.csv` at the repository root. Do not
commit competition data or credentials.

| Folder | Starting owner | Run |
| --- | --- | --- |
| [`00_shared_head/`](00_shared_head/) | Herman | Frozen ConvNeXt backbone with a trained five-label head |
| [`01_convnext_standard/`](01_convnext_standard/) | Herman | Ordinary ConvNeXt fine-tuning; control for the custom block |
| [`02_convnext_custom/`](02_convnext_custom/) | Teammate 1 | ConvNeXt with our residual block |
| [`03_densenet/`](03_densenet/) | Teammate 2 | Pretrained DenseNet comparison |

These owners are a starting point, not a restriction on where a run happens.

## Run order

1. The shared training code is still to be written; only the
   [timing profiler](../notebooks/04_convnext_profile.ipynb) runs today. Build
   the common data loading, metrics, and result saving first. That profiler
   uses random weights and is not a trained model.
2. Time a short training and validation pass on the device you will use.
   Include image loading and checkpoint saving. On a 16 GB Mac, try a smaller
   microbatch and accumulate gradients if needed.
3. Train the head-only model, then the standard ConvNeXt on fold 0. The custom
   model should start from the same pretrained backbone and trained head as
   the standard model. DenseNet can run once the shared data and metrics work.
4. Review the fold-0 results before spending time on extra folds or optimizer
   changes. Fold 0 is a quick comparison, not a five-fold CV result. Leave
   time for test inference and a submission from the strongest model.

## Keep comparisons fair

For the first pass, use `folds.csv`, fold 0, seed 42, labels in the order
`Cat, Dog, Moris, Motya, Biatrix`, BCE with logits, and threshold 0.25. Report
macro F1, F1 for each label, and average precision for each label and their
macro mean. Validation transforms must be deterministic.

Write down the pretrained checkpoint, image size and normalization, training
augmentations, trainable layers, optimizer, scheduler, precision, device,
physical and effective batch sizes, and checkpoint-selection rule. The custom
block needs a standard ConvNeXt run with the same training settings; otherwise
we cannot tell whether the block made the difference. Save curves, gradient
measurements, and class-specific GradCAM images for the models we examine in
the final report.

## Share results

Create `<experiment>/results/` for the first run. Save one JSON file per run
using [`RUN_RECORD_TEMPLATE.json`](RUN_RECORD_TEMPLATE.json), including failed
runs. Save validation probabilities in `<run_id>_val.csv` with columns
`image_id,fold,Cat,Dog,Moris,Motya,Biatrix`. Include exactly the held-out image
IDs for that fold, and record the Git commit in the JSON. Small JSON and CSV
files go through Git.

Keep data, credentials, caches, and temporary or per-epoch checkpoints out of
Git. A Colab resume checkpoint needs durable storage such as Drive. Put only
selected model checkpoints in `<experiment>/checkpoints/`, where Git LFS
applies. Install LFS before pulling one:

```sh
brew install git-lfs
git lfs install
```

Check `git lfs ls-files` before pushing a checkpoint and coordinate large
uploads with the team; LFS storage and download allowances are shared. Kaggle
submission is a separate, manual step after validation, disabled by default
in the notebook.
