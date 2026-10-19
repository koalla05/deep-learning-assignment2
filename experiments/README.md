# Stage 2 experiments

The notebooks compare CNNs on the same saved fold assignment and five labels:
`Cat`, `Dog`, `Moris`, `Motya`, and `Biatrix`. The rare named pets contribute
three-fifths of macro F1. The [Stage 2 plan](../docs/stage-2-plan.md) gives the
experimental rationale; the [root report](../README.md) holds the score table.

| Folder | Question | Notebook |
| --- | --- | --- |
| [exp0](exp0/) | How well does a frozen ConvNeXt with a trained head work? | [05](../notebooks/05_convnext_head.ipynb) |
| [exp1](exp1/) | What does full fine-tuning add without changing the architecture? | [06](../notebooks/06_convnext_standard.ipynb) |
| [exp2](exp2/) | Does a residual spatial block help under the same recipe? | [07](../notebooks/07_convnext_custom.ipynb) |
| [exp3](exp3/) | Does a different CNN family (EfficientNetV2-S) add to the ensemble? | [09](../notebooks/09_efficientnet.ipynb) |
| [exp4](exp4/) | Does averaging completed runs beat the best single model? | [08](../notebooks/08_ensemble.ipynb), [10](../notebooks/10_ensemble_exp1_exp3.ipynb) |

## Folder convention

Put each attempt under `exp<index>/results/<run-id>/`. A smoke or later
evaluation gets its own run ID, not extra files in a completed run. Keep the
run JSON, fold predictions, curves, diagnostics, and an executed notebook
together. Never overwrite an earlier run. Put only selected, shareable model
files in `exp<index>/checkpoints/`; Git LFS covers that path. Temporary or
per-epoch checkpoints and W&B internals stay ignored inside the run directory.

Exp0–Exp2 were produced before this layout was settled. Their original JSON
records were kept intact, including paths from their training runtimes. The
files are now grouped by run ID, and selected checkpoints moved to the LFS
location. Use the JSON's checkpoint SHA256, not its old local path, when
matching a checkpoint to a record. The source notebooks use the new layout
for future runs.

Share small run records, prediction CSVs, plots, and executed notebooks through
Git. Keep competition data, credentials, W&B caches, and routine checkpoints
out of it. Do not call a fold-0 score five-fold CV; the Exp0 head was selected
on fold 0 and cannot be reused for a clean test on another validation fold.

## Local setup

From the repository root:

```sh
uv sync --locked --group dev
uv run --locked python -c "import torch, timm; print(torch.__version__, timm.__version__)"
```

Use the uv-created `.venv` as the notebook kernel. The data stays in ignored
`data/`; `folds.csv` stays at the root. Notebook flags default to no training,
weight download, or submission-file creation. Kaggle upload is always manual.

`stage2/` is the importable training and evaluation code used by notebooks;
`scripts/` holds standalone Colab/Kaggle helpers. They stay separate so
notebook imports and terminal commands remain clear.

## Colab CLI and result recovery

These are examples; set your session and run ID first. Mounting Drive is
interactive. The CLI may reuse a kernel, so upload changed helper files and
the matching source notebook before a run. Never pass API keys as CLI values
or commit the ignored `.env`. The notebooks accept `WANDB_API_KEY` from that
file or a runtime secret. CLI-driven execution may require a temporary `.env`
upload; `STAGE2_REMOVE_RUNTIME_ENV_AFTER_LOAD=1` removes it after loading.

```sh
COLAB_SESSION=ai301-teammate-t4
RUN_ID=convnext_standard_fold0_seed42_newrun
colab sessions
colab status -s "$COLAB_SESSION"
colab url -s "$COLAB_SESSION"
colab drivemount -s "$COLAB_SESSION"
colab ls -s "$COLAB_SESSION" /content/drive/MyDrive
colab log -s "$COLAB_SESSION" -n 20
colab upload -s "$COLAB_SESSION" stage2/finetune.py /content/ai301/stage2/finetune.py
colab upload -s "$COLAB_SESSION" stage2/run_records.py /content/ai301/stage2/run_records.py
colab upload -s "$COLAB_SESSION" notebooks/06_convnext_standard.ipynb /content/ai301/notebooks/06_convnext_standard.ipynb
colab exec -s "$COLAB_SESSION" -f notebooks/06_convnext_standard.ipynb --timeout 21600 --env STAGE2_ROOT=/content/ai301 --env STAGE2_RUN_TRAINING=1 --env STAGE2_SMOKE_TEST=0 --env STAGE2_PROFILE_ONLY=0 --env STAGE2_RUN_ID="$RUN_ID" --env STAGE2_BACKUP_TO_DRIVE=1
colab ls -s "$COLAB_SESSION" "/content/drive/MyDrive/AI301-stage2/01_convnext_standard/$RUN_ID"
```

`colab exec` produces an output-bearing notebook separate from the clean
source. Save it in the matching run directory with the JSON and CSV. Drive
backup retains verified epoch checkpoints and completed results when enabled,
but not necessarily the executed notebook. Check local or Drive copies before
stopping a VM. A new-VM resume is not automated. While a long run is active,
check no more often than every ten minutes unless someone requests a check.

For selected checkpoints, install Git LFS before pulling or pushing:

```sh
brew install git-lfs
git lfs install
git lfs ls-files
```

The old run record template remains at [RUN_RECORD_TEMPLATE.json](RUN_RECORD_TEMPLATE.json).
It is a reference, not a substitute for the richer JSON written by the current
notebooks.
