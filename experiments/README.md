# Stage 2 experiments

This is the CNN part of the Cats and Dogs++ assignment. Each image has five
binary labels: `Cat`, `Dog`, `Moris`, `Motya`, and `Biatrix`. The competition
scores macro F1, so the three rare named-pet labels matter as much as the two
common labels. Our previous frozen ConvNeXt embeddings plus logistic
regression scored 0.957 in the fair local CV check and **0.92922** on the
public leaderboard. The saved split is in [`folds.csv`](../folds.csv); the
full experiment rationale is in the [Stage 2 plan](../docs/stage-2-plan.md).
The folds come from shuffled `StratifiedKFold(n_splits=5, random_state=42)`
over the five-label combinations, not from a fresh random split in each
notebook. One four-image combination cannot appear in every fold; the saved
assignment is the common comparison point.

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

## Colab CLI quick reference

Use the standalone `colab` CLI from a terminal. The session name below is the
one used for our current fold-0 work; replace it if Colab assigns a new VM.
Drive mounting is interactive, so run that command yourself and approve the
Google prompt. A new runtime may need a fresh mount. If `colab` is not on your
PATH, use the full path to its executable (on Herman's Mac,
`/Users/hermanhavva/.local/bin/colab`).

```sh
colab sessions
colab status -s ai301-head-fold0
colab url -s ai301-head-fold0
colab drivemount -s ai301-head-fold0
colab ls -s ai301-head-fold0 /content/drive/MyDrive
colab log -s ai301-head-fold0 -n 20
```

From the repository root, `colab exec -s ai301-head-fold0 -f
notebooks/06_convnext_standard.ipynb --timeout 21600 --env
STAGE2_ROOT=/content/ai301 ...` executes the local notebook on the VM and
writes a separate `_output.ipynb` beside the source notebook. Use the run
flags described in that notebook; for a protected Colab training run include
`--env STAGE2_BACKUP_TO_DRIVE=1` and a fresh `STAGE2_RUN_ID`. Do not launch a
full run until its epoch count and time budget are agreed. The CLI needs a
connection to return the executed notebook. If that connection drops, do not
assume training continued; inspect the session and Drive backups before
restarting anything.

Before execution, upload the matching helper and source notebook so the VM's
run record hashes the code that actually ran. The notebook reloads the helper
module because CLI calls can reuse a Python kernel:

```sh
colab upload -s ai301-head-fold0 stage2/finetune.py /content/ai301/stage2/finetune.py
colab upload -s ai301-head-fold0 stage2/run_records.py /content/ai301/stage2/run_records.py
colab upload -s ai301-head-fold0 notebooks/06_convnext_standard.ipynb /content/ai301/notebooks/06_convnext_standard.ipynb
```

Inspect or retrieve individual VM files with these commands (replace
`<run-id>` with the actual name, without angle brackets):

```sh
colab ls -s ai301-head-fold0 /content/ai301/artifacts
colab ls -s ai301-head-fold0 /content/drive/MyDrive/AI301-stage2/01_convnext_standard
colab download -s ai301-head-fold0 /content/ai301/artifacts/<run-id>/timing_gate.json ./timing_gate.json
colab log -s ai301-head-fold0 -o ./colab_session_log.ipynb
```

The output notebook is produced by `colab exec`; the session log is a
fallback, not a substitute for that notebook. Copy and verify the output
notebook and small results under the experiment's `results/` folder before
sharing. Never use `colab stop` until the needed VM files have been checked
locally or on Drive. We check long-running sessions no more frequently than
once every ten minutes unless a teammate asks for an immediate status check.

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

1. The [head-only notebook](../notebooks/05_convnext_head.ipynb) has completed
   its first fold-0 run. Its saved head and validation predictions are under
   [`00_shared_head/`](00_shared_head/). The older
   [timing profiler](../notebooks/04_convnext_profile.ipynb) uses random weights
   and is not a trained model.
   Training runs send telemetry to the `ai301-stage2` W&B project. Set
   `WANDB_API_KEY` in the environment before launching Jupyter, or add a Colab
   Secret with that name and grant the notebook access. The key never belongs
   in a notebook, run record, shell command history, or Git. CSV curves are
   also saved locally; W&B does not upload checkpoints or training photos.
   Colab Secrets are not available to CLI-driven execution. For a CLI run,
   upload a temporary, W&B-only `.env` to the runtime and set
   `STAGE2_REMOVE_RUNTIME_ENV_AFTER_LOAD=1` so the notebook removes that file
   as soon as it loads the key. Never pass the key's value in a CLI argument.
2. Smoke-test full fine-tuning with real images, then use its first complete
   epoch to measure training and validation time, memory, and checkpoint cost.
   On a 16 GB Mac, try a smaller physical batch and accumulate gradients if
   needed. Do not extrapolate full fine-tuning time from the frozen-head run.
3. Train the unchanged ConvNeXt on fold 0 as the control. The custom model
   should start from the same pretrained backbone and trained fold-0 head,
   with the same training settings. DenseNet can run in parallel once the
   shared data and metrics work.
4. Review the fold-0 results before spending time on extra folds or optimizer
   changes. Fold 0 is a quick comparison, not a five-fold CV result. Reusing
   the fold-0 head for other validation folds would leak data; each additional
   fold needs its own training-only head warm-up or a fresh shared start.
   Leave time for test inference and a submission from the strongest model.

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

Preserve an output-bearing notebook for every substantive Colab run. Executing
an `.ipynb` with the Colab CLI saves a separate `_output.ipynb`; move that
copy to the experiment's `results/` with a run-specific name so it can be
reviewed and committed alongside the run record. Keep the source notebook
clean. Before sharing the executed copy, check for error outputs,
excessive progress logs, and credentials. If Colab has no Recent copy, the
CLI session history may still hold outputs that can be matched to source cells.

Keep data, credentials, caches, and temporary or per-epoch checkpoints out of
Git. With `STAGE2_BACKUP_TO_DRIVE=1`, the standard ConvNeXt notebook saves a
full resume checkpoint after every completed epoch and a copy of each newly
selected best model under `MyDrive/<folder-name>/01_convnext_standard/<run-id>/`.
Set `STAGE2_DRIVE_FOLDER_NAME` to choose that folder name (default
`AI301-stage2`); it must be a single folder directly under MyDrive.
It also copies curves, timing records, fold predictions, and the final run
record as they become available. Each copy is checksum-verified before it
receives its final filename; an incomplete `.partial` file needs manual
inspection before retrying. Ensure Drive has room for several full training
checkpoints. The saved full state contains optimizer and scheduler data, but
restoring it into a fresh VM and continuing the notebook is not automated yet.
The CLI's output-bearing notebook still needs to be preserved
separately after execution. Put only
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
