# Kaggle GPU profiling

This is an optional recipe for repeating the `04_convnext_profile.ipynb` compute
probe on Kaggle. The team's interactive Kaggle sessions remained queued, so
Colab is currently the working free-GPU route. The notebook uses random
weights, reads 16 real training images, and makes no Kaggle submission. It
needs two inputs: the Cats and Dogs++ competition data and a small bundle
containing this repo's saved folds and `stage2.profile` helper. The competition
images already live on Kaggle; do not upload the local 1.8 GB copy.

## Prepare the small input bundle locally

From the repository root:

```bash
uv run --locked python scripts/build_kaggle_profile_bundle.py
```

This creates `artifacts/kaggle-profile-bundle.zip` (ignored by Git). It contains only `folds.csv` and two Python source files. Make a **private** Kaggle Dataset from that ZIP. Kaggle may extract ZIP contents when it attaches a Dataset; the notebook handles both extracted files and an intact ZIP. No credentials are needed by the notebook.

## Run the Kaggle notebook

1. Import `notebooks/04_convnext_profile.ipynb` into a Kaggle Notebook.
2. In the Notebook's **Input** pane, attach the Cats and Dogs++ competition data and the private profile-bundle Dataset. Accept the competition rules if prompted.
3. In **Settings**, select a GPU accelerator. Start with the default microbatches `(2, 4)`; if 4 exceeds memory, rerun with `(2,)` and record the failure. The notebook checks for CUDA and prints the GPU name and package versions.
4. Run all cells. Copy the `profile_results` and `estimates` output into the experiment log before ending the session. Compare seconds per 16-image update with the M2 measurements in `stage-2-plan.md`.

The first probe uses Kaggle's installed PyTorch and timm, so its version printout is part of the result. If `timm` is missing, stop and configure a pinned version through Kaggle's Dependency Manager before rerunning. No checkpoint download is needed for this probe.

Check the current [Kaggle Notebooks documentation](https://www.kaggle.com/docs/notebooks)
and [GPU quota guidance](https://www.kaggle.com/docs/efficient-gpu-usage)
before scheduling a long run; session limits and quotas can change. A training
notebook needs a durable checkpoint and a resume path. This profile estimate
is for one configuration, not an entire experiment sweep.
