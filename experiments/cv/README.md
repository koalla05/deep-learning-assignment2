# Five-fold cross-validation

Results of the notebooks in [notebooks/cv/](../../notebooks/cv/). Every model family is trained with its
earlier recipe on all five saved folds, so each training image gets an out-of-fold (OOF) prediction and
the test set gets the mean of five fold models. The shared code is [stage2/cv.py](../../stage2/cv.py);
per-label thresholds are in [stage2/thresholds.py](../../stage2/thresholds.py).

| Folder | Recipe | Notebook | Fold 0 |
| --- | --- | --- | --- |
| `convnext/` | Exp1 ConvNeXt-base | [01](../../notebooks/cv/01_cv_convnext.ipynb) | imported from `exp1` |
| `convnext_custom/` | Exp2 ConvNeXt + residual block (starts from the `convnext/` head of each fold) | [02](../../notebooks/cv/02_cv_convnext_custom.ipynb) | imported from `exp2` |
| `efficientnetv2s/` | Exp3 EfficientNetV2-S | [03](../../notebooks/cv/03_cv_efficientnetv2s.ipynb) | imported from `exp3` |
| `bit_r50x1/` | Exp5 BiT ResNetV2-50x1 | [04](../../notebooks/cv/04_cv_bit_resnetv2.ipynb) | imported from `exp5` |
| `hrnet_w32/` | Exp7 HRNet-W32 | [05](../../notebooks/cv/05_cv_hrnet_w32.ipynb) | imported from `exp07` |
| `ensemble/` | Weighted means of the models above | [06](../../notebooks/cv/06_cv_ensemble.ipynb) | — |

Run 01 before 02. Run 06 after at least two models have a `cv.json`.

## Layout of one model

```
<model>/fold<k>/head/head.pt, head.json     linear head trained on frozen features of the training folds
<model>/fold<k>/checkpoints/last.ckpt       full training state after every epoch (Git-ignored)
<model>/fold<k>/checkpoints/best.ckpt       selected weights (Git-ignored)
<model>/fold<k>/curves/version_*/metrics.csv  per-epoch log; a resumed session adds a version
<model>/fold<k>/training_done.json          training has ended; the selected checkpoint is final
<model>/fold<k>/oof.csv, oof_logits.csv     held-out fold probabilities and logits
<model>/fold<k>/test.csv, test_logits.csv   test probabilities and logits of this fold's model
<model>/fold<k>/fold.json                   completion record with the SHA-256 of every file above
<model>/fold<k>/failure_*.json              one note per failed attempt (error type and frames only)
<model>/cv.json, oof.csv, test_mean.csv     CV scores, per-label thresholds, OOF and fold-mean test probabilities
```

A fold is finished only when `fold.json` exists and its hashes match. Rerunning a notebook skips finished
folds, resumes an unfinished one from `last.ckpt`, and never trains a fold past `training_done.json`.
Records and submissions are written once and never overwritten.

## Running

Training needs `STAGE2_RUN_TRAINING=1`. Useful variables: `STAGE2_CV_FOLDS=1,2` (folds of this session),
`STAGE2_SMOKE_TEST=1` (a few batches into `<model>_smoke/`), `STAGE2_BATCH_SIZE`, `STAGE2_PRECISION`,
`STAGE2_USE_WANDB=0`, `STAGE2_REUSE_FOLD0=0` (retrain fold 0), `STAGE2_ALLOW_WEIGHT_DOWNLOAD=1` (first run).

On Colab, set `STAGE2_CV_OUTPUT` to a Drive folder (for example `/content/drive/MyDrive/AI301-stage2/cv`), so
every epoch survives the VM, and copy the folder back into `experiments/cv/` afterwards. The notebooks
read the same variable, so they can also aggregate directly from Drive.
