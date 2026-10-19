# Exp3: EfficientNetV2-S

[Notebook](../../notebooks/09_efficientnet.ipynb)

A different CNN family for the ensemble: `tf_efficientnetv2_s.in21k_ft_in1k`
from timm, pretrained on ImageNet only (21k classes, then fine-tuned on 1k).
It has 20M parameters against 88M for ConvNeXt-base, and uses MBConv blocks with
squeeze-and-excitation and BatchNorm, so its mistakes should overlap less with
Exp1's. It is not a control for the Exp2 block. There is no completed run or
reportable score yet.

The protocol follows Exp1: fold 0 of the saved folds, 384-pixel letterbox,
BCE loss, fixed 0.25 threshold, checkpoint chosen by fold-0 macro F1, flip-only
augmentation, effective batch 16. Three things differ because the architecture
does:

- **Own head warm-up.** Exp0's head fits ConvNeXt's features, so phase 1 trains
  a new linear head on the frozen backbone (up to 8 epochs at LR `1e-3`; three epochs were not enough for the rare labels), then phase 2
  fine-tunes everything. As with Exp0, the head was selected on fold 0, so it is
  valid for a fold-0 run only.
- **Higher learning rates:** backbone `1e-4`, head `1e-3` (ConvNeXt used
  `1e-5`/`1e-4`). These are untuned starting guesses.
- **Frozen BatchNorm statistics** (`stage2/batchnorm.py`), because the physical
  batch is small. BatchNorm scale and shift still train.

The model name is not pinned to a Hugging Face revision; the run record stores
timm's resolved source and library versions.

Keep each run under `results/<run-id>/` and the selected checkpoint under
`checkpoints/<run-id>_best.ckpt`, following the [experiments guide](../README.md).
With `STAGE2_BUILD_TEST_PREDICTIONS=1` the notebook also saves test
probabilities in `results/<run-id>_test/`, the layout the ensemble notebook
reuses.
