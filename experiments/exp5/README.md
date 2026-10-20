# Exp5: BiT ResNetV2-50x1

[Notebook](../../notebooks/12_bit_resnetv2.ipynb)

A third CNN family for the ensemble: `resnetv2_50x1_bit.goog_in21k_ft_in1k` from
timm (Google's BiT-M, pretrained on ImageNet-21k and fine-tuned on ImageNet-1k),
with 23.5M parameters. Its normalization differs from the other members:
GroupNorm with weight-standardized convolutions and no BatchNorm, against
ConvNeXt's LayerNorm (Exp1) and EfficientNet's BatchNorm (Exp3).

Held-out fold 0 macro F1 is 0.99851, but the public leaderboard score is only
0.94035, so on its own BiT is clearly weaker than the other members. Averaged
with Exp1 and Exp3 it still raised the public score from 0.97761 to 0.98047
(see [exp4](../exp4/)). The fold-0 score did not predict its test performance.

The protocol follows Exp3: the saved folds, 384-pixel letterbox, BCE loss, fixed
0.25 threshold, checkpoint chosen by held-out-fold macro F1, flip-only
augmentation, effective batch 16, and its own head warm-up (8 epochs) before the
full fine-tune. Differences from Exp3:

- **No BatchNorm freezing**, because BiT has no running statistics.
- **Learning rates `3e-5` (backbone) and `3e-4` (head)**, between ConvNeXt's
  `1e-5`/`1e-4` and Exp3's `1e-4`/`1e-3`. These values are untuned guesses.
- The classifier is a 1×1 convolution on pooled features, which is a linear layer.
- Native size is 448; we kept 384 for identical preprocessing. Whether the lower
  resolution explains the weak test score is untested.

The full run took 2.1 hours on a Mac GPU (30 minutes head warm-up, 94 minutes
fine-tune at 11.8 minutes per epoch). The held-out fold defaults to 0; set
`STAGE2_VALIDATION_FOLD` to run another one (the run name follows it, and each
fold trains its own head).

Keep each run under `results/<run-id>/` and the selected checkpoint under
`checkpoints/<run-id>_best.ckpt`, following the [experiments guide](../README.md).
With `STAGE2_BUILD_TEST_PREDICTIONS=1` the notebook saves test probabilities in
`results/<run-id>_test/`, the layout the ensemble notebooks reuse.
