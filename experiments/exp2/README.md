# Exp2: ConvNeXt with a residual spatial block

[Notebook](../../notebooks/07_convnext_custom.ipynb) · [Run record](results/convnext_custom_fold0_seed42_mps_gradcam3/convnext_custom_fold0_seed42_mps_gradcam3.json)

This experiment asks whether a small spatial block helps distinguish the named
pets. It starts from the same pretrained ConvNeXt and Exp0 head as Exp1. The
block sits before pooling: a 1×1 channel reduction, 3×3 depthwise convolution,
1×1 restoration, and residual addition. Its final projection starts at zero,
so the initial logits match the unchanged model.

Fold, transforms, loss, optimizer, scheduler, effective batch, threshold, and
checkpoint rule match Exp1. The block uses the backbone rate `1e-5`. The run
completed eight epochs and reached 0.97401 fold-0 macro F1; its public Kaggle
score was 0.96043. The difference from Exp1 is evidence for this particular
recipe, not a five-fold estimate of the block's effect.

The run directory holds the record, held-out probabilities, curves, diagnostics,
GradCAM images, and executed notebook. The
[selected checkpoint](checkpoints/convnext_custom_fold0_seed42_mps_gradcam3_best.ckpt)
is in the LFS directory. The historical JSON
retains paths recorded on the training machine. Shared commands and conventions
are in the [experiments guide](../README.md).
