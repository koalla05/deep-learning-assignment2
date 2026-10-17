# ConvNeXt with a residual block

Add a small block between ConvNeXt's last spatial feature map and its pooling
layer: `1x1` reduction, GELU, `3x3` depthwise convolution, GELU, `1x1`
restoration, then add the input back. Zero-initialize the last projection so
the block initially acts like an identity. Check tensor shapes and the GradCAM
target layer before training.

Start from the same pretrained backbone and trained head as
`01_convnext_standard`. Match its fold, seed, transforms, optimizer, scheduler,
unfreezing, effective batch, epoch limit, and checkpoint rule. Use threshold
0.25. A smaller physical microbatch is fine on a 16 GB Mac if gradient
accumulation keeps the effective batch the same; record the difference.

Name the standard run used for comparison. If its settings differ, rerun the
control before comparing scores. Record extra parameters, time per epoch,
per-label scores, gradient flow, and GradCAM examples with the run files
described in the [experiment README](../README.md).
