# DenseNet comparison

Try pretrained `densenet121.ra_in1k` on the same saved fold and metrics. This
run asks whether another CNN family is worth pursuing; it is not the control
for the ConvNeXt block. It can run in parallel once the shared data and
evaluation code is ready.

First check memory and throughput on the chosen Mac or Colab runtime. Record
the pretrained checkpoint and its image preprocessing. DenseNet has BatchNorm
layers, so decide whether to keep their running statistics frozen during
fine-tuning and record that choice. Small physical batches make it important.

Use fold 0, seed 42, BCE, and threshold 0.25 for the first comparison. Choose
and record a DenseNet-specific learning rate and unfreezing policy rather than
copying ConvNeXt layer settings. Save per-label scores, curves, runtime,
fold-0 probabilities, and the run JSON as described in the
[experiment README](../README.md). Add a class-specific GradCAM example if
this becomes a finalist.
