# Exp3: DenseNet comparison

This is a planned comparison with pretrained `densenet121.ra_in1k`. It is a
different CNN family, not the control for the Exp2 block. There is no completed
run or reportable score yet.

Use the saved fold and five-label metric contract. Profile memory and runtime
before training, then record the DenseNet-specific learning rate, unfreezing
policy, image preprocessing, and BatchNorm policy. Small physical batches make
the treatment of BatchNorm running statistics especially important.

Keep each run under `results/<run-id>/` and selected checkpoints under
`checkpoints/`, following the [experiments guide](../README.md).
