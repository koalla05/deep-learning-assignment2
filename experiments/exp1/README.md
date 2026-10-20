# Exp1: standard ConvNeXt

[Notebook](../../notebooks/06_convnext_standard.ipynb) · [Run record](results/convnext_standard_fold0_seed42_20261018_full1/convnext_standard_fold0_seed42_20261018_full1.json)

This is the unchanged-architecture control for Exp2. It starts from the Exp0
head, unfreezes the whole ConvNeXt, and keeps the same fold-0 split, 384-pixel
letterbox input, BCE loss, and 0.25 decision threshold. The trained head is
specific to fold 0; reusing it for another held-out fold would leak data.

The full run used AdamW with backbone/head rates `1e-5`/`1e-4`, one warm-up
epoch followed by cosine decay, effective batch 16, weight decay 0.01 on
eligible weights, and an eight-epoch cap. Early stopping ended training after
five epochs. The best fold-0 macro F1 was 0.97518; the public Kaggle score was
0.97003. These are single-fold and public-test results, not five-fold CV.

The run directory holds the record, held-out probabilities, curves, diagnostics,
GradCAM images, and executed notebook. The
[selected checkpoint](checkpoints/convnext_standard_fold0_seed42_20261018_full1_best.ckpt)
is in the LFS directory. The older run record
retains its original Colab artifact paths; its SHA256 identifies the checkpoint.
Shared commands and folder conventions are in the [experiments guide](../README.md).
