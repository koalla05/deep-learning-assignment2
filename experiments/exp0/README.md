# Exp0: frozen ConvNeXt, trained head

[Notebook](../../notebooks/05_convnext_head.ipynb) · [Run record](results/convnext_head_fold0_seed42_colab/convnext_head_fold0_seed42_colab.json)

This run kept pretrained `convnext_base.fb_in22k_ft_in1k_384` frozen and trained
only a five-logit linear classifier. It provided a CNN starting point and a
shared head for Exp1 and Exp2. The head was trained on folds 1–4 and selected
on fold 0, so it cannot initialize a clean evaluation on another held-out fold.

The Colab T4 run used AdamW at `1e-3`, weight decay 0.01, effective batch 16,
and a fixed 0.25 threshold. Ten epochs took about 92 minutes; the best fold-0
macro F1 was 0.93577, and the public Kaggle score was 0.93365. This is a
single-fold screening result, not five-fold CV.

The run directory contains probabilities, the original JSON record, and the
recovered output-bearing notebook. The selected
[head checkpoint](checkpoints/convnext_head_fold0_seed42_colab_head.pt) is in the
LFS directory. The record retains its original runtime paths;
the shared [experiments guide](../README.md) explains the current layout.
