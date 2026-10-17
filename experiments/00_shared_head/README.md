# Frozen ConvNeXt, trained head

Use pretrained `convnext_base.fb_in22k_ft_in1k_384` and train a linear head
that outputs five logits. Keep the backbone frozen and in evaluation mode.
This gives us a direct CNN benchmark alongside the older frozen-embedding
logistic-regression result.

Start with a short fold-0 run to measure memory and time per epoch. Then use
seed 42, AdamW with head learning rate `1e-3` and weight decay `1e-2`, BCE,
effective batch 16, and validation threshold 0.25. Ten epochs with patience
three is an upper limit for now; adjust it after seeing the learning curve.
The first implementation is [the head-only notebook](../../notebooks/05_convnext_head.ipynb).
Its training and checkpoint download switches start off. Run the smoke mode
first, then change the run ID and train the complete fold; do not quote the
partial-batch smoke score as a validation result.

Record the exact pretrained checkpoint, transforms, physical batch size,
timings, and validation scores. Save the trained head separately so both
ConvNeXt fine-tuning runs can start from the same weights. Put the run JSON
and held-out probabilities in `results/` as described in the
[experiment README](../README.md).

The first Colab T4 run is recorded in
[`results/convnext_head_fold0_seed42_colab.json`](results/convnext_head_fold0_seed42_colab.json).
Ten epochs took 92 minutes, and the best checkpoint scored 0.93577 macro F1
on fold 0 at threshold 0.25. This is a screening result, not five-fold CV.
The held-out probabilities and small trained head are beside the record.
