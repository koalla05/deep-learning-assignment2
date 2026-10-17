# Frozen ConvNeXt, trained head

Use pretrained `convnext_base.fb_in22k_ft_in1k_384` and train a linear head
that outputs five logits. Keep the backbone frozen and in evaluation mode.
This gives us a direct CNN benchmark alongside the older frozen-embedding
logistic-regression result.

Start with a short fold-0 run to measure memory and time per epoch. Then use
seed 42, AdamW with head learning rate `1e-3` and weight decay `1e-2`, BCE,
effective batch 16, and validation threshold 0.25. Ten epochs with patience
three is an upper limit for now; adjust it after seeing the learning curve.

Record the exact pretrained checkpoint, transforms, physical batch size,
timings, and validation scores. Save the trained head separately so both
ConvNeXt fine-tuning runs can start from the same weights. Put the run JSON
and held-out probabilities in `results/` as described in the
[experiment README](../README.md).
