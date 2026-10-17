# Standard ConvNeXt

Fine-tune the pretrained ConvNeXt with the fold-0 head from `00_shared_head`.
Do not add a feature block: this is the control for `02_convnext_custom`.
The head can only be reused for a fold-0 screening run. It was trained on
images from folds 1–4, so it would leak validation data in other folds.

Start on fold 0 with seed 42, the same image processing as the head run,
BCEWithLogitsLoss, and threshold 0.25. Unfreeze the full backbone and head.
The first recipe is AdamW (backbone LR `1e-5`, head LR `1e-4`, betas
`(0.9, 0.999)`, weight decay `1e-2` except on biases and normalization),
one epoch of step-based warm-up followed by cosine decay, effective batch 16,
and gradient clipping at 1.0. Cap the run at ten epochs with patience three.
Select the best checkpoint by full-fold macro F1 at 0.25. Use the first
complete epoch to measure actual training-plus-validation time before
committing to the remaining budget. A few real batches should first verify
backward, optimizer, logging, and checkpoint reload. If only the final
backbone stage is trainable, label that as a different partial-fine-tuning
control, not this full-fine-tuning run.

Keep this recipe fixed for the matched custom-block comparison. Test other
learning rates, optimizers, or schedulers only under new run IDs after timing
the control. Save its curves, gradient norms, per-label scores, run JSON, and
fold-0 probabilities following the [experiment README](../README.md). The
[Stage 2 plan](../../docs/stage-2-plan.md) gives the full decision sequence.
