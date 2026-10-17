# Standard ConvNeXt

Fine-tune the pretrained ConvNeXt with the head from `00_shared_head`. Do not
add a new feature block: this run is the reference for `02_convnext_custom`.

Start on fold 0 with seed 42, BCE, and threshold 0.25. The first recipe is
AdamW (backbone LR `1e-5`, head LR `1e-4`, weight decay `1e-2`), one warm-up
epoch followed by cosine decay, and gradient clipping at 1.0. Cap the run at
ten epochs with patience three, but use the real-data timing and learning
curve to decide whether that is affordable. If only the final backbone stage
is trainable, label the run as partial fine-tuning.

Keep the first run as the control. Test other learning rates, optimizers, or
schedulers under new run IDs. Save its curves, per-label scores, run JSON, and
fold-0 probabilities following the [experiment README](../README.md).
