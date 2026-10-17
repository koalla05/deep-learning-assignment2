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
and gradient clipping at 1.0. Keep the checkpoint's input normalization and
ConvNeXt's built-in LayerNorm; add no new normalization layer. Cap the run at
eight epochs with patience three.
Select the best checkpoint by full-fold macro F1 at 0.25. Use the first
complete epoch, including validation and checkpoint writing, to project the
eight-epoch run. If the projection with a safety margin exceeds five hours,
the notebook stops after that epoch; discuss a smaller run or another device
before proceeding. A few real batches should first verify
backward, optimizer, logging, and checkpoint reload. If only the final
backbone stage is trainable, label that as a different partial-fine-tuning
control, not this full-fine-tuning run.

The run is in `notebooks/06_convnext_standard.ipynb`. It starts with training
off and smoke mode on. A saved full checkpoint can be evaluated later by
setting `STAGE2_EVAL_CHECKPOINT`, without starting another training run.
An eight-batch Colab T4 smoke run passed, including checkpoint reload and a
finite-logit forward pass. Its [executed notebook](results/convnext_standard_fold0_seed42_smoke1_executed.ipynb)
is saved separately; partial-batch scores in it are not validation results.

A real-data, one-epoch T4 profile then took 1,685 seconds (28.1 minutes)
including full-fold validation. A later Lightning integration test showed
that the best-checkpoint write happened after this timer stopped, so the
profile did **not** measure checkpoint I/O or the new Drive backup. The ten-epoch
projection is 4.68 hours before margin and 5.15 hours with the agreed 10%
margin, just over the five-hour gate and still an underestimate of the
protected run. Future epoch timings include those writes. The profile stopped after epoch one;
no full fine-tuning result has been claimed. Peak allocated GPU memory was
2.84 GB. See the [timing record](results/convnext_standard_fold0_seed42_profile3_timing.json),
[CSV metrics](results/convnext_standard_fold0_seed42_profile3_metrics.csv),
[executed notebook](results/convnext_standard_fold0_seed42_profile3_executed.ipynb),
and [W&B run](https://wandb.ai/models-kyiv-school-of-economics/ai301-stage2/runs/c1yn0imt).
We chose an eight-epoch cap for the next control attempt, but its first
protected epoch must still pass the five-hour gate; the earlier profile did
not include checkpoint and Drive-copy costs. The profile checkpoint is only
on the temporary Colab VM.

For a Colab run, mount Drive and opt in with `STAGE2_BACKUP_TO_DRIVE=1`.
`STAGE2_DRIVE_FOLDER_NAME` names the folder directly under MyDrive (default
`AI301-stage2`); for example, `AI301-convnext-controls` puts backups under
`MyDrive/AI301-convnext-controls/01_convnext_standard/<run-id>/`. Use a fresh
`STAGE2_RUN_ID`, because existing backups are not overwritten. The backup
stores epoch resume state, best evaluation checkpoints, and available run
results. Keep the CLI output-bearing notebook separately; a new-VM restore
is not automated yet.

Keep this recipe fixed for the matched custom-block comparison. Test other
learning rates, optimizers, or schedulers only under new run IDs after timing
the control. Save its curves, gradient norms, per-label scores, run JSON, and
fold-0 probabilities following the [experiment README](../README.md). The
[Stage 2 plan](../../docs/stage-2-plan.md) gives the full decision sequence.
New completed records include starting-head/fold SHA256 hashes and label order
for the custom notebook's provenance checks. Upload `stage2/run_records.py`
with the other helpers before running the updated control notebook on Colab.
