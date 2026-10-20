# HRNet-W32 training plan

Status: agreed plan. Implementation and training have not started.

## Settled requirements

- Branch: `herman/HRNet-W32`, based on `ensemble`.
- Model: `hrnet_w32.ms_in1k`, ImageNet-1k pretrained; adapt its native 224-pixel pretraining to 384×384 task inputs.
- Targets: `Cat`, `Dog`, `Moris`, `Motya`, `Biatrix`, with five independent logits.
- Train the new linear classifier locally first with the HRNet backbone frozen. Save the selected classifier weights for transfer.
- Run joint backbone and classifier fine-tuning on Colab through the CLI, using a readable notebook that runs from top to bottom with minimal manual interaction. Both backbone and classifier parameters are trainable in this phase, with learning-rate warm-up and cosine decay.
- Interpret "freeze its weights" between phases as preserving the selected local head checkpoint, not leaving the head frozen during the explicitly requested joint fine-tuning.
- Preserve model weights and run artifacts in Google Drive using the existing checksum-verified transfer mechanism.
- During the training workflow, reload the selected best checkpoint and generate validation and full competition-test sigmoid probabilities. Test inference runs after checkpoint selection, not during every epoch.
- Save both sigmoid probabilities and raw logits in separate files, indexed by `image_id`, with explicit label order. Use probabilities in the existing local ensemble; retain logits for possible later calibration or stacking.
- Probabilities are unrounded outputs before thresholding, with no normalization across labels. Sigmoid maps each logit independently to [0, 1]; the five probabilities need not sum to one. Do not mix logits into an ensemble that averages probabilities.
- Perform threshold adjustment locally. The initial comparison uses the existing fixed 0.25 threshold.
- No Kaggle submission is part of this training workflow.
- Compute budget: approximately five hours on Colab, including setup, smoke testing, joint fine-tuning, prediction export, and verified Drive persistence. Local head training is outside this Colab budget.
- Select the best epoch by validation macro F1 at the fixed 0.25 threshold. Monitor training and validation loss to diagnose overfitting; loss does not replace the agreed primary selection metric.
- No image exclusions. Use all rows in the selected training partitions and preserve the stored validation partition.

## First training recipe

- Stored fold 0, seed 42; keep all existing training rows for the baseline.
- Existing 384×384 letterbox, ImageNet normalization, horizontal flip augmentation.
- Unweighted BCEWithLogitsLoss, matching the existing ConvNeXt, EfficientNetV2-S, and BiT training code in `stage2/train.py`. No `weight` or `pos_weight` multipliers. Pass raw logits to this numerically stable loss; apply sigmoid explicitly for prediction/export, not before the loss.
- Local classifier training: frozen backbone in evaluation mode, LR 1e-3, up to eight epochs initially; evaluate whether the rare labels need more local epochs.
- Full fine-tuning: backbone LR 3e-5, classifier LR 3e-4; AdamW, weight decay 1e-4 excluding normalization parameters and biases.
- Effective batch 16. Choose physical batch and CUDA precision after a smoke test; freeze BatchNorm running statistics while allowing affine parameters to train during fine-tuning.
- One epoch learning-rate warm-up followed by cosine decay to 10% of peak learning rates, following the existing scheduler; fine-tuning cap 20 epochs.
- One fold-0 run. Measure real throughput before choosing the Colab epoch count and cosine schedule horizon; reserve time for best-checkpoint validation/test inference and verified Drive copies. The fine-tuning epoch cap is a ceiling, not a promise to fit 20 epochs into five hours.
- Select by highest validation macro F1 at 0.25; break ties with lower validation BCE, then the earlier epoch. Stop joint fine-tuning after four epochs without improvement under this rule, or at the budget boundary reserved for export. Restore the selected checkpoint before producing final predictions.

Training loss includes augmentation and changes in model parameters throughout the epoch; validation loss is computed without augmentation at the epoch's final parameters. Interpret sustained trends rather than treating their absolute gap as proof of overfitting. Examine per-label loss and F1 because aggregate BCE can hide rare-pet deterioration.

These settings are hypotheses, not measured optima.

## Artifact and recovery requirements

- Preserve both selected model weights for inference and the latest complete training state for resume. Record selected epoch and checkpoint checksum.
- The local head must use the same training fold, label order, preprocessing, and exact pretrained HRNet backbone as the Colab run. Verify this metadata on transfer; start a fresh joint-training optimizer after loading the selected head.
- Save validation/test probabilities from the same selected checkpoint, with finite values in [0, 1], unique image IDs, complete expected coverage, and verified label order.
- Save corresponding finite logits with identical IDs and label order, and verify that their sigmoid matches the probability files within serialization precision. Save validation targets for local evaluation; competition test targets are unavailable.
- Save configuration, checkpoint source, package versions, code revision, folds/data identity, preprocessing, curves, timing, and executed notebook.
- Report macro and per-label F1 at the fixed baseline threshold, plus macro/per-label average precision. Do not describe a selected-checkpoint validation score as an unbiased final evaluation.
- Keep each run in a distinct directory. Verify persistent copies before treating a Colab run as safely archived.
- Separate setup/authentication and explicit artifact transfer from the experiment. Reuse prior Drive transfer conventions where practical.
- Fail early on missing data, missing GPU for the intended Colab run, alignment errors, or non-finite values. Never silently start a costly CPU run.
- Persist mandatory weights, predictions, and run metadata before optional diagnostics such as GradCAM. A diagnostic failure must not prevent artifact export.

## Implementation and execution sequence

1. Build readable local-head and Colab fine-tuning notebooks with compact configuration, fresh-kernel execution, and working CLI environment overrides. Use the installed notebook skill's experiment template. Keep computation, persistence, and external setup separate.
2. Verify data and folds, checkpoint identity, label order, and a small end-to-end smoke run. Seed all random sources and use the locked uv environment.
3. Train and select the local head. Preserve the head checkpoint and metadata and provide local recovery during this phase.
4. Mount Drive through the existing interactive authentication step, transfer the required inputs, and verify the Colab starting state. Further manual interaction should only be needed for an actionable failure.
5. Measure HRNet throughput and memory on the available GPU; choose physical batch, mixed precision, and an affordable epoch count while retaining effective batch 16. Record these choices before joint training.
6. Fine-tune both body and head with the agreed scheduler, selection rule, loss curves, and per-epoch verified Drive backups. Implement explicit checkpoint restoration with optimizer and scheduler state for interrupted Colab runs.
7. Reload the selected checkpoint, export validation/test probabilities and logits, verify alignment and checksums, and archive weights, results, and the executed notebook to Drive.
8. Evaluate and adjust thresholds locally from saved predictions. No automatic Kaggle submission.

No remaining design questions. Hardware-dependent settings are determined by the smoke test and recorded in the run configuration. This plan does not claim that the five-hour limit guarantees a particular epoch count or leaderboard improvement.

## Existing workflow findings

`notebooks/12_bit_resnetv2.ipynb` already exports validation and test probabilities from the best checkpoint and copies artifacts to Drive. `stage2/finetune.py` provides `DriveRunBackup` and checksum-verified copies using temporary destinations, with overwrite protection.

The current notebook initializes backups after head warm-up and does not pass a resume checkpoint to training. The HRNet workflow therefore needs protection during both training phases and an explicit restore path; merely saving a checkpoint does not implement recovery.

Threshold and ensemble-weight tuning require predictions separate from the fold used for final reporting. The first screening run does not by itself establish five-fold generalization or an ensemble improvement.
