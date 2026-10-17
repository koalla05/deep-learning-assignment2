# Stage 2 CNN plan

Status: the shared helpers and first head-only notebook are implemented. The pretrained, head-only ConvNeXt benchmark completed on fold 0 in Colab. Standard fine-tuning, the custom block, DenseNet, and cross-fold evaluation remain to be run.

## Confirmed scope

- Start with ConvNeXt and benchmark a version without our custom architectural changes.
- Include both head-only and standard fine-tuning benchmarks (accepted).
- Public leaderboard reference: 0.92922 from the existing submission. This is not the local CV reference or the competition's solid-baseline score.
- Initial hardware: user's MacBook Pro M2 with 32 GB unified memory; feasibility and throughput must be measured before committing to a sweep.
- Separate notebook orchestration from shared helpers, while keeping architecture-specific training policies explicit (accepted).
- Evaluate another pretrained family, potentially DenseNet, even if ConvNeXt succeeds. The aspirational target is +0.01 absolute local macro F1 against a comparable baseline; 0.93922 is a separate public leaderboard aspiration. Differences below 0.005 remain hard to interpret.
- Keep training, evaluation, final result formation, and submission reusable across model choices.
- Study a custom architectural change and compare it with the unchanged architecture.
- If the agreed acceptance criterion is not met within the agreed experiment budget, evaluate a different pretrained CNN family.
- Three team members can run experiments. The two teammates have newer Macs with 16 GB unified memory; their usable microbatch sizes must be measured. Work is divided by experiment under `experiments/`, with compact results exchanged through Git.

## Proposed notebook sequence

1. Objective, reference scores, hypotheses, and experiment budget.
2. Imports, seeds, paths, device, logging configuration, and disabled submission flag.
3. Data checks and image-ID joins to the existing folds; label order fixed explicitly.
4. Training and validation transforms; record preprocessing for each checkpoint.
5. Model construction: accept an experiment specification and return a model producing five logits per image. Keep preprocessing requirements and GradCAM target selection alongside the model definition.
6. Shared training and logging: loss, optimizer, scheduler, checkpoint selection, and epoch-level metrics.
7. Unmodified-architecture reference runs, then controlled optimization experiments.
8. Custom-block experiment with a matched control run.
9. Shared diagnostics: curves, layer gradient statistics, and class-specific GradCAM on held-out images.
10. Shared result formation: run configuration, fold scores, out-of-fold probabilities keyed by image ID, per-label metrics, runtime, and checkpoint references.
11. Shared test inference and submission formation: image IDs plus five probabilities, explicit threshold policy, then schema-checked binary CSV.
12. Optional Kaggle submission guarded by a default-false flag; manually recorded leaderboard score and adequacy discussion.

Model-specific code must not enter result or CSV cells. Prediction records carry image IDs, label order, run ID, and fold provenance; filename and threshold are explicit inputs. Loading a saved run should permit evaluation and CSV generation without retraining. Avoid a general plugin framework: a small model factory and shared functions are sufficient initially.

## Proposed ConvNeXt comparisons

- Historical reference: frozen ConvNeXt embeddings plus logistic regression. Preserve its preprocessing and metric caveats when comparing it with new runs.
- Head-only reference: original ConvNeXt backbone and pooling, with a necessary five-output task head; freeze the backbone and train the head.
- Standard fine-tuning reference: same architecture, trainable backbone and task head. No custom feature block. Fine-tuning weights is distinct from modifying architecture.
- Custom variant candidate: a residual bottleneck after the final spatial feature stage and before the existing pooling/head. Candidate structure: 1x1 channel reduction, GELU, 3x3 depthwise convolution, GELU, 1x1 channel restoration, residual addition. Initialize the final projection to zero so the new block initially preserves its input. Validate feature shape and normalization placement before implementation.
- Compare the custom variant with standard fine-tuning under the same data, seeds, preprocessing, training budget, and threshold policy. Changing optimizer or augmentation simultaneously would confound attribution.

The custom block is a hypothesis about useful spatial adaptation for small named pets, not an assumed improvement. Report additional parameters, runtime, overfitting, and GradCAM evidence. Its location cannot recover spatial detail already lost by the backbone.

## Experiment 2: unchanged ConvNeXt fine-tuning

The immediate goal is a measured fold-0 control, not a five-fold score. Keep the same pinned ConvNeXt-base backbone, five-logit classifier, 384-pixel mean-colour letterbox, normalization, horizontal-flip-only training augmentation, label order, and saved fold assignment as Experiment 1. Load its trained head together with the exact pretrained backbone. That head was trained on folds 1–4 and selected using fold 0, so it is suitable for a fold-0 screening run but not as the starting point for a run that holds out another fold.

1. **Check the setup.** Reconstruct the model, load the head with strict key/shape checks, and confirm a forward pass returns five finite logits. Use seed 42 and the fixed fold-0 split. Run a few real training and validation batches in float32 on the chosen device; check memory, backward pass, optimizer step, logging, and checkpoint reload. A partial-batch score is not a validation result.
2. **Run the control.** Unfreeze the entire backbone and head. Use ordinary BCEWithLogitsLoss, AdamW with backbone LR `1e-5` and head LR `1e-4`, betas `(0.9, 0.999)`, and weight decay `1e-2` on weights but not biases or normalization parameters. Keep effective batch 16 through accumulation; record the physical batch and accumulation factor. Clip gradient norm at 1.0. Warm the learning rates over the first epoch, then decay them with a step-based cosine schedule to 10% of their starting peak by the planned final update. Cap at ten epochs and stop after three validation epochs without improvement. Select one checkpoint by full-fold macro F1 at the fixed 0.25 threshold; macro AP is a diagnostic, not a second selection rule.
3. **Use the first complete epoch as the timing gate.** Include image loading, backward passes, validation, logging, and checkpoint writes in the measurement. Project the remaining run time and leave room within the roughly 20-hour elapsed target for the custom comparison, diagnostics, and submission. If full fine-tuning will not fit on the available device, record the reason and either move the unchanged run to another device or define a new *partial-fine-tuning* control; do not silently change the control halfway through.
4. **Evaluate and preserve the result.** Reload the selected checkpoint, predict every fold-0 image, and save probabilities keyed by image ID. Report macro and per-label F1 at 0.25, macro and per-label AP, F1 at 0.5 for context, the learning curves, gradient norms, epoch timings, peak memory, trainable-parameter count, device, code revision, and W&B run ID. Keep test inference and Kaggle submission separate and opt-in. Compare the fold-0 score with Experiment 1's 0.93577 under the same threshold and images; do not compare it numerically with the old five-fold or public-leaderboard scores as if they used the same evaluation.

The custom ConvNeXt run must start from the same fold-0 backbone and head snapshot and use the same split, preprocessing, optimizer policy, schedule, effective batch, epoch cap, checkpoint rule, and threshold. The added block is then the principal difference. We will decide whether to spend compute on learning-rate, optimizer, or scheduler variants only after the control's measured learning curve and time are known; each change gets a separate run ID. If we extend either variant to all five folds, repeat the head warm-up separately within each fold's training partition (or restart both variants from a fresh pretrained backbone/head under a newly declared common protocol). Reusing the fold-0 head in other folds would leak their held-out images into training.

## Proposed experiment discipline

- Begin with ordinary BCE on logits, a fixed shared threshold of 0.25, and conservative augmentation; also report F1 at 0.5 and threshold-free average precision.
- Benchmark the unchanged architecture before custom modifications. Study learning rate/unfreezing first; then a bounded optimizer/scheduler comparison. Consider imbalance treatment only if rare-label diagnostics justify it.
- Use a fixed existing fold for screening if compute is limited; label those results as screening, not five-fold CV. Evaluate finalists across the stored folds and acknowledge model-selection optimism.
- Keep threshold tuning separate from architecture comparisons. Any later tuned thresholds require an inner training-only calibration split or a properly nested procedure; reused OOF scores are not automatically leakage-free.
- Log all runs, including failures. Record gradient statistics after backward, before gradients are cleared; if mixed precision is used, account for scaling. GradCAM needs gradients enabled and class-specific target logits.
- Retain trial and final-result logs with CV/LB fields. Mark missing leaderboard measurements explicitly; never substitute local scores for them.

## Training recipe proposal — second interview round

- First profile the existing ConvNeXt-base checkpoint on MPS in float32: microbatch 2, then 4 if memory permits, roughly 384-pixel input, with training backward and optimizer steps included. Measure warm-up separately, synchronize timing, and include validation overhead when extrapolating runtime. No training speed or fit guarantee before measurement.
- Candidate input policy: aspect-preserving resize and padding to 384 square, checkpoint normalization, mild horizontal flips for training. Verify padding and small-pet detail visually. This differs from the historic no-resize embedding baseline and must be documented.
- Candidate effective batch: 16 via gradient accumulation. This reduces peak memory, not total compute; it is not equivalent to a physical batch of 16 for BatchNorm statistics.
- Head-only benchmark: frozen backbone in evaluation mode; AdamW, head LR 1e-3, weight decay 1e-2, maximum 10 epochs, early-stopping patience 3, constant learning rate. Keep deterministic transforms if caching embeddings; otherwise image augmentation requires repeated backbone inference.
- Standard fine-tuning: branch from the same saved head warm-up checkpoint, rebuild the optimizer over trainable parameters, backbone LR 1e-5 and head LR 1e-4, AdamW with betas (0.9, 0.999), weight decay 1e-2 excluding biases/norm parameters, one epoch of LR warm-up then cosine decay, maximum 10 fine-tuning epochs, patience 3, gradient-norm clipping 1.0. These are starting hypotheses, not established optima.
- First cheap fine-tuning probe may unfreeze only the final stage. Explicitly label it partial fine-tuning and do not silently treat it as full fine-tuning.
- First controlled changes: backbone LR {1e-5, 3e-5}, AdamW versus momentum SGD with independently suitable LR, cosine versus constant LR. Keep a bounded set of named experiments rather than their Cartesian product.
- Keep ordinary BCE and fixed threshold 0.25 initially; choose checkpoints consistently by validation macro F1 at that threshold, with validation AP as a diagnostic. Epoch selection and screening still introduce selection optimism.
- Memory optimization order: small microbatches, accumulation, frozen stages; activation checkpointing only if needed because it trades compute for memory. Test mixed precision against the installed PyTorch/Lightning/MPS combination before relying on it.
- DenseNet comparison needs its own freeze/unfreeze grouping, BatchNorm policy, pretrained preprocessing, and GradCAM layer. Small microbatches make BatchNorm running statistics a particular concern; propose freezing pretrained running statistics during fine-tuning and logging this policy.

Shared helpers accept a model and explicit trainable parameter groups/stage policy. They do not assume ConvNeXt layer names. Persist these policies with each run so a checkpoint can be reproduced and compared.

## Profiling gate, 17 October 2026

The first implementation slice is `notebooks/04_convnext_profile.ipynb`, with a locked Python environment. It uses the unchanged ConvNeXt-base architecture at 384 px, random weights, float32, AdamW, 16-image effective batches, real image decoding and mean-colour letterboxing. MPS on the user's Mac was available. The five-fold split has 9,808 train and 2,452 validation images per fold.

| Microbatch | Training time per 16-image update | Validation time per 16 images | Estimated five folds × ten epochs |
|---|---:|---:|---:|
| 2 | 3.15 s | 0.47 s | 28.49 h |
| 4 | 2.84 s | 0.49 s | 25.88 h |

The estimate includes measured image preparation and excludes logging, checkpoint saves, head-only work, diagnostics, test inference, and thermal effects. It is based on two timed updates after one warm-up, so it is a capacity estimate, not a full-epoch measurement. A serial five-fold, ten-epoch full fine-tuning run would exceed the 20-hour elapsed-time target on this Mac alone. Parallel teammate runs may still deliver multiple results within that deadline. Prioritize the two unmodified-architecture benchmarks and a matched custom-block comparison before broad optimizer sweeps. Do not commit to a five-fold run count or epoch count until the first pretrained one-fold benchmark shows convergence and checkpoint timing.

The profiler deliberately did not download pretrained weights. Kaggle interactive execution remained queued, so the team moved the compute check to Colab. The pretrained head-only benchmark is now complete. The next training gate is a standard fine-tuning benchmark initialized from that head; the matched custom block and DenseNet comparison follow.

The head-only gate is in `notebooks/05_convnext_head.ipynb`. It verifies that the saved folds exactly match shuffled, seed-42 `StratifiedKFold` over the five-label combinations, and keeps training, weight download, and test CSV formation disabled by default. Shared image loading, metric calculation, and CSV contracts live in `stage2/experiment.py`; the small Lightning adapter lives in `stage2/train.py`. Lightning writes local CSV curves and sends W&B telemetry online when training starts. The W&B key comes from the runtime environment or a Colab Secret, never the notebook or Git. An eight-batch pretrained smoke run on MPS and one on Colab completed; their partial validation scores are not model-quality results. The full head-only fold-0 run took 92 minutes for ten epochs plus validation and reached 0.93577 macro F1 and 0.96683 macro AP at the best checkpoint, using threshold 0.25. Its 2,452 held-out predictions and compact head are recorded under `experiments/00_shared_head/`. This is a single-fold screening result, not five-fold CV or a leaderboard score.

The Kaggle interactive session remained queued before any code executed. A free Colab T4 then completed a synthetic 384-pixel ConvNeXt update at microbatch 2, with roughly 2.1 seconds per warmed-up effective batch of 16 and 2.14 GB peak allocated GPU memory. That estimate excluded image loading, validation, checkpointing, and pretrained-weight setup; see `docs/colab-profile.md`. The measured head-only run above supersedes the synthetic timing for that setup. Full fine-tuning still needs a real-data timing gate before fixing the remaining trial count and epoch budgets.

## Team execution

The proposed split is a shared head gate and standard ConvNeXt control for Herman, a matched custom ConvNeXt experiment for teammate 1, and an independent DenseNet comparison for teammate 2. The folders and handoff contract are in `experiments/README.md`. Ownership can change without changing the result layout.

Share source, configuration, fold-0 prediction CSVs, and metric records through ordinary Git. Selected model checkpoints can be shared through Git LFS under `experiments/*/checkpoints/`; temporary and per-epoch checkpoints stay outside Git. A run record must identify the code revision and pretrained checkpoint so teammates can reproduce a result. Parallel execution shortens elapsed time but does not reduce total GPU-hours. The goal is useful results within about 20 hours of elapsed time, not a pooled 20 GPU-hour quota.

The completed Colab head run establishes a roughly 9-minute-per-epoch training-and-validation pace for this frozen-backbone setup, but it does not measure full fine-tuning or the 16 GB Macs. Keep additional optimizer and scheduler trials as options until a pretrained fine-tuning fold gives a real learning curve and runtime.

## Sources

- Assignment PDF, CNN Model requirements and reporting constraints.
- https://huggingface.co/timm/convnext_base.fb_in22k_ft_in1k_384 — current checkpoint provenance and model scale.
- https://huggingface.co/docs/timm/main/model_api — classifier replacement and common model interface.
- https://docs.pytorch.org/docs/main/notes/mps.html — Apple GPU backend support.
- https://huggingface.co/timm/densenet121.ra_in1k — possible ImageNet-pretrained alternative.
