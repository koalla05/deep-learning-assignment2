# Stage 2 CNN plan

Status: implementation started. MPS and a Colab T4 completed compute probes; training schedules remain provisional until a real-data learning-curve probe.

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

The profiler deliberately does not download pretrained weights. Kaggle interactive execution remained queued, so the team moved the compute check to Colab. The next training gate is a pretrained checkpoint, then a short head-only and standard fine-tuning benchmark with logging. The custom block and DenseNet comparison follow with a revised time budget.

The Kaggle interactive session remained queued before any code executed. A free Colab T4 then completed a synthetic 384-pixel ConvNeXt update at microbatch 2, with roughly 2.1 seconds per warmed-up effective batch of 16 and 2.14 GB peak allocated GPU memory. This excludes image loading, validation, checkpointing, and pretrained-weight setup; see `docs/colab-profile.md`. The next gate is a short real-data, pretrained, one-fold head-only and fine-tuning run. Time every stage before fixing the number of trials and epochs.

## Team execution

The proposed split is a shared head gate and standard ConvNeXt control for Herman, a matched custom ConvNeXt experiment for teammate 1, and an independent DenseNet comparison for teammate 2. The folders and handoff contract are in `experiments/README.md`. Ownership can change without changing the result layout.

Share source, configuration, fold-0 prediction CSVs, and metric records through ordinary Git. Selected model checkpoints can be shared through Git LFS under `experiments/*/checkpoints/`; temporary and per-epoch checkpoints stay outside Git. A run record must identify the code revision and pretrained checkpoint so teammates can reproduce a result. Parallel execution shortens elapsed time but does not reduce total GPU-hours. The goal is useful results within about 20 hours of elapsed time, not a pooled 20 GPU-hour quota.

The first real-data run must determine whether 16 GB Macs fit the chosen physical microbatch, how long one epoch and validation take, and whether the learning curve warrants ten epochs. Until then, the six candidate trials are options rather than a promised schedule.

## Sources

- Assignment PDF, CNN Model requirements and reporting constraints.
- https://huggingface.co/timm/convnext_base.fb_in22k_ft_in1k_384 — current checkpoint provenance and model scale.
- https://huggingface.co/docs/timm/main/model_api — classifier replacement and common model interface.
- https://docs.pytorch.org/docs/main/notes/mps.html — Apple GPU backend support.
- https://huggingface.co/timm/densenet121.ra_in1k — possible ImageNet-pretrained alternative.
