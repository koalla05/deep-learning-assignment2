# ConvNeXt with a residual block

The run is in [`notebooks/07_convnext_custom.ipynb`](../../notebooks/07_convnext_custom.ipynb).
It wraps the same pinned pretrained ConvNeXt and fold-0 classifier used by the
standard control. A small block sits between the last spatial feature map and
the existing pooling/head: `1x1` channel reduction, GELU, `3x3` depthwise
convolution, GELU, `1x1` restoration, then residual addition. The final
projection is zero-initialized, so the wrapper's initial logits must exactly
match the unchanged model; the notebook checks this before training. The
block adds no normalization layer.

Keep the control's fold 0, seed 42, transforms, AdamW groups, warm-up/cosine
schedule, full unfreezing, effective batch 16, eight-epoch cap, checkpoint
rule, and threshold 0.25. The new block is assigned the backbone LR `1e-5`;
testing a higher block LR would be a separate optimization experiment. A
smaller physical microbatch is fine on a 16 GB Mac if accumulation keeps the
effective batch the same; record that difference. No full custom training run
has been performed yet.

## Local smoke and short runtime profile

Set `LOCAL_SMOKE_TEST=True` in the notebook's configuration cell, then run it
top to bottom with the uv-created kernel. This one switch enables a bounded
smoke test and runtime profiling, and disables W&B, Drive and weight downloads.
The pinned pretrained backbone must already be cached and the shared head must
be present. Defaults remain non-training when the switch is false.

For terminal execution from the repository root:

```sh
uv sync --locked --group dev
STAGE2_LOCAL_SMOKE_TEST=1 \
STAGE2_RUN_ID=convnext_custom_local_probe1 \
uv run --locked jupyter execute notebooks/07_convnext_custom.ipynb \
  --timeout=600 --output=../artifacts/convnext_custom_local_probe1_executed
```

Choose a new run ID and output filename for each attempt. The smoke test runs
32 training batches (four optimizer updates) and eight validation batches at
physical batch two. It verifies updates to the backbone, classifier and block,
then checks that checkpoint reload exactly reproduces the final logits.
Profiling excludes the first complete accumulated update and first validation
batch. It measures loading and actual Lightning training on the selected
device, plus the validation-metrics/checkpoint tail, and writes
`artifacts/<run-id>_smoke/smoke_profile.json`. Set `PROFILE_RUNTIME=True` with
ordinary smoke mode to use the same short profile outside the local preset.
`PROFILE_ONLY=True` remains the separate, one-full-epoch timing mode.

The local MPS run `convnext_custom_fold0_seed42_mps_profile2` passed in a fresh
kernel: 64 training images, 16 validation images and four optimizer updates.
Training and validation took 17.6 seconds. After warm-up, the measured
16-image training update averaged 3.89 seconds. Extrapolation gives
39.7 minutes of training and 1.5 minutes of validation per full epoch,
or **6.07 hours for eight epochs plus final validation with a 10% margin**.
That is above the five-hour budget; no full run was launched.
The historical [profile record](results/convnext_custom_fold0_seed42_mps_profile2_profile.json)
and [executed notebook](results/convnext_custom_fold0_seed42_mps_profile2_executed.ipynb)
preserve the measurement. These are short-sample runtime estimates, not quality
scores or a measured full epoch. They exclude setup, Drive backup, test
inference and thermal drift; a full run still needs its protected epoch gate.
That executed copy predates the provenance and detailed diagnostics changes;
its source hashes identify the code used for that measurement.

After those changes, `convnext_custom_fold0_seed42_mps_profile3` passed the
same fresh-kernel smoke: four optimizer updates, finite changed backbone/head/
block weights, and exact checkpoint-logit reproduction. Fit time was 18.7 s.
Its short extrapolation is 43.4 min/epoch and **6.39 hours** for eight epochs
plus final validation and margin, still over budget. These small samples do
not establish a performance regression. Sampled MPS high-water marks were
1.33 GiB allocated and 3.76 GiB driver memory, not true peaks.
The [profile](results/convnext_custom_fold0_seed42_mps_profile3_profile.json),
[executed notebook](results/convnext_custom_fold0_seed42_mps_profile3_executed.ipynb),
[layer gradients](results/convnext_custom_fold0_seed42_mps_profile3_layer_gradients.csv)
and [memory record](results/convnext_custom_fold0_seed42_mps_profile3_memory.json)
preserve the updated evidence.

## Provenance, failures and diagnostics

Evaluation without training requires `STAGE2_EVAL_CHECKPOINT` **and**
`STAGE2_EVAL_RUN_RECORD` (the original completed custom run JSON). The notebook
checks the checkpoint SHA256, starting-head/fold hashes, label order and recipe
before loading checkpoint weights. Smoke/timing runs and older records missing
this provenance are rejected. Match the recorded training configuration in the
configuration cell; changing it does not change the checkpoint's history.

Use a fresh `STAGE2_RUN_ID` for evaluation too. It writes validation probabilities
and a separate evaluation JSON under `artifacts/<run-id>_evaluation/`, retaining
original training provenance and recording evaluation code, versions and device
separately. Optional GradCAM images and submission files stay in that new run
directory rather than modifying the original checkpoint directory.

Training writes `diagnostics/layer_gradients.csv` with per-layer mean/max L2
norms after unscaling and before clipping, and a separate custom-block gradient
curve. `diagnostics/memory.json` records CUDA allocator peaks, sampled MPS
allocation/driver high-water marks and process-lifetime peak RSS. MPS sampling
does not capture a true peak; RSS includes earlier allocations in that process.
Completed records include total trainable parameters and macro/per-label F1 at
both 0.25 and 0.5, plus macro/per-label AP.

Failures after run-directory creation write a local `failure.json` with phase,
configuration and exception type/stack locations, then re-raise the error.
Exception messages and source lines are deliberately excluded to avoid saving
credentials. Interruptions are recorded too. Failures before locating the repo
or creating the unique run directory still fail directly. Retrieve failed-run
records manually; automatic Drive failure backup is not implemented.

## Colab and matched comparison

Training, weight downloads, Drive backup, GradCAM, and submission formation
are disabled by default. Use a unique `STAGE2_RUN_ID` for each smoke, timing,
or full attempt. On Colab, mount Drive and opt in with
`STAGE2_BACKUP_TO_DRIVE=1`; `STAGE2_DRIVE_FOLDER_NAME` chooses the parent
folder under MyDrive (default `AI301-stage2`). Run artifacts then go under
`02_convnext_custom/<run-id>/`. A completed run can be evaluated later with
`STAGE2_EVAL_CHECKPOINT` and `STAGE2_EVAL_RUN_RECORD`. Set `BUILD_GRADCAM=True` in the notebook to generate
fixed, class-specific examples after evaluation.

Before a CLI-driven Colab run, upload the custom architecture, GradCAM helper,
and notebook alongside the shared files already on the VM. Use the same
credential procedure described in the [experiment README](../README.md),
without putting the W&B key in a command or notebook.

```sh
colab upload -s ai301-head-fold0 stage2/custom_convnext.py /content/ai301/stage2/custom_convnext.py
colab upload -s ai301-head-fold0 stage2/gradcam.py /content/ai301/stage2/gradcam.py
colab upload -s ai301-head-fold0 stage2/run_records.py /content/ai301/stage2/run_records.py
colab upload -s ai301-head-fold0 stage2/diagnostics.py /content/ai301/stage2/diagnostics.py
colab upload -s ai301-head-fold0 notebooks/07_convnext_custom.ipynb /content/ai301/notebooks/07_convnext_custom.ipynb
```

Set `STAGE2_CONTROL_RUN_ID` to identify the completed unchanged control used
for comparison. Until that control exists and its settings are checked, a
custom fold-0 score is exploratory, not evidence that the block helped. Keep
the output-bearing notebook, run JSON, validation probabilities, extra
parameter count, time per epoch, per-label scores, gradient curves, and
GradCAM examples with the run files described in the
[experiment README](../README.md).

The comparison uses the original recorded custom recipe and checks optimizer learning rates/decay, planned epoch cap,
patience, loss, clipping and full unfreezing as well as the shared data and
metric settings. Both notebooks now record `max_epochs` and `patience`;
both also record head/fold hashes and label order. Older records missing these
fields cannot establish a matched comparison.
