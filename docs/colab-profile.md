# Colab ConvNeXt compute probe

On 17 October 2026, a free Colab T4 runtime successfully ran one update of
`convnext_base.fb_in22k_ft_in1k_384` with five output logits at 384 pixels.
The probe used random weights, synthetic images, FP32, AdamW, an effective
batch of 16, and eight microbatches of 2. It did not use the assignment data.

| Measurement | Result |
| --- | ---: |
| First forward pass, including CUDA warmup | 3.20 s |
| Later forward passes | 0.05–0.06 s each |
| Later backward passes | 0.19–0.20 s each |
| AdamW step | 0.15 s |
| Peak allocated GPU memory | 2.14 GB |
| Peak reserved GPU memory | 2.26 GB |

The seven later microbatches plus one optimizer step imply roughly **2.1 seconds
per effective batch of 16**, before data loading, augmentation, validation,
pretrained weight loading, logging, and checkpointing. That is about **21 minutes
for one 9,808-image training epoch** on fold 0, with substantial uncertainty.
Five folds × ten epochs would already take roughly **17–18 hours of training
compute**, before the omitted work. This is an extrapolation, not a measured
epoch or a promise that six trials fit the 20-hour budget.

To repeat the synthetic compute check, use the Colab CLI as an isolated uv
tool after authenticating it. It is not a project dependency and does not need
to be installed inside the project's `.venv`:

```sh
uvx --from 'google-colab-cli==0.7.4' colab new -s ai301-probe --gpu T4
uvx --from 'google-colab-cli==0.7.4' colab install -s ai301-probe 'timm>=1.0,<2'
uvx --from 'google-colab-cli==0.7.4' colab exec -s ai301-probe -f scripts/colab_convnext_probe.py
uvx --from 'google-colab-cli==0.7.4' colab stop -s ai301-probe
```

This synthetic probe did not time image decoding, transfer, validation, or
checkpointing. The real-data head-only run below gives a more useful wall-clock
measurement for that setup. Full-backbone fine-tuning still needs its own
measurement. Save checkpoints outside the temporary VM if a long run must
survive a runtime reset.

## Pretrained head-only run

The first real-data run used the pinned pretrained ConvNeXt-base checkpoint,
fold 0, 384-pixel mean-colour letterboxing, microbatch 2, effective batch 16,
float32, and a frozen backbone. Ten epochs plus validation took 5,520 seconds
(92 minutes) on a free T4; predicting the complete held-out fold from the best
checkpoint took another 112 seconds. The best fold-0 macro F1 at threshold
0.25 was 0.93577, with macro average precision 0.96683. The run is recorded
in `experiments/exp0/results/convnext_head_fold0_seed42_colab/convnext_head_fold0_seed42_colab.json`
and [W&B](https://wandb.ai/models-kyiv-school-of-economics/ai301-stage2/runs/v93j8869).
This is a single-fold head benchmark, not a full fine-tuning speed estimate or
five-fold CV score. The small head weights and held-out probabilities were
copied back to the repository; the full Lightning checkpoint remains on the
temporary Colab VM.
