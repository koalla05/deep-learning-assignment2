"""Measure one ConvNeXt update on Colab without uploading assignment data."""

from __future__ import annotations

import json
import time

import timm
import torch


MODEL_NAME = "convnext_base.fb_in22k_ft_in1k_384"
IMAGE_SIZE = 384
MICROBATCH_SIZE = 2
EFFECTIVE_BATCH_SIZE = 16
MAX_PHASE_SECONDS = 90

started = time.perf_counter()


def mark(step: str, **details: object) -> None:
    print(json.dumps({"step": step, "elapsed_seconds": round(time.perf_counter() - started, 2), **details}), flush=True)


def timed_phase(name: str, operation) -> float:
    mark(f"{name}_start")
    phase_started = time.perf_counter()
    operation()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - phase_started
    mark(f"{name}_done", seconds=round(seconds, 2))
    if seconds > MAX_PHASE_SECONDS:
        raise TimeoutError(f"{name} exceeded the {MAX_PHASE_SECONDS}-second diagnostic limit")
    return seconds


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Select a CUDA GPU runtime before running this probe.")
    torch.manual_seed(42)
    mark("ready", gpu=torch.cuda.get_device_name(0), torch=torch.__version__, timm=timm.__version__)

    mark("model_create_start")
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=5)
    mark("model_created", parameters=sum(parameter.numel() for parameter in model.parameters()))
    model = model.cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    # Synthetic tensors isolate GPU compute from image decoding and data transfer.
    images = torch.randn(MICROBATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, device="cuda")
    targets = torch.zeros(MICROBATCH_SIZE, 5, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    optimizer.zero_grad(set_to_none=True)
    forward_times: list[float] = []
    backward_times: list[float] = []
    for batch_index in range(EFFECTIVE_BATCH_SIZE // MICROBATCH_SIZE):
        mark("microbatch", index=batch_index + 1)
        forward_start = time.perf_counter()
        mark("forward_start")
        logits = model(images)
        torch.cuda.synchronize()
        forward_seconds = time.perf_counter() - forward_start
        mark("forward_done", seconds=round(forward_seconds, 2))
        if forward_seconds > MAX_PHASE_SECONDS:
            raise TimeoutError("Forward pass exceeded diagnostic limit")
        forward_times.append(forward_seconds)
        loss = criterion(logits, targets) * (MICROBATCH_SIZE / EFFECTIVE_BATCH_SIZE)
        backward_times.append(timed_phase("backward", loss.backward))

    optimizer_seconds = timed_phase("optimizer", optimizer.step)
    mark(
        "result",
        forward_seconds_total=round(sum(forward_times), 2),
        backward_seconds_total=round(sum(backward_times), 2),
        optimizer_seconds=round(optimizer_seconds, 2),
        peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
        peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2),
    )


if __name__ == "__main__":
    main()
