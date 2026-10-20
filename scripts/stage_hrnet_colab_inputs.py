"""Prepare one upload folder for the HRNet-W32 Colab notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_source(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.out or root / "artifacts" / "hrnet_w32_input").resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite staged inputs: {output}")
    head = root / "artifacts" / "hrnet_head" / "hrnet_w32_head_fold0_seed42"
    sources = [root / name for name in ("folds.csv", "pyproject.toml", "uv.lock")]
    sources += sorted((root / "stage2").glob("*.py"))
    sources += [root / "notebooks" / "17_hrnet_w32_finetune.ipynb"]
    sources += [head / name for name in ("head_best.pt", "head_best.json", "backbone.json")]
    if any(not path.is_file() for path in sources):
        raise FileNotFoundError("Missing source, lockfile, notebook, or completed local head")
    data = root / "data"
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        if not (data / name).is_file():
            raise FileNotFoundError(data / name)
    for name in ("train", "test"):
        if not (data / name).is_dir():
            raise FileNotFoundError(data / name)
    manifest = {"files": {}, "git_revision": subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()}
    for source in sources:
        if source.is_relative_to(head):
            target = output / "head" / source.name
        else:
            target = output / "repo" / source.relative_to(root)
        copy_source(source, target)
        manifest["files"][str(target.relative_to(output))] = sha256(target)
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        source = data / name
        target = output / "data" / name
        copy_source(source, target)
        manifest["files"][str(target.relative_to(output))] = sha256(target)
    for name in ("train", "test"):
        target_dir = output / "data" / name
        target_dir.mkdir(parents=True)
        for source in (data / name).glob("*.jpg"):
            target = target_dir / source.name
            target.hardlink_to(source)
        print(f"Staged {len(list(target_dir.glob('*.jpg')))} {name} images as local hard links")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Upload this folder to MyDrive/AI301-stage2/hrnet_w32_input: {output}")


if __name__ == "__main__":
    main()
