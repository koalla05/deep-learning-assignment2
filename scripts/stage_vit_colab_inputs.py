"""Prepare one upload folder for notebooks/cv/07_cv_vit.ipynb on Colab.

Creates artifacts/vit_colab_input/ with
  code/      stage2/*.py, folds.csv, pyproject.toml, the two notebooks (a few hundred KB)
  data.zip   train/ test/ images and the three CSVs, stored without compression (~1.8 GB, one file uploads reliably)
Upload the folder's contents to MyDrive/AI301-stage2/vit_input/. Re-run with --code-only after editing stage2/ to refresh code/.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--code-only", action="store_true", help="refresh code/ only; leave data.zip as it is")
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "artifacts" / "vit_colab_input"
    code = output / "code"
    if code.exists():
        shutil.rmtree(code)  # code/ is always rebuilt; it only holds copies
    for name in ("folds.csv", "pyproject.toml"):
        (code / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, code / name)
    for source in sorted((root / "stage2").glob("*.py")):
        (code / "stage2").mkdir(exist_ok=True)
        shutil.copy2(source, code / "stage2" / source.name)
    for name in ("notebooks/cv/07_cv_vit.ipynb", "notebooks/cv/06_cv_ensemble.ipynb"):
        (code / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, code / name)
    print(f"code/ ready: {sum(1 for p in code.rglob('*') if p.is_file())} files")
    if args.code_only:
        return
    archive = output / "data.zip"
    if archive.exists():
        print("data.zip already exists; delete it to rebuild.")
        return
    data = root / "data"
    temporary = archive.with_suffix(".partial")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED) as zf:  # JPEGs do not compress further
        for name in ("train.csv", "test.csv", "sample_submission.csv"):
            zf.write(data / name, f"data/{name}")
        for split in ("train", "test"):
            images = sorted((data / split).glob("*.jpg"))
            for image in images:
                zf.write(image, f"data/{split}/{image.name}")
            print(f"{split}: {len(images)} images")
    temporary.rename(archive)
    print(f"data.zip: {archive.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
