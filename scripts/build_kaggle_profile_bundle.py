"""Package the saved folds and profile helper for a Kaggle Notebook input."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/kaggle-profile-bundle.zip"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output = args.out.resolve()
    if output.exists():
        raise SystemExit(f"Output already exists: {output}. Choose a new --out path.")
    output.parent.mkdir(parents=True, exist_ok=True)

    names = ("folds.csv", "stage2/__init__.py", "stage2/profile.py")
    with ZipFile(output, mode="x") as bundle:
        for name in names:
            info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            bundle.writestr(info, (root / name).read_bytes())
    print(f"Created {output} ({output.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
