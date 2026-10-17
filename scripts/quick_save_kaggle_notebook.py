"""Upload a private Kaggle notebook version without executing its cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
from kagglesdk.kernels.types.kernels_enums import KernelExecutionType


def metadata_bool(metadata: dict[str, object], name: str) -> bool:
    value = metadata[name]
    if value in (True, "true"):
        return True
    if value in (False, "false"):
        return False
    raise ValueError(f"{name} must be true or false")


def build_request(folder: Path) -> tuple[ApiSaveKernelRequest, Path]:
    metadata = json.loads((folder / "kernel-metadata.json").read_text())
    code_file = (folder / metadata["code_file"]).resolve()
    if code_file.suffix != ".ipynb" or not code_file.is_file():
        raise ValueError(f"Notebook not found: {code_file}")
    if metadata["kernel_type"] != "notebook" or metadata["language"] != "python":
        raise ValueError("This uploader accepts Python notebooks only.")
    if not metadata_bool(metadata, "is_private"):
        raise ValueError("Refusing to upload a public diagnostic notebook.")

    notebook = json.loads(code_file.read_text())
    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        raise ValueError("The notebook is not a version-4 Jupyter notebook.")
    for cell in notebook["cells"]:
        if isinstance(cell.get("source"), list):
            cell["source"] = "".join(cell["source"])
        if cell.get("cell_type") == "code":
            if cell.get("outputs") or cell.get("execution_count") is not None:
                raise ValueError("Clear notebook outputs and execution counts before uploading.")

    request = ApiSaveKernelRequest()
    request.slug = metadata["id"]
    request.new_title = metadata["title"]
    request.text = json.dumps(notebook)
    request.language = "python"
    request.kernel_type = "notebook"
    request.is_private = True
    request.enable_gpu = metadata_bool(metadata, "enable_gpu")
    request.enable_internet = metadata_bool(metadata, "enable_internet")
    request.dataset_data_sources = metadata.get("dataset_sources", [])
    request.competition_data_sources = metadata.get("competition_sources", [])
    request.kernel_data_sources = metadata.get("kernel_sources", [])
    request.model_data_sources = metadata.get("model_sources", [])
    request.kernel_execution_type = KernelExecutionType.QUICK_SAVE
    return request, code_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Folder containing kernel-metadata.json")
    parser.add_argument("--dry-run", action="store_true", help="Validate locally without contacting Kaggle")
    args = parser.parse_args()
    request, code_file = build_request(args.folder)
    if args.dry_run:
        print(f"Ready for QUICK_SAVE: {request.slug} from {code_file}")
        return

    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.save_kernel(request)
    if response.error:
        raise RuntimeError(f"Kaggle rejected Quick Save: {response.error}")
    print(f"Quick Save version {response.version_number}: {response.url}")


if __name__ == "__main__":
    main()
