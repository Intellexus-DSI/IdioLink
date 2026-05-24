"""
Download IdioLink dataset from HuggingFace into the local data/ directory.

Populates:
    data/train/indexes.json  data/train/queries.json
    data/val/indexes.json    data/val/queries.json
    data/test/indexes.json   data/test/queries.json

Usage:
    python download_data.py
"""
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "Intellexus/IdioLink"

HF_FILES = [
    "data/train/indexes.json",
    "data/train/queries.json",
    "data/val/indexes.json",
    "data/val/queries.json",
    "data/test/indexes.json",
    "data/test/queries.json",
]

def main():
    root = Path(__file__).parent

    for hf_path in HF_FILES:
        print(f"Downloading {hf_path} ...", end=" ", flush=True)
        hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            filename=hf_path,
            local_dir=root,
        )
        print("done")

    print("\nAll files downloaded to data/. You can now run experiments.")


if __name__ == "__main__":
    main()
