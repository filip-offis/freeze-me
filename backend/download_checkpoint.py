import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

CHECKPOINTS = {
    "large": "sam2.1_hiera_large.pt",
    "b_plus": "sam2.1_hiera_base_plus.pt",
    "small": "sam2.1_hiera_small.pt",
    "tiny": "sam2.1_hiera_tiny.pt",
}

BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"


def main():
    version = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SAM_VERSION", "tiny")).lower()
    if version not in CHECKPOINTS:
        valid_versions = ", ".join(CHECKPOINTS)
        raise SystemExit(f"Unsupported SAM version '{version}'. Use one of: {valid_versions}.")

    checkpoints_dir = Path(__file__).resolve().parent / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    filename = CHECKPOINTS[version]
    target_path = checkpoints_dir / filename
    if target_path.exists():
        print(f"Checkpoint already exists: {target_path}")
        return

    source_url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename} to {target_path}...")
    urlretrieve(source_url, target_path)
    print("Download complete.")


if __name__ == "__main__":
    main()
