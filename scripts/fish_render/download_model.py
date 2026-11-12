"""Download the BarramundiFish model (Khronos glTF Sample Assets, CC0 1.0) next to this script.

    python scripts/fish_render/download_model.py

The file (12.5 MB) is not kept in the repository. Its size and SHA-256 are checked against the copy used for
the figures of the paper.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = ("https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/"
       "Models/BarramundiFish/glTF-Binary/BarramundiFish.glb")
SIZE = 12488144
SHA256 = "ecc3bafb6b00f2c8b810863c388e3768a7b7ea0d0335e8cb8c574c266e571f4a"
OUT = Path(__file__).resolve().parent / "BarramundiFish.glb"


def main():
    if not OUT.exists():
        print("downloading", URL)
        urllib.request.urlretrieve(URL, OUT)
    data = OUT.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != SIZE or digest != SHA256:
        print(f"warning: {OUT.name} differs from the copy used for the paper "
              f"(size {len(data)}, sha256 {digest})", file=sys.stderr)
    else:
        print(OUT, "ok")


if __name__ == "__main__":
    main()
