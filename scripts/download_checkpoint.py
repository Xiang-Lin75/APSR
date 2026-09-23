"""Download the published E102 inference weights with SHA256 verification."""

import argparse
from pathlib import Path
from urllib.request import urlopen
from apsr.utils.runtime import sha256

FILENAME = "apsr_wsj0_2mix_r4_seed43_e102.pt"
SHA256 = "4cafe7f02bdd5e0a0e6025e845067ae341d26bcf53fdf64a0b54018db6944762"
URL = (
    "https://github.com/Xiang-Lin75/APSR/releases/download/wsj0-2mix-r4-seed43-e102/"
    + FILENAME
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="checkpoints/" + FILENAME)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        if sha256(path) != SHA256:
            raise ValueError("Existing destination has a different checksum")
        print(f"Already verified: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    with urlopen(URL, timeout=60) as response, temporary.open("wb") as handle:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            handle.write(chunk)
    if sha256(temporary) != SHA256:
        temporary.unlink()
        raise ValueError("Downloaded checkpoint failed SHA256 validation")
    temporary.replace(path)
    print(f"Verified: {path}")


if __name__ == "__main__":
    main()
