"""Convert a Roboflow "folder" export of the bottle-anomaly project to the layout dataset.py reads.

Roboflow export (classification "folder" format)::

    <export>/train/normal/*.jpg
    <export>/valid/{normal,anomalous}/*.jpg
    <export>/test/{normal,anomalous}/*.jpg

Output (MVTec AD layout plus a ``valid`` split)::

    <out>/<category>/train/good/*.png
    <out>/<category>/valid/good/*.png               normal images used for threshold calibration
    <out>/<category>/valid/<defect>/*.png
    <out>/<category>/test/good/*.png
    <out>/<category>/test/<defect>/*.png
    <out>/<category>/ground_truth/<defect>/<stem>_mask.png

Roboflow only knows normal/anomalous and renames files to
``<original>_png.rf.<hash>.jpg``.  The original name is recovered from that
prefix, and ``manifest.csv`` (written by prepare_roboflow_batches.py) gives
back the defect type and the original MVTec mask.  Each mask is resized to the
exported image size so pixel-level metrics still work.

Usage:
    python scripts/convert_roboflow_export.py data/roboflow_export/v2 data/roboflow_v2
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

from PIL import Image

PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT / "data" / "roboflow_upload" / "bottle" / "manifest.csv"
RF_NAME = re.compile(r"^(?P<stem>.+)_png\.rf\.(?P<hash>[0-9a-f]+)$")


def original_stem(path: Path) -> tuple[str, str]:
    match = RF_NAME.match(path.stem)
    if not match:
        raise ValueError(f"Unexpected Roboflow file name: {path.name}")
    return match["stem"], match["hash"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export", type=Path, help="unzipped Roboflow folder export")
    ap.add_argument("out", type=Path, help="dataset root to create (set MVTEC_ROOT to it)")
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = {Path(r["file"]).stem: r for r in csv.DictReader(f)}

    dest = args.out / args.category
    if dest.exists():
        shutil.rmtree(dest)

    counts: dict[str, int] = {}
    seen: dict[tuple[str, str], int] = {}
    for split in ("train", "valid", "test"):
        for cls_dir in sorted((args.export / split).glob("*")):
            if not cls_dir.is_dir():
                continue
            for src in sorted(cls_dir.glob("*.jpg")):
                stem, _ = original_stem(src)
                row = manifest[stem]
                defect = row["label"]  # "good" or the MVTec defect type
                if (defect == "good") != (cls_dir.name == "normal"):
                    raise ValueError(f"Label mismatch for {src.name}: Roboflow {cls_dir.name}, manifest {defect}")
                if split == "train" and defect != "good":
                    raise ValueError(f"Defective image in train split: {src.name}")

                # Augmented versions give several outputs per original image.
                n = seen.get((split, stem), 0)
                seen[(split, stem)] = n + 1
                name = stem if n == 0 else f"{stem}_aug{n}"

                out_dir = dest / split / defect
                out_dir.mkdir(parents=True, exist_ok=True)
                with Image.open(src) as image:
                    image.convert("RGB").save(out_dir / f"{name}.png")
                    size = image.size

                if defect != "good":
                    mask_dir = dest / "ground_truth" / defect
                    mask_dir.mkdir(parents=True, exist_ok=True)
                    with Image.open(row["mask"]) as mask:
                        mask.convert("L").resize(size, Image.NEAREST).save(mask_dir / f"{name}_mask.png")

                key = f"{split}/{defect}"
                counts[key] = counts.get(key, 0) + 1

    for key in sorted(counts):
        print(f"{key:28s} {counts[key]:4d}")
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
