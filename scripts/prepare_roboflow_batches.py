"""Build the 200-image, 80/10/10 Roboflow upload set from one MVTec AD category.

Output (default: data/roboflow_upload/<category>/):

    train_normal/      160 good         -> Roboflow split: Training
    valid_normal/       10 good         -> Validation
    valid_anomalous/    10 defective    -> Validation
    test_normal/        10 good         -> Testing
    test_anomalous/     10 defective    -> Testing
    manifest.csv        new file -> original MVTec image + mask (for converting back)

Anomalous images never enter training, as PatchCore requires. Defects are spread
evenly across defect types and are disjoint between valid and test.

Usage:
    python scripts/prepare_roboflow_batches.py --category bottle
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

DEFAULT_SRC = Path.home() / "fun" / "industrial-anomaly-detection" / "data" / "mvtec_ad"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "roboflow_upload"


def pngs(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.png"))


def round_robin(groups: dict[str, list[Path]], n: int) -> list[tuple[str, Path]]:
    """Take n items cycling through defect types so each is represented."""
    picked, i = [], 0
    pools = {k: list(v) for k, v in groups.items()}
    keys = sorted(pools)
    while len(picked) < n:
        k = keys[i % len(keys)]
        if pools[k]:
            picked.append((k, pools[k].pop(0)))
        elif not any(pools.values()):
            raise ValueError(f"Not enough defective images for {n}")
        i += 1
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-train", type=int, default=160)
    ap.add_argument("--n-eval", type=int, default=10, help="per normal/anomalous val/test batch")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cat = args.src / args.category
    out = args.out / args.category
    if out.exists():
        shutil.rmtree(out)

    train_good = pngs(cat / "train" / "good")
    test_good = pngs(cat / "test" / "good")
    rng.shuffle(train_good)
    rng.shuffle(test_good)

    defects = {d.name: pngs(d) for d in sorted((cat / "test").iterdir()) if d.is_dir() and d.name != "good"}
    for v in defects.values():
        rng.shuffle(v)

    n, k = args.n_train, args.n_eval
    if len(train_good) < n + k or len(test_good) < k:
        raise SystemExit("Not enough good images for the requested sizes")

    anomalous = round_robin(defects, 2 * k)
    batches: dict[str, list[tuple[str, Path]]] = {
        "train_normal": [("good", p) for p in train_good[:n]],
        # MVTec's test/good is small, so validation normals come from unused train/good.
        "valid_normal": [("good", p) for p in train_good[n : n + k]],
        "test_normal": [("good", p) for p in test_good[:k]],
        "valid_anomalous": anomalous[0::2],
        "test_anomalous": anomalous[1::2],
    }

    rows = []
    for batch, items in batches.items():
        (out / batch).mkdir(parents=True)
        for label, src in items:
            # MVTec reuses names like 000.png in every folder, so make them unique.
            origin = src.parent.parent.name  # train / test
            new_name = f"{args.category}_{origin}_{label}_{src.stem}.png"
            shutil.copy2(src, out / batch / new_name)
            mask = cat / "ground_truth" / label / f"{src.stem}_mask.png"
            rows.append({
                "batch": batch,
                "file": new_name,
                "label": label,
                "status": "normal" if label == "good" else "anomalous",
                "source": str(src),
                "mask": str(mask) if mask.exists() else "",
            })

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    total = len(rows)
    for batch, items in batches.items():
        print(f"{batch:16s} {len(items):4d}")
    print(f"{'total':16s} {total:4d}  -> {out}")


if __name__ == "__main__":
    main()
