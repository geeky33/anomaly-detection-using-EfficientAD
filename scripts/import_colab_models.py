"""Import the checkpoints trained in Colab (roboflow_v2_models.zip) and evaluate them.

Usage:
    python scripts/import_colab_models.py colab/roboflow_v2_models.zip [--dataset roboflow_v2] [--no-eval]

Unpacks into artifacts/<dataset>/main_models/<architecture>/bottle/, where the app
looks for main-platform checkpoints when MVTEC_ROOT points at data/<dataset>, then
runs evaluation.py for every imported model on the same test images as PatchCore ResNet18.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
KEYS = {"patchcore": "patchcore_wrn50", "efficientad_s": "efficientad_s", "efficientad_m": "efficientad_m"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip", type=Path)
    ap.add_argument("--dataset", default="roboflow_v2")
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    dest = PROJECT / "artifacts" / args.dataset / "main_models"
    with zipfile.ZipFile(args.zip) as zf:
        names = zf.namelist()
        for arch in KEYS:
            member = f"{arch}/{args.category}/checkpoint.ckpt"
            if not any(n.lstrip("./") == member for n in names):
                print(f"skip {arch}: not in zip")
                continue
            target = dest / arch / args.category
            if target.exists():
                shutil.rmtree(target)
        zf.extractall(dest)

    imported = []
    for arch, key in KEYS.items():
        meta_file = dest / arch / args.category / "metadata.json"
        if not (dest / arch / args.category / "checkpoint.ckpt").is_file():
            continue
        meta = json.loads(meta_file.read_text())
        iters = meta.get("training_config", {}).get("total_iters")
        print(f"imported {key:16s} protocol={meta.get('protocol')} iters={iters} device={meta.get('device')}")
        imported.append(key)

    if args.no_eval:
        return
    env = dict(os.environ, MVTEC_ROOT=str(PROJECT / "data" / args.dataset))
    for key in imported:
        print(f"\n=== evaluating {key}")
        subprocess.run([sys.executable, "evaluation.py", "--category", args.category, "--model", key],
                       cwd=PROJECT, env=env, check=True)


if __name__ == "__main__":
    main()
