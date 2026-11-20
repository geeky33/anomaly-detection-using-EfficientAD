"""Build, calibrate and evaluate the model on one MVTec AD category.

Usage::

    python evaluation.py --category bottle            # build (if needed) + evaluate
    python evaluation.py --category bottle --rebuild  # force a new memory bank

Protocol
--------
* Memory bank: 90 % of ``train/good`` (seeded split).
* Calibration: the other 10 % of ``train/good`` - normal images the model has
  not seen.  Test images / masks are NOT used to choose any threshold.
* Evaluation: every image of the official test split.
  - Image level (detection): label 0 = ``test/good``, 1 = any defect folder.
  - Pixel level (localisation): every anomaly map and ground-truth mask is
    resized to a common 256x256 grid (bilinear for maps, nearest for masks);
    normal images contribute all-zero masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import resource
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

import dataset as ds
from model import PatchCoreLite

PROJECT = Path(__file__).resolve().parent
EVAL_GRID = 256


def artifacts_dir(category: str) -> Path:
    return PROJECT / "artifacts" / ds.output_key(category)


def results_dir(category: str, model_key: str = "patchcore_r18") -> Path:
    return PROJECT / "results" / ds.output_key(category) / model_key


# ----------------------------------------------------------------------- metrics
def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Precision, recall, F1, specificity, IoU and the confusion matrix."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        "iou": float(tp / (tp + fp + fn)) if tp + fp + fn else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# ---------------------------------------------------------------------- pipeline
def build_model(category: str, root: Path | None = None, progress=None, log=print) -> PatchCoreLite:
    """Memory bank from the normal training images + threshold calibration on normal validation images.

    A dataset with its own ``valid/good`` split (Roboflow export) uses it for
    calibration and all of ``train/good`` for the memory bank.  Official MVTec
    AD has no validation split, so 10 % of ``train/good`` is held out instead.
    """
    root = root or ds.find_dataset_root()
    val = ds.valid_samples(root, category)
    if val:
        train = ds.train_samples(root, category)
        source = f"valid/good split ({len(val)} images); image = max validation score, pixel = 99.5th percentile"
    else:
        train, val = ds.split_train_validation(ds.train_samples(root, category))
        source = "10 % of train/good (seed 0); image = max validation score, pixel = 99.5th percentile"
    log(f"Building memory bank from {len(train)} normal images; {len(val)} normal images for calibration.")
    model = PatchCoreLite()
    info = model.fit([s.path for s in train], ds.load_rgb, progress)
    log(f"Memory bank: {info['memory_bank_size']} of {info['patch_vectors']} patch vectors ({info['fit_seconds']} s).")
    calibration = model.calibrate([s.path for s in val], ds.load_rgb)
    log(f"Thresholds: image {calibration.image_threshold:.4f}, pixel {calibration.pixel_threshold:.4f}.")
    model.info.update({"category": category, "validation_images": [s.path.name for s in val],
                       "calibration_source": source, "dataset": ds.dataset_tag() or "mvtec_ad"})
    model.save(artifacts_dir(category))
    return model


def load_or_build(category: str, rebuild: bool = False, log=print) -> PatchCoreLite:
    directory = artifacts_dir(category)
    if not rebuild and (directory / "memory_bank.pt").is_file():
        return PatchCoreLite.load(directory)
    return build_model(category, log=log)


def evaluate(model, category: str, root: Path | None = None, progress=None, limit: int | None = None,
             model_key: str = "patchcore_r18") -> dict:
    """Run every official test image through a model and compute the metrics.

    ``model`` is anything with ``predict(image)`` and ``calibration`` (see
    ``model_registry``), so every model is scored by exactly this code on
    exactly the same images.
    """
    root = root or ds.find_dataset_root()
    samples = ds.test_samples(root, category)[:limit]
    cal = model.calibration
    labels, scores, latencies, rows = [], [], [], []
    masks = np.zeros((len(samples), EVAL_GRID, EVAL_GRID), dtype=np.uint8)
    maps = np.zeros((len(samples), EVAL_GRID, EVAL_GRID), dtype=np.float32)

    model.predict(ds.load_rgb(samples[0].path))  # warm-up, not timed
    for index, sample in enumerate(samples):
        image = ds.load_rgb(sample.path)
        result = model.predict(image)
        gt = ds.load_mask(sample, image.shape[:2])
        masks[index] = cv2.resize(gt, (EVAL_GRID, EVAL_GRID), interpolation=cv2.INTER_NEAREST)
        maps[index] = cv2.resize(result["anomaly_map"], (EVAL_GRID, EVAL_GRID), interpolation=cv2.INTER_LINEAR)
        labels.append(sample.label)
        scores.append(result["score"])
        latencies.append(result["latency_ms"])
        pred_mask = maps[index] >= cal.pixel_threshold
        inter = int((pred_mask & (masks[index] > 0)).sum())
        union = int((pred_mask | (masks[index] > 0)).sum())
        rows.append({
            "image": sample.name, "label": sample.label, "defect_type": sample.defect_type,
            "score": round(result["score"], 5), "predicted": int(result["score"] >= cal.image_threshold),
            "latency_ms": round(result["latency_ms"], 2),
            # IoU only for defective images; undefined without a ground-truth region.
            "iou": round(inter / union, 4) if sample.label == 1 else None,
            "gt_pixels": int(masks[index].sum()), "pred_pixels": int(pred_mask.sum()),
        })
        if progress:
            progress((index + 1) / len(samples))

    y = np.array(labels)
    s = np.array(scores)
    image = binary_metrics(y, (s >= cal.image_threshold).astype(int))
    image["auroc"] = float(roc_auc_score(y, s)) if len(set(labels)) == 2 else float("nan")
    fpr, tpr, _ = roc_curve(y, s)

    flat_gt = masks.ravel()
    flat_map = maps.ravel()
    pixel = binary_metrics(flat_gt, (flat_map >= cal.pixel_threshold).astype(np.uint8))
    pixel["auroc"] = float(roc_auc_score(flat_gt, flat_map)) if flat_gt.any() else float("nan")
    defect_rows = [r for r in rows if r["label"] == 1]
    pixel["mean_iou_defective_images"] = float(np.mean([r["iou"] for r in defect_rows])) if defect_rows else float("nan")

    lat = np.array(latencies)
    metrics = {
        "category": category,
        "num_test_images": len(samples),
        "num_normal": int((y == 0).sum()),
        "num_defective": int((y == 1).sum()),
        "image_level": image,
        "pixel_level": pixel,
        "thresholds": {
            "image": cal.image_threshold, "pixel": cal.pixel_threshold,
            "source": f"{cal.num_validation_images} held-out normal training images (no test data)",
        },
        "latency_ms": {"mean": float(lat.mean()), "median": float(np.median(lat)),
                       "p95": float(np.percentile(lat, 95)), "fps": float(1000 / lat.mean())},
        "pixel_eval_grid": f"{EVAL_GRID}x{EVAL_GRID}",
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "model_key": model_key,
        "model": model.info,
        "device": model.device,
        # Peak resident memory of the evaluating process (meaningful for CLI runs,
        # where the process holds only this model).
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "torch_threads": torch.get_num_threads(),
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = results_dir(category, model_key)
    out.mkdir(parents=True, exist_ok=True)
    if limit is None:
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        with open(out / "per_image.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return metrics


def load_metrics(category: str, model_key: str = "patchcore_r18") -> dict | None:
    path = results_dir(category, model_key) / "metrics.json"
    return json.loads(path.read_text()) if path.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--model", default="patchcore_r18",
                        help="patchcore_r18 (built here) or a main-platform checkpoint: "
                             "patchcore_wrn50, efficientad_s, efficientad_m")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="smoke test: first N test images, not saved")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    if args.model == "patchcore_r18":
        model = load_or_build(args.category, args.rebuild)
    else:
        import model_registry

        model = model_registry.load(args.model, args.category)
    metrics = evaluate(model, args.category, limit=args.limit, model_key=args.model)
    i, p, lat = metrics["image_level"], metrics["pixel_level"], metrics["latency_ms"]
    print(f"\n{args.category}: {metrics['num_test_images']} test images "
          f"({metrics['num_normal']} normal, {metrics['num_defective']} defective)")
    print(f"Image  AUROC {i['auroc']:.4f}  P {i['precision']:.4f}  R {i['recall']:.4f}  F1 {i['f1']:.4f}  {i['confusion_matrix']}")
    print(f"Pixel  AUROC {p['auroc']:.4f}  P {p['precision']:.4f}  R {p['recall']:.4f}  F1 {p['f1']:.4f}  "
          f"IoU {p['iou']:.4f}  mean IoU (defective) {p['mean_iou_defective_images']:.4f}")
    print(f"Latency mean {lat['mean']:.1f} ms  median {lat['median']:.1f}  p95 {lat['p95']:.1f}  FPS {lat['fps']:.1f}")


if __name__ == "__main__":
    main()
