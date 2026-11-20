"""Heatmaps, overlays and evaluation plots.

Heatmap colour scale: fixed per model, not per image.  The low end is the
median normal anomaly score level (0 colour) and the top is twice the pixel
threshold, so a defect-free image stays blue instead of being stretched to
look "hot" by per-image min-max normalisation.
"""

from __future__ import annotations

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def normalise(anomaly_map: np.ndarray, pixel_threshold: float) -> np.ndarray:
    """Map scores to [0, 1]; the pixel threshold lands at 0.5."""
    return np.clip(anomaly_map / (2.0 * pixel_threshold), 0.0, 1.0)


def heatmap(anomaly_map: np.ndarray, pixel_threshold: float) -> np.ndarray:
    """RGB uint8 JET heatmap."""
    values = (normalise(anomaly_map, pixel_threshold) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(values, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def overlay(image: np.ndarray, anomaly_map: np.ndarray, mask: np.ndarray, pixel_threshold: float,
            gt_mask: np.ndarray | None = None, alpha: float = 0.45) -> np.ndarray:
    """Image + heatmap; predicted mask outlined in red, ground truth (optional) in green."""
    blended = cv2.addWeighted(image, 1 - alpha, heatmap(anomaly_map, pixel_threshold), alpha, 0)
    thickness = max(2, image.shape[0] // 200)
    if gt_mask is not None and gt_mask.any():
        contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (0, 255, 0), thickness)
    if mask.any():
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (255, 0, 0), thickness)
    return blended


def mask_image(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8) * 255)


def confusion_figure(cm: dict, title: str):
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:,}", ha="center", va="center",
                color="white" if value > matrix.max() / 2 else "black", fontsize=11)
    ax.set_xticks([0, 1], ["Normal", "Defective"])
    ax.set_yticks([0, 1], ["Normal", "Defective"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def roc_figure(fpr: list, tpr: list, auroc: float):
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"AUROC = {auroc:.3f}")
    ax.plot([0, 1], [0, 1], color="#9ca3af", lw=1, ls="--", label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Image-level ROC", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def score_histogram(rows: list[dict], threshold: float):
    """Distribution of image scores for normal vs defective test images."""
    normal = [r["score"] for r in rows if int(r["label"]) == 0]
    defective = [r["score"] for r in rows if int(r["label"]) == 1]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    bins = np.linspace(min(normal + defective), max(normal + defective), 30)
    ax.hist(normal, bins=bins, alpha=0.7, color="#16a34a", label=f"normal (n={len(normal)})")
    ax.hist(defective, bins=bins, alpha=0.7, color="#dc2626", label=f"defective (n={len(defective)})")
    ax.axvline(threshold, color="black", ls="--", lw=1.5, label="threshold (from validation)")
    ax.set_xlabel("Image anomaly score")
    ax.set_ylabel("Images")
    ax.legend(fontsize=8)
    ax.set_title("Test-set score distribution", fontsize=10)
    fig.tight_layout()
    return fig


def figure_to_png(fig) -> bytes:
    import io

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130)
    plt.close(fig)
    return buffer.getvalue()


# ------------------------------------------------------------ region analysis
def regions(mask: np.ndarray, min_area_ratio: float = 0.0005) -> list[dict]:
    """Connected regions of a binary mask (8-connectivity) with bounding boxes.

    Regions smaller than ``min_area_ratio`` of the image are ignored as specks.
    """
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    min_area = min_area_ratio * mask.size
    found = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < min_area:
            continue
        found.append({"x0": int(x), "y0": int(y), "x1": int(x + w), "y1": int(y + h), "area_px": int(area),
                      "area_pct": 100.0 * area / mask.size,
                      "centroid": (float(centroids[index][0]), float(centroids[index][1]))})
    return sorted(found, key=lambda r: -r["area_px"])


def mask_agreement(pred: np.ndarray, gt: np.ndarray) -> dict:
    """IoU and Dice between a predicted and a ground-truth mask (same shape)."""
    p, g = pred.astype(bool), gt.astype(bool)
    inter, union = int((p & g).sum()), int((p | g).sum())
    total = int(p.sum() + g.sum())
    return {"iou": inter / union if union else None, "dice": 2 * inter / total if total else None,
            "tp_px": inter, "fp_px": int((p & ~g).sum()), "fn_px": int((~p & g).sum())}


def gt_comparison(image: np.ndarray, pred: np.ndarray, gt: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Prediction vs. ground truth: red = predicted only, green = ground truth only, yellow = overlap."""
    p, g = pred.astype(bool), gt.astype(bool)
    colour = np.zeros_like(image)
    colour[p & ~g] = (239, 68, 68)
    colour[~p & g] = (34, 197, 94)
    colour[p & g] = (250, 204, 21)
    painted = p | g
    out = image.copy()
    out[painted] = (image[painted] * (1 - alpha) + colour[painted] * alpha).astype(np.uint8)
    return out


def coloured_mask(mask: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
    """Binary mask as an RGB image (mask pixels in ``colour`` on dark background)."""
    out = np.full((*mask.shape, 3), (10, 15, 28), dtype=np.uint8)
    out[mask.astype(bool)] = colour
    return out
