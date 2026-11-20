"""PDF report generation (reportlab) from the saved evaluation results."""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

TITLE = ("Industrial Anomaly Detection and Localization using EfficientAD "
         "(with PatchCore baseline) on MVTec AD")

LIMITATIONS = [
    "EfficientAD-S and EfficientAD-M were trained with a reduced protocol (2,500 and 1,500 "
    "iterations on a Colab T4 GPU) instead of the paper's 70,000, because CPU training would take "
    "~32 h (S) and ~225 h (M). They are therefore weaker than the published EfficientAD models.",
    "PatchCore ResNet18 is simplified: lighter backbone, no per-layer 1024-d projection, no "
    "image-score re-weighting (paper Eq. 7). Results are not a reproduction of the published numbers.",
    "The Roboflow dataset is a 200-image subset of MVTec AD bottle (test split: 10 normal + 10 "
    "defective images), so metrics have wide uncertainty; only one product category was evaluated.",
    "Thresholds are calibrated on a small set of held-out normal training images, so the "
    "precision/recall trade-off depends on that set; AUROC is threshold-independent.",
    "Pixel-level metrics are computed on a 256x256 grid, not at the original resolution.",
    "CPU-only latency on a shared laptop; numbers vary with system load.",
    "Uploaded images that do not resemble the training product are still scored and will "
    "usually be called defective - the model only knows what normal looks like.",
]


def _array_to_flowable(array: np.ndarray, width_cm: float) -> RLImage:
    buffer = io.BytesIO()
    image = Image.fromarray(array)
    image.thumbnail((600, 600))  # full 900x900 PNGs make an 8 MB report
    image.save(buffer, format="JPEG", quality=88)
    buffer.seek(0)
    h, w = array.shape[:2]
    return RLImage(buffer, width=width_cm * cm, height=width_cm * cm * h / w)


def _table(rows: list[list[str]], widths: list[float]) -> Table:
    table = Table(rows, colWidths=[w * cm for w in widths])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def build_pdf(path: Path, dataset_info: dict, metrics: dict | None, model_info: dict,
              samples: list[dict], figures: list[bytes] | None = None, comparison: list[dict] | None = None,
              unavailable: list[str] | None = None) -> Path:
    """Write the report.

    ``samples``: dicts with ``name``, ``score``, ``prediction``, ``latency_ms``,
    ``image``, ``heatmap``, ``overlay`` (RGB arrays).
    """
    styles = getSampleStyleSheet()
    h1, h2, body = styles["Title"], styles["Heading2"], styles["BodyText"]
    story = [Paragraph(TITLE, h1),
             Paragraph(f"Generated {time.strftime('%Y-%m-%d %H:%M')}", body), Spacer(1, 10)]

    story += [Paragraph("1. Project", h2), Paragraph(
        "Unsupervised visual inspection: the model learns only from defect-free product images "
        "and flags images (detection) and regions (localisation) that differ from them.", body)]

    default_description = (
        "PatchCore-style nearest-neighbour model. A frozen ImageNet-pretrained ResNet18 extracts "
        "layer2 and layer3 features; each feature map is averaged over a 3x3 neighbourhood and the "
        "two layers are concatenated into one 384-d vector per image patch (28x28 patches for a "
        "224x224 input). The vectors of all normal training images form a memory bank, reduced "
        "with greedy k-center coreset selection. A test patch's anomaly score is its Euclidean "
        "distance to the nearest memory-bank vector; the image score is the maximum patch score.")
    story += [Paragraph(f"2. Model - {model_info.get('label', 'PatchCore · ResNet18')}", h2),
              Paragraph(model_info.get("description", default_description), body),
        Spacer(1, 6),
        _table([["Setting", "Value"],
                ["Model", model_info.get("label", "PatchCore · ResNet18")],
                ["Input size", f"{model_info.get('image_size', 224)} x {model_info.get('image_size', 224)}"],
                ["Memory bank", f"{model_info.get('memory_bank_size', '-'):,} of "
                                f"{model_info.get('patch_vectors', '-'):,} patch vectors"
                 if isinstance(model_info.get('memory_bank_size'), int) else "-"],
                ["Training images (memory bank)", str(model_info.get("train_images", "-"))],
                ["Build / training time", f"{model_info.get('fit_seconds') or '-'} s"]], [6, 10])]

    story += [Paragraph("3. Dataset", h2), _table(
        [["Item", "Value"],
         ["Dataset", dataset_info.get("dataset_name", "MVTec AD (official release)")],
         ["Source", Paragraph(dataset_info.get("dataset_source", "-"), body)],
         ["Category", dataset_info.get("category", "-")],
         ["Training images (normal)", str(dataset_info.get("train_images", "-"))],
         ["Validation images", str(dataset_info.get("valid_images", "-"))],
         ["Test images", f"{dataset_info.get('test_images')} ({dataset_info.get('test_normal')} normal, "
                         f"{dataset_info.get('test_defective')} defective)"],
         ["Defect types", ", ".join(dataset_info.get("defect_types", []))],
         ["Ground-truth masks", str(dataset_info.get("masks", "-"))]], [6, 10])]

    story += [Paragraph(f"4. Evaluation ({dataset_info.get('split', 'official test split')})", h2)]
    if metrics:
        i, p, lat = metrics["image_level"], metrics["pixel_level"], metrics["latency_ms"]
        story += [_table(
            [["Metric", "Image level (detection)", "Pixel level (localisation)"],
             ["AUROC", f"{i['auroc']:.4f}", f"{p['auroc']:.4f}"],
             ["Precision", f"{i['precision']:.4f}", f"{p['precision']:.4f}"],
             ["Recall", f"{i['recall']:.4f}", f"{p['recall']:.4f}"],
             ["F1 score", f"{i['f1']:.4f}", f"{p['f1']:.4f}"],
             ["IoU", "-", f"{p['iou']:.4f} (mean per defective image {p['mean_iou_defective_images']:.4f})"],
             ["Confusion (TN/FP/FN/TP)",
              "{tn}/{fp}/{fn}/{tp}".format(**i["confusion_matrix"]),
              "{tn:,}/{fp:,}/{fn:,}/{tp:,}".format(**p["confusion_matrix"])]], [4.5, 4.5, 7.5]),
            Spacer(1, 4),
            Paragraph(f"Thresholds from {metrics['thresholds']['source']}: image "
                      f"{metrics['thresholds']['image']:.4f}, pixel {metrics['thresholds']['pixel']:.4f}. "
                      f"Pixel metrics on a {metrics['pixel_eval_grid']} grid. Latency (CPU, "
                      f"{metrics.get('torch_threads')} threads): mean {lat['mean']:.1f} ms, median "
                      f"{lat['median']:.1f} ms, p95 {lat['p95']:.1f} ms ({lat['fps']:.1f} FPS).", body)]
        for png in figures or []:
            story += [Spacer(1, 4), RLImage(io.BytesIO(png), width=8 * cm, height=6.4 * cm)]
    else:
        story += [Paragraph("No evaluation has been run yet.", body)]

    if comparison:
        story += [Paragraph("Model comparison (same test images, same evaluation code)", h2), _table(
            [["Model", "Img AUROC", "Px AUROC", "Img F1", "Px F1", "IoU", "Latency ms"]]
            + [[r["label"], f"{r['image_auroc']:.4f}", f"{r['pixel_auroc']:.4f}", f"{r['img_f1']:.3f}",
                f"{r['px_f1']:.3f}", f"{r['px_iou']:.3f}", f"{r['lat_mean']:.0f}"] for r in comparison],
            [4.2, 2.0, 2.0, 1.8, 1.8, 1.6, 2.1])]
    if unavailable:
        story += [Spacer(1, 4), Paragraph("Not compared (no trained checkpoint - no results are shown or simulated):", body)]
        story += [Paragraph(f"&bull; {item}", body) for item in unavailable]

    if samples:
        story += [PageBreak(), Paragraph("5. Sample inspection results", h2)]
        for sample in samples:
            story += [Paragraph(
                f"<b>{sample['name']}</b> - score {sample['score']:.3f}, predicted "
                f"<b>{sample['prediction']}</b>, {sample['latency_ms']:.0f} ms", body),
                Table([[_array_to_flowable(sample["image"], 5), _array_to_flowable(sample["heatmap"], 5),
                        _array_to_flowable(sample["overlay"], 5)]]), Spacer(1, 6)]

    story += [Paragraph("6. Limitations", h2)]
    story += [Paragraph(f"&bull; {item}", body) for item in LIMITATIONS]

    path.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(path), pagesize=A4, title=TITLE,
                      leftMargin=1.8 * cm, rightMargin=1.8 * cm).build(story)
    return path
