"""Build Project_Presentation.pptx (19 slides) from the project's real files.

Every number comes from the saved results:
  results/roboflow_v{1,2}/bottle/<model>/metrics.json        evaluation metrics
  artifacts/roboflow_v{1,2}/bottle/model.json                 PatchCore ResNet18 build info
  artifacts/roboflow_v2/main_models/<arch>/bottle/metadata.json   EfficientAD / WRN-50 training config + loss history
  artifacts/roboflow_v2/main_models/colab_run.json            Colab GPU run times
  data/roboflow_v{1,2}/bottle/                                dataset counts (via dataset.py)
Screenshots: docs/screenshots/new_*.png (cropped, never stretched).

Run:  ~/miniconda3/bin/python make_presentation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.oxml.ns import qn  # noqa: E402
from pptx.util import Emu, Inches, Pt  # noqa: E402

import dataset as ds  # noqa: E402

PROJECT = Path(__file__).resolve().parent
ASSETS = PROJECT / "presentation_assets"
SHOTS = PROJECT / "docs" / "screenshots"
OUT = PROJECT / "Project_Presentation.pptx"
TOTAL = 19

# ------------------------------------------------------------------ identity
TITLE = "Industrial Anomaly Detection and Localization using EfficientAD (with PatchCore Baseline) on MVTec AD"
NAME = "Aarya Pandey"
REG = "231071003"
PROGRAM = "Final Year B.Tech · Computer Engineering"
DEPT = "Department of Computer Engineering & IT"
COLLEGE = "Veermata Jijabai Technological Institute (VJTI), Mumbai"
GUIDE = "Prof. Harshala Dalal"
YEAR = "Academic Year 2026–27"

# ------------------------------------------------------------------ theme
BG = RGBColor(0x0F, 0x17, 0x2A)
PANEL = RGBColor(0x1E, 0x29, 0x3B)
PANEL2 = RGBColor(0x17, 0x22, 0x36)
LINE = RGBColor(0x33, 0x41, 0x55)
INDIGO = RGBColor(0x63, 0x66, 0xF1)
TEAL = RGBColor(0x14, 0xB8, 0xA6)
AMBER = RGBColor(0xF5, 0x9E, 0x0B)
ROSE = RGBColor(0xF4, 0x3F, 0x5E)
WHITE = RGBColor(0xF1, 0xF5, 0xF9)
MUTED = RGBColor(0x94, 0xA3, 0xB8)
HEX = {"bg": "#0F172A", "panel": "#1E293B", "line": "#334155", "indigo": "#6366F1", "teal": "#14B8A6",
       "amber": "#F59E0B", "rose": "#F43F5E", "white": "#F1F5F9", "muted": "#94A3B8"}
FONT = "Calibri"

MODELS = [("efficientad_s", "EfficientAD-S"), ("efficientad_m", "EfficientAD-M"),
          ("patchcore_r18", "PatchCore R18"), ("patchcore_wrn50", "PatchCore WRN-50")]
MODEL_HEX = [HEX["indigo"], "#A5B4FC", HEX["teal"], HEX["amber"]]


# ================================================================== data
def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_data() -> dict:
    metrics = {k: load_json(PROJECT / "results" / "roboflow_v2" / "bottle" / k / "metrics.json") for k, _ in MODELS}
    v1_metrics = load_json(PROJECT / "results" / "roboflow_v1" / "bottle" / "patchcore_r18" / "metrics.json")
    r18 = {tag: load_json(PROJECT / "artifacts" / tag / "bottle" / "model.json") for tag in ("roboflow_v1", "roboflow_v2")}
    main = PROJECT / "artifacts" / "roboflow_v2" / "main_models"
    meta = {a: load_json(main / a / "bottle" / "metadata.json") for a in ("efficientad_s", "efficientad_m", "patchcore")}
    colab = load_json(main / "colab_run.json")
    splits = {}
    for tag in ("roboflow_v1", "roboflow_v2"):
        root = PROJECT / "data" / tag
        info = ds.validate_category(root, "bottle")
        defects = {}
        for split in ("valid", "test"):
            for d in sorted((root / "bottle" / split).iterdir()):
                defects.setdefault(split, {})[d.name] = len(list(d.glob("*.png")))
        splits[tag] = {"info": info, "per_split": defects}
    return {"metrics": metrics, "v1": v1_metrics, "r18": r18, "meta": meta, "colab": colab, "splits": splits}


# ================================================================== assets
def crop(src: str, box: tuple[int, int, int, int], name: str) -> Path:
    ASSETS.mkdir(exist_ok=True)
    out = ASSETS / name
    Image.open(SHOTS / src).convert("RGB").crop(box).save(out)
    return out


def v1_grid() -> Path:
    """Real v1 training images: two originals and their two Roboflow-augmented copies each (no resizing)."""
    folder = PROJECT / "data" / "roboflow_v1" / "bottle" / "train" / "good"
    rows = [[folder / f"bottle_train_good_{i:03d}{suffix}.png" for suffix in ("", "_aug1", "_aug2")] for i in (0, 1)]
    tile, gap = 224, 10
    grid = Image.new("RGB", (3 * tile + 2 * gap, 2 * tile + gap), HEX["bg"])
    for r, row in enumerate(rows):
        for c, path in enumerate(row):
            grid.paste(Image.open(path).convert("RGB"), (c * (tile + gap), r * (tile + gap)))
    out = ASSETS / "v1_augmented_grid.png"
    grid.save(out)
    return out


def style_axes(ax) -> None:
    ax.set_facecolor(HEX["bg"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(HEX["line"])
    ax.tick_params(colors=HEX["white"], labelsize=12)
    ax.yaxis.grid(True, color=HEX["line"], linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def fig_save(fig, name: str) -> Path:
    out = ASSETS / name
    fig.savefig(out, dpi=200, facecolor=HEX["bg"], bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out


def chart_metrics(data: dict) -> Path:
    names = ["Accuracy", "Precision", "Recall", "F1", "Image\nAUROC", "Pixel\nAUROC", "Pixel\nIoU"]
    fig, ax = plt.subplots(figsize=(8.6, 4.4), facecolor=HEX["bg"])
    style_axes(ax)
    width = 0.2
    for i, (key, label) in enumerate(MODELS):
        m = data["metrics"][key]
        il, pl = m["image_level"], m["pixel_level"]
        values = [il["accuracy"], il["precision"], il["recall"], il["f1"], il["auroc"], pl["auroc"], pl["iou"]]
        xs = [x + (i - 1.5) * width for x in range(len(names))]
        ax.bar(xs, values, width * 0.92, label=label, color=MODEL_HEX[i])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, color=HEX["white"], fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score (0–1)", color=HEX["muted"], fontsize=12)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False, labelcolor=HEX["white"],
              fontsize=12)
    return fig_save(fig, "chart_metrics.png")


def chart_split(data: dict) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 2.9), facecolor=HEX["bg"])
    style_axes(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=HEX["line"], linewidth=0.6, alpha=0.7)
    rows = []
    for tag, label in (("roboflow_v2", "v2 (final)"), ("roboflow_v1", "v1 (augmented)")):
        info = data["splits"][tag]["info"]
        rows.append((label, info["train_images"], info["valid_images"], info["test_images"]))
    colors = [HEX["indigo"], HEX["teal"], HEX["amber"]]
    for y, (label, *counts) in enumerate(rows):
        total, left = sum(counts), 0.0
        small = []
        for count, colour, part in zip(counts, colors, ("train", "valid", "test")):
            pct = 100 * count / total
            ax.barh(y, pct, left=left, color=colour, height=0.55, label=part if y == 0 else None)
            if pct >= 15:
                ax.text(left + pct / 2, y, f"{count} ({pct:.0f}%)", ha="center", va="center", color="#0F172A",
                        fontsize=12, fontweight="bold")
            else:
                small.append(f"{part} {count} ({pct:.1f}%)")
            left += pct
        if small:
            ax.text(50, y - 0.42, "  ·  ".join(small), ha="center", va="center", color=HEX["white"], fontsize=11)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], color=HEX["white"], fontsize=13)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.75, len(rows) - 0.45)
    ax.set_xlabel("share of images (%)", color=HEX["muted"], fontsize=12)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.32), frameon=False, labelcolor=HEX["white"],
              fontsize=12)
    return fig_save(fig, "chart_split.png")


def chart_augmentation(data: dict) -> Path:
    v1, v2 = data["v1"], data["metrics"]["patchcore_r18"]
    t1, t2 = data["r18"]["roboflow_v1"]["info"]["fit_seconds"], data["r18"]["roboflow_v2"]["info"]["fit_seconds"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0), facecolor=HEX["bg"], gridspec_kw={"width_ratios": [1.6, 1]})
    for ax in (a, b):
        style_axes(ax)
    labels = ["Pixel AUROC", "Pixel IoU"]
    vals1 = [v1["pixel_level"]["auroc"], v1["pixel_level"]["iou"]]
    vals2 = [v2["pixel_level"]["auroc"], v2["pixel_level"]["iou"]]
    for i, (vals, lab, col) in enumerate(((vals1, "v1 · augmented", HEX["amber"]), (vals2, "v2 · no aug.", HEX["teal"]))):
        xs = [x + (i - 0.5) * 0.36 for x in range(2)]
        bars = a.bar(xs, vals, 0.34, color=col, label=lab)
        for bar, v in zip(bars, vals):
            a.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", color=HEX["white"], fontsize=11)
    a.set_xticks(range(2))
    a.set_xticklabels(labels, color=HEX["white"], fontsize=12)
    a.set_ylim(0, 1.18)
    a.legend(frameon=False, labelcolor=HEX["white"], fontsize=11, loc="upper right", ncol=1)
    bars = b.bar(["v1", "v2"], [t1, t2], color=[HEX["amber"], HEX["teal"]], width=0.55)
    for bar, v in zip(bars, (t1, t2)):
        b.text(bar.get_x() + bar.get_width() / 2, v + 12, f"{v:.0f} s", ha="center", color=HEX["white"], fontsize=11)
    b.set_ylim(0, max(t1, t2) * 1.25)
    b.set_title("Build time", color=HEX["white"], fontsize=12)
    b.tick_params(axis="x", labelsize=12)
    fig.tight_layout()
    return fig_save(fig, "chart_augmentation.png")


def chart_loss(data: dict) -> Path:
    fig, ax = plt.subplots(figsize=(4.4, 2.6), facecolor=HEX["bg"])
    style_axes(ax)
    for arch, label, colour in (("efficientad_s", "EfficientAD-S", HEX["indigo"]),
                                ("efficientad_m", "EfficientAD-M", HEX["amber"])):
        hist = data["meta"][arch]["summary"]["loss_history"]
        ax.plot([h["iteration"] for h in hist], [h["loss_total"] for h in hist], color=colour, linewidth=2.2,
                label=label)
    ax.set_xlabel("training iteration", color=HEX["muted"], fontsize=12)
    ax.set_ylabel("total training loss", color=HEX["muted"], fontsize=12)
    ax.legend(frameon=False, labelcolor=HEX["white"], fontsize=12)
    return fig_save(fig, "chart_loss.png")


# ================================================================== pptx helpers
def rgb(shape_fill, colour: RGBColor) -> None:
    shape_fill.solid()
    shape_fill.fore_color.rgb = colour


def box(slide, x, y, w, h, fill=PANEL, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08, line_w=1.0):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        s.fill.background()
    else:
        rgb(s.fill, fill)
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(line_w)
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    s.shadow.inherit = False
    return s


def text(slide, x, y, w, h, runs, size=18, colour=WHITE, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=1.0, space_after=0):
    """runs: str, or list of paragraphs; a paragraph is a str or a list of (text, {style}) tuples."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.vertical_anchor = anchor
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, Emu(0))
    paragraphs = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        p.space_after = Pt(space_after)
        parts = para if isinstance(para, list) else [(para, {})]
        for content, style in parts:
            r = p.add_run()
            r.text = content
            f = r.font
            f.name = FONT
            f.size = Pt(style.get("size", size))
            f.bold = style.get("bold", bold)
            f.color.rgb = style.get("colour", colour)
    return tb


def bullets(slide, x, y, w, h, items, size=18, colour=WHITE, bullet_colour=TEAL, gap=8):
    paras = [[("▸  ", {"colour": bullet_colour, "bold": True}), (item, {})] for item in items]
    return text(slide, x, y, w, h, paras, size=size, colour=colour, space_after=gap)


def picture(slide, path: Path, x, y, max_w, max_h, frame=TEAL):
    """Fit inside (max_w, max_h), centred, aspect preserved, with a rounded frame."""
    iw, ih = Image.open(path).size
    scale = min(max_w / iw, max_h / ih)
    w, h = iw * scale, ih * scale
    px, py = x + (max_w - w) / 2, y + (max_h - h) / 2
    pad = 0.06
    box(slide, px - pad, py - pad, w + 2 * pad, h + 2 * pad, fill=PANEL2, line=frame, radius=0.04, line_w=1.25)
    slide.shapes.add_picture(str(path), Inches(px), Inches(py), Inches(w), Inches(h))
    return px, py, w, h


def arrow(slide, x1, y1, x2, y2, colour=MUTED, width=2.0):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = colour
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)
    return c


def new_slide(prs, number: int, kicker: str, title: str, notes: str):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    number = len(prs.slides)  # slides are numbered in build order
    rgb(slide.background.fill, BG)
    if number > 1:
        box(slide, 0.45, 0.42, 0.09, 0.82, fill=INDIGO, shape=MSO_SHAPE.RECTANGLE)
        text(slide, 0.7, 0.32, 11.5, 0.3, kicker.upper(), size=14, colour=TEAL, bold=True)
        text(slide, 0.7, 0.6, 12.0, 0.7, title, size=34, bold=True)
        box(slide, 0.45, 6.92, 12.43, 0.02, fill=LINE, shape=MSO_SHAPE.RECTANGLE)
        text(slide, 0.45, 7.0, 9.0, 0.3, f"{NAME} · {REG} · VJTI Mumbai", size=14, colour=MUTED)
        text(slide, 10.88, 7.0, 2.0, 0.3, f"{number} / {TOTAL}", size=14, colour=MUTED, align=PP_ALIGN.RIGHT)
    slide.notes_slide.notes_text_frame.text = notes
    return slide


def card(slide, x, y, w, h, heading, body, accent=TEAL, head_size=20, body_size=16, number=None):
    box(slide, x, y, w, h, fill=PANEL, line=LINE)
    box(slide, x, y + 0.18, 0.07, h - 0.36, fill=accent, shape=MSO_SHAPE.RECTANGLE)
    left = x + 0.3
    if number is not None:
        circle = box(slide, x + 0.28, y + 0.22, 0.5, 0.5, fill=accent, shape=MSO_SHAPE.OVAL)
        circle.text_frame.text = str(number)
        para = circle.text_frame.paragraphs[0]
        para.alignment = PP_ALIGN.CENTER
        para.runs[0].font.size, para.runs[0].font.bold, para.runs[0].font.name = Pt(18), True, FONT
        para.runs[0].font.color.rgb = BG
        circle.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        left = x + 0.95
    text(slide, left, y + 0.22, w - (left - x) - 0.2, 0.5, heading, size=head_size, bold=True, colour=WHITE)
    text(slide, left, y + 0.22 + head_size / 72 * 1.45, w - (left - x) - 0.2, h - 0.8, body, size=body_size,
         colour=MUTED)


def stat(slide, x, y, w, h, value, label, accent=INDIGO, value_size=36, label_size=15):
    box(slide, x, y, w, h, fill=PANEL, line=accent, line_w=1.5)
    text(slide, x + 0.1, y + 0.12, w - 0.2, h * 0.55, value, size=value_size, bold=True, colour=accent,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(slide, x + 0.12, y + h * 0.6, w - 0.24, h * 0.36, label, size=label_size, colour=WHITE,
         align=PP_ALIGN.CENTER)


def flow_node(slide, x, y, w, h, title, sub, accent):
    box(slide, x, y, w, h, fill=PANEL, line=accent, line_w=1.75)
    text(slide, x + 0.1, y + 0.12, w - 0.2, 0.4, title, size=17, bold=True, colour=accent, align=PP_ALIGN.CENTER)
    text(slide, x + 0.12, y + 0.52, w - 0.24, h - 0.6, sub, size=14, colour=WHITE, align=PP_ALIGN.CENTER)


# ================================================================== slides
def build() -> None:
    data = load_data()
    m = data["metrics"]
    meta, colab = data["meta"], data["colab"]
    v2info, v1info = data["splits"]["roboflow_v2"]["info"], data["splits"]["roboflow_v1"]["info"]
    per_split = data["splits"]["roboflow_v2"]["per_split"]
    total_v2 = v2info["train_images"] + v2info["valid_images"] + v2info["test_images"]
    s_img, s_px = m["efficientad_s"]["image_level"], m["efficientad_s"]["pixel_level"]
    es_cfg, em_cfg = meta["efficientad_s"]["training_config"], meta["efficientad_m"]["training_config"]
    pc_cfg = meta["patchcore"]["training_config"]
    r18_cfg = data["r18"]["roboflow_v2"]["config"]
    r18_v1, r18_v2 = data["r18"]["roboflow_v1"]["info"], data["r18"]["roboflow_v2"]["info"]

    ASSETS.mkdir(exist_ok=True)
    img_heat = crop("new_model_comparison.png", (886, 602, 1054, 770), "title_heatmap.png")
    img_orig = crop("new_model_comparison.png", (336, 602, 504, 770), "title_original.png")
    img_gt = crop("new_model_comparison.png", (1262, 602, 1412, 770), "title_gt.png")
    img_compare = crop("new_model_comparison.png", (330, 596, 1420, 862), "shot_model_comparison.png")
    gt_src = Image.open(SHOTS / "new_ground_truth_comparison.png").convert("RGB")
    left, right = gt_src.crop((336, 462, 507, 633)), gt_src.crop((897, 462, 1068, 633))
    pair = Image.new("RGB", (171 * 2 + 12, 171), HEX["bg"])
    pair.paste(left, (0, 0))
    pair.paste(right, (183, 0))
    img_gtcomp = ASSETS / "shot_gt_comparison.png"
    pair.save(img_gtcomp)
    img_analytics = crop("new_analytics.png", (336, 505, 985, 790), "shot_analytics.png")
    c_metrics, c_split = chart_metrics(data), chart_split(data)
    c_aug, c_loss = chart_augmentation(data), chart_loss(data)

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

    # ---------------------------------------------------------------- 1 title
    s = new_slide(prs, 1, "", "", (
        f"Good morning. I am {NAME}, final year B.Tech Computer Engineering, registration number {REG}. "
        "My project is industrial anomaly detection and localization: finding and outlining defects in "
        "product images. The main model is EfficientAD, and PatchCore is used as the baseline. "
        f"The work was done under the guidance of {GUIDE}."))
    box(s, 0, 0, 0.22, 7.5, fill=INDIGO, shape=MSO_SHAPE.RECTANGLE)
    box(s, 0.22, 0, 0.06, 7.5, fill=TEAL, shape=MSO_SHAPE.RECTANGLE)
    text(s, 0.8, 0.95, 8.2, 0.35, "FINAL YEAR PROJECT REVIEW", size=16, bold=True, colour=TEAL)
    text(s, 0.8, 1.4, 8.0, 1.75, TITLE, size=34, bold=True, spacing=0.95)
    box(s, 0.8, 3.3, 1.6, 0.06, fill=AMBER, shape=MSO_SHAPE.RECTANGLE)
    text(s, 0.8, 3.6, 8.3, 2.9, [
        [(NAME, {"size": 24, "bold": True})],
        [(f"Registration No. {REG}", {"size": 17, "colour": MUTED})],
        [(PROGRAM, {"size": 17, "colour": MUTED})],
        [(f"{DEPT}, {COLLEGE}", {"size": 17, "colour": MUTED})],
        [("Guide: ", {"size": 17, "colour": TEAL, "bold": True}), (GUIDE, {"size": 17})],
        [(YEAR, {"size": 17, "colour": AMBER, "bold": True})],
    ], space_after=6)
    for i, (img, cap, col) in enumerate(((img_orig, "Input image", MUTED), (img_heat, "EfficientAD-S heatmap", INDIGO),
                                         (img_gt, "Ground truth", TEAL))):
        y = 0.75 + i * 2.05
        picture(s, img, 10.15, y, 1.75, 1.6, frame=col)
        text(s, 9.45, y + 1.66, 3.15, 0.3, cap, size=14, colour=MUTED, align=PP_ALIGN.CENTER)

    # ---------------------------------------------------------------- 2 problem
    s = new_slide(prs, 2, "Problem Statement", "Finding defects that are rare and unpredictable", (
        "In a factory a camera photographs every product, and we want to find the defective ones automatically. "
        "The hard part is that defects are rare and keep changing: there may be thousands of good bottles but "
        "only a handful of broken ones. So instead of learning what a defect looks like, our models learn what "
        "a normal product looks like and flag anything different. They solve two tasks: is the image defective, "
        "and where exactly is the defect."))
    card(s, 0.7, 1.65, 3.85, 2.0, "Real-world context",
         "A camera photographs every product on the line; cracks, breaks and contamination must be caught "
         "automatically.", accent=INDIGO)
    card(s, 4.74, 1.65, 3.85, 2.0, "Why it is hard",
         "Defects are rare and unpredictable: many good products, very few defective ones, and new defect "
         "types can appear.", accent=AMBER)
    card(s, 8.78, 1.65, 3.85, 2.0, "Our approach",
         "Learn only what normal looks like (one-class / unsupervised) and flag anything that deviates.",
         accent=TEAL)
    box(s, 0.7, 3.9, 11.93, 2.75, fill=PANEL2, line=LINE)
    text(s, 1.0, 4.12, 11.4, 0.4, "Two tasks solved for every product image", size=21, bold=True, colour=TEAL)
    for i, (head, body, col) in enumerate((
            ("Detection (image level)", "Is this product normal or defective?  →  anomaly score + verdict",
             INDIGO),
            ("Localisation (pixel level)", "Where is the defect?  →  anomaly heatmap + defect mask", AMBER))):
        y = 4.8 + i * 0.85
        box(s, 1.0, y, 11.33, 0.68, fill=PANEL, line=col, radius=0.2)
        box(s, 1.2, y + 0.23, 0.22, 0.22, fill=col, shape=MSO_SHAPE.OVAL)
        text(s, 1.6, y, 10.6, 0.68, [[(head + ":  ", {"bold": True, "colour": col}), (body, {})]], size=19,
             anchor=MSO_ANCHOR.MIDDLE)

    # ---------------------------------------------------------------- 3 existing
    s = new_slide(prs, 3, "Existing System & Its Limitations", "Why current approaches fall short", (
        "Today, inspection is often done by people, which is slow and inconsistent over long shifts. "
        "The usual machine-learning alternative is a supervised classifier, trained on labelled examples of "
        "each defect. That needs many defect images, which are rare, and it cannot recognise a defect type it "
        "has never seen. It also only says good or bad, without showing where the defect is. These gaps "
        "motivate an unsupervised approach that learns from good products only."))
    rows = [("Manual visual inspection", "Slow, tiring and inconsistent between inspectors and over long shifts.",
             ROSE),
            ("Supervised defect classifiers", "Need many labelled images of every defect type, but defects are rare.",
             AMBER),
            ("Unseen defect types", "A supervised model cannot detect a defect type that was not in its training data.",
             AMBER),
            ("No localisation", "An image-level good / bad label does not show where the defect is.", INDIGO)]
    for i, (head, body, col) in enumerate(rows):
        y = 1.65 + i * 1.28
        box(s, 0.7, y, 11.93, 1.1, fill=PANEL, line=LINE)
        box(s, 0.95, y + 0.3, 0.5, 0.5, fill=col, shape=MSO_SHAPE.OVAL)
        text(s, 0.95, y + 0.3, 0.5, 0.5, "✕", size=18, bold=True, colour=BG, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)
        text(s, 1.75, y + 0.15, 3.9, 0.8, head, size=20, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        text(s, 5.7, y + 0.15, 6.75, 0.8, body, size=17, colour=MUTED, anchor=MSO_ANCHOR.MIDDLE)

    # ---------------------------------------------------------------- 4 objectives
    s = new_slide(prs, 4, "Objectives", "What the project set out to do", (
        "We had five objectives. First, build our own dataset on Roboflow with an 80/10/10 split, preprocessing "
        "and augmentation. Second, train EfficientAD, our main model, in two sizes. Third, compare it against two "
        "PatchCore baselines. Fourth, evaluate all models fairly on the same test images, for both detection and "
        "localisation. Fifth, deliver everything in an interactive web platform with PDF reports."))
    objectives = [
        ("Build the dataset", "Roboflow project from scratch: 80/10/10 split, preprocessing, 7 augmentations (v1) "
                              "and a clean version (v2).", INDIGO),
        ("Train EfficientAD", "Main model in two sizes (S and M) on a GPU, learning from normal images only.", TEAL),
        ("PatchCore baselines", "ResNet18 and WideResNet-50 PatchCore as strong training-free baselines.", AMBER),
        ("Fair evaluation", "Same 20 test images, same code: detection (image) and localisation (pixel) metrics.",
         INDIGO),
        ("Inspection platform", "Streamlit web app: live inspection, model comparison, analytics, PDF reports.",
         TEAL)]
    for i, (head, body, col) in enumerate(objectives):
        col_i, row = i % 3, i // 3
        w = 3.85
        x = 0.7 + col_i * 4.04 if row == 0 else 2.72 + col_i * 4.04
        card(s, x, 1.7 + row * 2.55, w, 2.3, head, body, accent=col, number=i + 1, body_size=17)

    # ---------------------------------------------------------------- 5 proposed
    s = new_slide(prs, 5, "Proposed System", "What we actually built", (
        "Our system has three parts. The data part is our Roboflow dataset in two versions: v1 with seven "
        "augmentations and v2 without, which became the final dataset after we compared them. The model part is "
        "EfficientAD S and M as the main model, with two PatchCore baselines. The platform part is a Streamlit "
        "website where you can switch between v1 and v2, run any model on an image, compare them and download "
        "PDF reports."))
    pillars = [
        ("Data · Roboflow", INDIGO, [f"{total_v2} MVTec AD bottle images", "Split 80 / 10 / 10, train = normal only",
                                     "v1: 7 augmentations (×3 train)", "v2: no augmentation (final)"]),
        ("Models", TEAL, ["EfficientAD-S and -M (main)", "PatchCore ResNet18 (baseline)", "PatchCore WRN-50 (baseline)",
                          "Learn from normal images only"]),
        ("Platform · Streamlit", AMBER, ["Dataset switch: v2 / v1 / MVTec", "Live inspection + heatmaps",
                                         "4-model comparison, analytics", "PDF reports, history"])]
    for i, (head, col, items) in enumerate(pillars):
        x = 0.7 + i * 4.04
        box(s, x, 1.65, 3.85, 3.35, fill=PANEL, line=col, line_w=1.5)
        box(s, x, 1.65, 3.85, 0.8, fill=col, radius=0.15)
        box(s, x, 2.15, 3.85, 0.3, fill=col, shape=MSO_SHAPE.RECTANGLE)
        text(s, x, 1.65, 3.85, 0.8, head, size=21, bold=True, colour=BG, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)
        bullets(s, x + 0.3, 2.68, 3.4, 2.25, items, size=17, bullet_colour=col, gap=8)
    strip = [("2", "Roboflow dataset versions (v1, v2)", INDIGO), ("4", "trained models, same test images", TEAL),
             ("10", "pages in the web platform", AMBER)]
    for i, (v, lab, col) in enumerate(strip):
        x = 0.7 + i * 4.04
        box(s, x, 5.25, 3.85, 1.4, fill=PANEL2, line=col, line_w=1.25)
        text(s, x + 0.3, 5.25, 0.85, 1.4, v, size=40, bold=True, colour=col, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x + 1.15, 5.25, 2.55, 1.4, lab, size=17, colour=WHITE, anchor=MSO_ANCHOR.MIDDLE)

    # ---------------------------------------------------------------- 6 architecture
    s = new_slide(prs, 6, "System Architecture", "From product images to an inspection verdict", (
        "This is the end-to-end flow. Bottle images from MVTec AD go into our Roboflow project, where they are "
        "labelled, split and preprocessed, and exported as v1 and v2. A converter puts them back into the MVTec "
        "folder layout and restores the ground-truth masks. The models are then trained: EfficientAD and WRN-50 "
        "on a Colab GPU, ResNet18 on the laptop. Finally every model is evaluated on the same test images and "
        "served in the Streamlit platform."))
    top = [("Input", "MVTec AD bottle images + defect masks", MUTED),
           ("Roboflow", "Label · split 80/10/10 · preprocess · augment", INDIGO),
           ("Export v1 / v2", "Convert to MVTec layout, restore masks", INDIGO),
           ("Training", "EfficientAD-S/M, WRN-50 (T4 GPU) · R18 (CPU)", TEAL)]
    w, h, gap, y1 = 2.6, 1.5, 0.51, 1.8
    for i, (t, sub, col) in enumerate(top):
        x = 0.7 + i * (w + gap)
        flow_node(s, x, y1, w, h, t, sub, col)
        if i < len(top) - 1:
            arrow(s, x + w + 0.05, y1 + h / 2, x + w + gap - 0.05, y1 + h / 2, colour=TEAL, width=2.5)
    bottom = [("Output", "Verdict, heatmap, mask, PDF report", AMBER),
              ("Platform", "Streamlit: inspection, comparison, analytics", AMBER),
              ("Thresholds", "From normal validation images only", TEAL),
              ("Evaluation", f"Same {v2info['test_images']} test images, image + pixel metrics", TEAL)]
    y2 = 4.4
    for i, (t, sub, col) in enumerate(bottom):
        x = 0.7 + i * (w + gap)
        flow_node(s, x, y2, w, h, t, sub, col)
        if i < len(bottom) - 1:
            arrow(s, x + w + gap - 0.05, y2 + h / 2, x + w + 0.05, y2 + h / 2, colour=TEAL, width=2.5)
    last_x = 0.7 + 3 * (w + gap) + w / 2
    arrow(s, last_x, y1 + h + 0.05, last_x, y2 - 0.05, colour=TEAL, width=2.5)
    text(s, 0.7, 6.15, 11.9, 0.4, "Anomalous images are used only in validation and test, never in training.",
         size=15, colour=MUTED, align=PP_ALIGN.CENTER)

    # ---------------------------------------------------------------- 7 algorithm
    s = new_slide(prs, 7, "Algorithm / Model Used", "EfficientAD (main) and PatchCore (baseline)", (
        "EfficientAD uses a teacher network that was pretrained to describe image patches, and a student network "
        "that we train to copy the teacher, but only on normal images. On a defect the student cannot copy the "
        "teacher, and that difference is the anomaly. An autoencoder adds a second, global view for logical "
        "defects. We chose it because the paper reports top accuracy at millisecond GPU latency and gives "
        "detailed maps. PatchCore is the baseline: it stores normal patch features and needs no training at all."))
    box(s, 0.7, 1.65, 7.3, 5.0, fill=PANEL, line=INDIGO, line_w=1.5)
    text(s, 1.0, 1.85, 6.8, 0.5, "EfficientAD  ·  main model", size=24, bold=True, colour=INDIGO)
    text(s, 1.0, 2.35, 6.8, 0.4, "Batzner, Heckler, König · WACV 2024", size=15, colour=MUTED)
    steps = [("Teacher", "pretrained patch-description network (frozen)", INDIGO),
             ("Student", "trained to copy the teacher on normal images only", TEAL),
             ("Autoencoder", "global view for logical anomalies", AMBER)]
    for i, (head, body, col) in enumerate(steps):
        y = 2.95 + i * 0.78
        box(s, 1.0, y, 6.7, 0.66, fill=PANEL2, line=col)
        text(s, 1.2, y, 1.7, 0.66, head, size=18, bold=True, colour=col, anchor=MSO_ANCHOR.MIDDLE)
        text(s, 2.9, y, 4.7, 0.66, body, size=16, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 1.0, 5.4, 6.7, 1.1, [[("Key idea: ", {"bold": True, "colour": AMBER}),
                                   ("where the student fails to copy the teacher, the region is unlike anything "
                                    "seen in training, so it is anomalous.", {})]], size=17)
    box(s, 8.2, 1.65, 4.43, 2.55, fill=PANEL, line=TEAL, line_w=1.5)
    text(s, 8.45, 1.82, 4.0, 0.45, "Why EfficientAD?", size=20, bold=True, colour=TEAL)
    bullets(s, 8.45, 2.32, 4.0, 1.85, ["Top accuracy on MVTec AD (paper)",
                                       "Millisecond latency on a GPU", "Pixel-level anomaly maps",
                                       "Learns from normal images only"], size=16, gap=6)
    box(s, 8.2, 4.35, 4.43, 2.3, fill=PANEL, line=AMBER, line_w=1.5)
    text(s, 8.45, 4.52, 4.0, 0.45, "Baseline: PatchCore", size=20, bold=True, colour=AMBER)
    bullets(s, 8.45, 5.02, 4.0, 1.6, ["Memory bank of normal patch features",
                                      "Nearest-neighbour distance = anomaly", "No gradient training",
                                      "Strong, training-free reference"],
            size=16, bullet_colour=AMBER, gap=6)

    # ---------------------------------------------------------------- 8 PatchCore step by step
    r18_info = data["r18"]["roboflow_v2"]["info"]
    pc_sum = meta["patchcore"]["summary"]
    grid_side = int(round((r18_info["patch_vectors"] / r18_info["train_images"]) ** 0.5))
    s = new_slide(prs, 8, "How PatchCore Works", "Baseline, step by step: remember normal patches", (
        "PatchCore remembers small pieces of normal images. Step one: each image goes through a ResNet18 that was "
        "pretrained on ImageNet and is frozen, so its weights never change. We take features from layer2 and "
        f"layer3: layer1 is too basic and layer4 too tied to ImageNet classes. This gives a {grid_side} by "
        f"{grid_side} grid of {r18_info['feature_dim']}-dimensional patch vectors per image. Step two stores the "
        f"patch vectors of all {r18_info['train_images']} normal training images, {r18_info['patch_vectors']:,} "
        f"vectors. Step three, coreset selection, keeps only ten percent, {r18_info['memory_bank_size']:,}, using "
        "greedy k-center: it keeps adding the vector farthest from everything already chosen, so every kind of "
        "normal appearance is still covered. Step four: for every patch of a test image we find its nearest "
        "neighbour in the bank; a large distance means the patch is anomalous, and the image score is the "
        "maximum patch distance, because one bad region is enough to reject a bottle. Step five turns the grid "
        "of distances into a heatmap. Key point: there is no training loop, no loss and no backpropagation."))
    flow = [("1", "Extract features", f"Frozen ImageNet CNN, layer2 + layer3 → {grid_side}×{grid_side} grid of "
                                       f"{r18_info['feature_dim']}-dim vectors", INDIGO),
            ("2", "Memory bank", f"Store all patches of the {r18_info['train_images']} normal images: "
                                 f"{r18_info['patch_vectors']:,} vectors", INDIGO),
            ("3", "Coreset", f"Greedy k-center keeps {r18_cfg['coreset_ratio']:.0%} "
                             f"({r18_info['memory_bank_size']:,}) yet covers every normal look", TEAL),
            ("4", "Nearest match", "Distance of each test patch to its closest normal patch; "
                                   "image score = max", TEAL),
            ("5", "Heatmap + mask", f"Upsample, Gaussian blur (σ {r18_cfg['smoothing_sigma']:g}), "
                                    "pixel threshold → mask", AMBER)]
    w, gap = 2.1, 0.36
    for i, (num, head, body, col) in enumerate(flow):
        x = 0.7 + i * (w + gap)
        box(s, x, 1.6, w, 2.5, fill=PANEL, line=col, line_w=1.5)
        box(s, x + w / 2 - 0.3, 1.78, 0.6, 0.6, fill=col, shape=MSO_SHAPE.OVAL)
        text(s, x + w / 2 - 0.3, 1.78, 0.6, 0.6, num, size=22, bold=True, colour=BG, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)
        text(s, x + 0.12, 2.5, w - 0.24, 0.4, head, size=18, bold=True, colour=col, align=PP_ALIGN.CENTER)
        text(s, x + 0.15, 2.95, w - 0.3, 1.45, body, size=15, align=PP_ALIGN.CENTER)
        if i < len(flow) - 1:
            arrow(s, x + w + 0.04, 2.85, x + w + gap - 0.04, 2.85, colour=TEAL, width=2.5)
    box(s, 0.7, 4.35, 5.85, 2.3, fill=PANEL2, line=AMBER)
    text(s, 0.95, 4.5, 5.4, 0.4, "Why these choices?", size=18, bold=True, colour=AMBER)
    bullets(s, 0.95, 4.98, 5.4, 1.6, [
        "layer2 + 3: mid-level detail (layer1 too basic, layer4 too class-specific)",
        "Max, not mean: one bad patch rejects the bottle",
        "No loss, no backprop: storing features is the learning"], size=16, bullet_colour=AMBER, gap=6)
    box(s, 6.78, 4.35, 5.85, 2.3, fill=PANEL2, line=TEAL)
    text(s, 7.03, 4.5, 5.4, 0.4, "Two backbones (real build numbers)", size=18, bold=True, colour=TEAL)
    bullets(s, 7.03, 4.98, 5.4, 1.6, [
        f"ResNet18: {r18_info['patch_vectors']:,} → {r18_info['memory_bank_size']:,} vectors, "
        f"{r18_info['feature_dim']}-d, {m['patchcore_r18']['latency_ms']['mean']:.0f} ms",
        f"WRN-50: {pc_sum['total_embeddings']:,} → {pc_sum['coreset_size']:,} vectors, "
        f"{pc_sum['embedding_dim']}-d, {m['patchcore_wrn50']['latency_ms']['mean']:.0f} ms",
        "WideResNet-50: richer features, about 5× slower"], size=16, gap=6)

    # ---------------------------------------------------------------- 9 EfficientAD step by step
    es_sum = meta["efficientad_s"]["summary"]
    s = new_slide(prs, 9, "How EfficientAD Works", "Main model: a student learns to copy a teacher", (
        "EfficientAD has two branches. The local branch has a teacher, a small network called the Patch "
        "Description Network that was distilled from a large pretrained network and is frozen. The student is "
        "trained to imitate the teacher, but only on normal bottles. On a crack or scratch the student has never "
        "seen anything similar, so it fails to copy the teacher, and that difference is the local anomaly map. "
        "The global branch is an autoencoder that reconstructs the teacher's features from a compressed code; the "
        "student also predicts the autoencoder's output, and their difference flags logical defects such as a "
        "missing or misplaced part. Two training tricks matter: the hard-feature loss only counts the worst "
        "copied features, the top 0.1 percent, and the ImageNet penalty stops the student from copying the "
        "teacher on everything, so it stays a specialist in normal bottles. Both maps are normalised with "
        "quantiles from held-out normal images and averaged. S and M are the same design; M has wider networks."))
    box(s, 0.7, 1.6, 1.75, 2.75, fill=PANEL, line=MUTED, line_w=1.5)
    text(s, 0.8, 2.35, 1.55, 0.4, "Input", size=18, bold=True, colour=WHITE, align=PP_ALIGN.CENTER)
    text(s, 0.8, 2.8, 1.55, 0.8, f"{es_cfg['image_size']}×{es_cfg['image_size']} image (train: normal only)",
         size=14, colour=MUTED, align=PP_ALIGN.CENTER)
    branches = [("Teacher (PDN)", "frozen, pretrained", "Student", "imitates the teacher",
                 "Local map", "student ≠ teacher → cracks, scratches", INDIGO, 1.6),
                ("Autoencoder", "reconstructs teacher", "Student (2nd head)", "predicts the AE",
                 "Global map", "student ≠ AE → missing / misplaced part", AMBER, 3.1)]
    bw = 2.05
    for a, a_sub, b, b_sub, c, c_sub, col, y in branches:
        arrow(s, 2.5, y + 0.62, 2.7, y + 0.62, colour=TEAL, width=2.5)
        for x, head, sub, accent in ((2.75, a, a_sub, col), (5.1, b, b_sub, TEAL), (7.45, c, c_sub, col)):
            box(s, x, y, bw, 1.25, fill=PANEL, line=accent, line_w=1.75)
            text(s, x + 0.08, y + 0.1, bw - 0.16, 0.4, head, size=17, bold=True, colour=accent,
                 align=PP_ALIGN.CENTER)
            text(s, x + 0.1, y + 0.5, bw - 0.2, 0.7, sub, size=14, colour=WHITE, align=PP_ALIGN.CENTER)
        arrow(s, 4.83, y + 0.62, 5.07, y + 0.62, colour=TEAL, width=2.5)
        arrow(s, 7.18, y + 0.62, 7.42, y + 0.62, colour=TEAL, width=2.5)
        arrow(s, 9.53, y + 0.62, 9.77, y + 0.62, colour=TEAL, width=2.5)
    box(s, 9.8, 1.6, 2.83, 2.75, fill=PANEL, line=TEAL, line_w=2.0)
    text(s, 9.92, 1.75, 2.6, 0.45, "Anomaly map", size=19, bold=True, colour=TEAL, align=PP_ALIGN.CENTER)
    text(s, 9.95, 2.25, 2.53, 2.0, [f"Normalise each map with quantiles {es_cfg['q_a']:g} / {es_cfg['q_b']:g} "
                                    f"of {es_sum['validation_images']} held-out normal images", "",
                                    "Average → one map, score = its maximum"],
         size=14, colour=WHITE, align=PP_ALIGN.CENTER)
    cards = [("Hard-feature loss", f"Only the worst-copied features count (top {1 - es_cfg['p_hard']:.1%}), so the "
                                   "student focuses on what it gets wrong.", INDIGO),
             ("ImageNet penalty", f"{es_sum['imagenet_images']:,} tiny-ImageNet images stop the student copying "
                                  "everything; it stays a normal-bottle specialist.", AMBER),
             ("S vs M · our run", f"Same design, M is wider. Colab T4: S {es_cfg['total_iters']:,} it., "
                                  f"M {em_cfg['total_iters']:,} it. (paper 70,000).", TEAL)]
    for i, (head, body, col) in enumerate(cards):
        card(s, 0.7 + i * 4.04, 4.55, 3.85, 1.6, head, body, accent=col, head_size=18, body_size=15)
    text(s, 0.7, 6.28, 11.93, 0.4, [[("Key idea: ", {"bold": True, "colour": AMBER}),
                                     ("where the student fails to copy the teacher, the region is unlike any normal "
                                      "bottle, so it is anomalous.", {})]], size=16, align=PP_ALIGN.CENTER)

    # ---------------------------------------------------------------- 10 workflow
    s = new_slide(prs, 10, "Algorithm Workflow", "How EfficientAD is trained and used", (
        "Step one computes the teacher's channel statistics on the normal training images. Step two trains the "
        "student and the autoencoder with Adam, using a hard-feature loss and an ImageNet penalty. Step three "
        "computes normalisation quantiles on held-out normal images. Step four combines the local and global "
        "maps into one anomaly map. Step five applies two thresholds, also from normal images: the image score "
        "decides normal or defective, and the pixel threshold gives the defect mask."))
    flow = [("1", "Teacher statistics", "Channel mean / std on normal training images", INDIGO),
            ("2", "Train student", f"Student + autoencoder; Adam, lr {es_cfg['lr']:g}; hard-feature loss "
                                    f"+ ImageNet penalty", INDIGO),
            ("3", "Quantile normalisation", f"q {es_cfg['q_a']:g} / {es_cfg['q_b']:g} on "
                                            f"{meta['efficientad_s']['summary']['validation_images']} held-out "
                                            "normal images", TEAL),
            ("4", "Anomaly map", "Average of local (student-teacher) and global (autoencoder) maps", TEAL),
            ("5", "Decision", "Score above threshold → DEFECTIVE; map above pixel threshold → mask", AMBER)]
    w, gap = 2.1, 0.36
    for i, (num, head, body, col) in enumerate(flow):
        x = 0.7 + i * (w + gap)
        box(s, x, 1.75, w, 3.45, fill=PANEL, line=col, line_w=1.5)
        circle = box(s, x + w / 2 - 0.36, 1.98, 0.72, 0.72, fill=col, shape=MSO_SHAPE.OVAL)
        text(s, x + w / 2 - 0.36, 1.98, 0.72, 0.72, num, size=24, bold=True, colour=BG, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)
        text(s, x + 0.12, 2.88, w - 0.24, 0.7, head, size=18, bold=True, colour=col, align=PP_ALIGN.CENTER)
        text(s, x + 0.15, 3.62, w - 0.3, 1.5, body, size=15, align=PP_ALIGN.CENTER)
        if i < len(flow) - 1:
            arrow(s, x + w + 0.04, 3.45, x + w + gap - 0.04, 3.45, colour=TEAL, width=2.5)
    box(s, 0.7, 5.45, 11.93, 1.2, fill=PANEL2, line=AMBER)
    text(s, 1.0, 5.45, 11.4, 1.2, [[("Thresholds  ", {"bold": True, "colour": AMBER}),
                                     ("image = highest score among normal validation images  ·  pixel = 99.5th "
                                      "percentile of their anomaly-map pixels  ·  no test image is used", {})]],
         size=17, anchor=MSO_ANCHOR.MIDDLE)

    # ---------------------------------------------------------------- 9 dataset
    s = new_slide(prs, 9, "Dataset Used", "Our Roboflow dataset, built from MVTec AD", (
        f"The dataset is our own Roboflow project called bottle-anomaly. It contains {total_v2} images of the "
        "bottle category of MVTec AD, the standard benchmark, with two classes: normal and anomalous, and three "
        "defect types. We generated two versions on Roboflow: v1 with seven augmentations, which grows the "
        "training set to 480 images, and v2 without augmentation. The picture shows real augmented training "
        "images from v1."))
    defect_types = v2info["defect_types"]
    stats = [(str(total_v2), "images selected", INDIGO), ("2", "classes: normal / anomalous", TEAL),
             (str(len(defect_types)), "defect types", AMBER), ("224²", "pixels after preprocessing", INDIGO)]
    for i, (v, lab, col) in enumerate(stats):
        stat(s, 0.7 + i * 3.02, 1.6, 2.83, 1.35, v, lab, accent=col, value_size=34, label_size=15)
    text(s, 0.7, 3.1, 11.9, 0.35, [[("Source: ", {"bold": True, "colour": TEAL}),
                                     ("MVTec AD (Bergmann et al., CVPR 2019) · bottle · licence CC BY-NC-SA 4.0  ·  "
                                      "defects: " + ", ".join(defect_types), {})]], size=15, colour=MUTED)
    versions = [("v1 · 7 augmentations", AMBER, [f"{v1info['train_images']} train images (160 × 3)",
                                                 "flip, flop, 90° rotate, crop/zoom, brightness, exposure, blur"]),
                ("v2 · no augmentation (final)", TEAL, [f"{v2info['train_images']} train images",
                                                        "Used by all 4 models · default dataset in the website"])]
    for i, (head, col, items) in enumerate(versions):
        y = 3.6 + i * 1.6
        box(s, 0.7, y, 6.5, 1.45, fill=PANEL, line=col, line_w=1.5)
        text(s, 0.95, y + 0.12, 6.0, 0.4, head, size=19, bold=True, colour=col)
        text(s, 0.95, y + 0.55, 6.0, 0.85, items, size=16, colour=WHITE)
    grid = v1_grid()
    picture(s, grid, 7.5, 3.62, 5.1, 2.62, frame=AMBER)
    text(s, 7.5, 6.32, 5.1, 0.35, "Real v1 training images: original  |  augmented copy 1  |  copy 2",
         size=14, colour=MUTED, align=PP_ALIGN.CENTER)

    # ---------------------------------------------------------------- 10 preprocessing & split
    s = new_slide(prs, 10, "Data Preprocessing & Split", "Same 80 / 10 / 10 split in both versions", (
        "Both versions get the same preprocessing: auto-orient, then a stretch to 224 by 224 pixels. The split is "
        "80, 10 and 10 percent: 160 training images, all normal, and 20 validation and 20 test images, each half "
        "normal and half defective. In v1 only the training images are augmented three times, so it has 480 "
        "training images while validation and test stay the same. Train has zero defective images on purpose: a "
        "defect in training would be learned as normal and then missed. Augmentation did not help: pixel AUROC "
        "was slightly lower and the build took 2.5 times longer, because a rotated or brighter bottle becomes "
        "acceptable and the model gets less sensitive. So v2 is our final dataset."))
    pre = [("Auto-Orient", "all images", INDIGO), ("Resize", "stretch to 224×224", INDIGO),
           ("Augment ×3", "v1 train only", AMBER)]
    for i, (head, sub, col) in enumerate(pre):
        x = 0.7 + i * 2.0
        flow_node(s, x, 1.65, 1.75, 1.1, head, sub, col)
        if i < 2:
            arrow(s, x + 1.78, 2.2, x + 1.97, 2.2, colour=TEAL, width=2.5)
    test_def = per_split["test"]
    valid_def = per_split["valid"]
    split_lines = [
        [("Train  ", {"bold": True, "colour": INDIGO}), (f"{v2info['train_images']} normal (v1: {v1info['train_images']})", {})],
        [("Valid  ", {"bold": True, "colour": TEAL}),
         (f"{valid_def['good']} normal + {sum(v for k, v in valid_def.items() if k != 'good')} defective", {})],
        [("Test   ", {"bold": True, "colour": AMBER}),
         (f"{test_def['good']} normal + {sum(v for k, v in test_def.items() if k != 'good')} defective", {})],
        [("Defects (valid / test): " + ", ".join(f"{k} {valid_def[k]}/{test_def[k]}" for k in test_def if k != "good"),
          {"size": 14, "colour": MUTED})]]
    box(s, 0.7, 2.95, 5.75, 1.6, fill=PANEL, line=LINE)
    text(s, 0.95, 3.07, 5.35, 1.45, split_lines, size=15, space_after=5)
    box(s, 0.7, 4.75, 5.75, 1.9, fill=PANEL2, line=AMBER)
    text(s, 0.95, 4.87, 5.3, 0.4, "Augmentation study (v1 vs v2)", size=18, bold=True, colour=AMBER)
    bullets(s, 0.95, 5.32, 5.3, 1.3, [
        f"Pixel AUROC {data['v1']['pixel_level']['auroc']:.4f} (v1) vs {m['patchcore_r18']['pixel_level']['auroc']:.4f} (v2)",
        f"Build {r18_v1['fit_seconds']:.0f} s vs {r18_v2['fit_seconds']:.0f} s  →  v2 kept as final",
        "Why: augmenting normal images widens what counts as normal"],
        size=15, bullet_colour=AMBER, gap=4)
    picture(s, c_split, 6.7, 1.62, 5.93, 2.45, frame=LINE)
    picture(s, c_aug, 6.7, 4.3, 5.93, 2.35, frame=LINE)

    # ---------------------------------------------------------------- 11 implementation
    s = new_slide(prs, 11, "Implementation Details", "Tech stack, training setup and hardware", (
        "The project is written in Python with PyTorch and torchvision for the models, scikit-learn for metrics, "
        "OpenCV for image processing and Streamlit for the website. EfficientAD was trained on a free Google Colab "
        "Tesla T4 GPU because CPU training would take about 32 hours for S. It uses Adam with learning rate "
        "0.0001, batch size 1, and 2,500 and 1,500 iterations, which is a reduced protocol compared to the paper's "
        "70,000. PatchCore needs no training, and inference runs on the laptop CPU."))
    stack = ["Python", "PyTorch 2.10", "torchvision 0.25", "scikit-learn 1.7", "OpenCV 4.13", "Streamlit 1.47",
             "Plotly", "ReportLab (PDF)", "Roboflow", "Google Colab"]
    x, y = 0.7, 1.65
    for item in stack:
        w = 0.32 + len(item) * 0.112
        if x + w > 12.63:
            x, y = 0.7, y + 0.6
        box(s, x, y, w, 0.48, fill=PANEL, line=TEAL, radius=0.5)
        text(s, x, y, w, 0.48, item, size=15, bold=True, colour=TEAL, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        x += w + 0.14
    hp_rows = [("Setting", "EfficientAD-S / M", "PatchCore R18 / WRN-50"),
               ("Input size", f"{es_cfg['image_size']}×{es_cfg['image_size']}",
                f"{r18_cfg['image_size']}×{r18_cfg['image_size']} / {pc_cfg['image_size']}×{pc_cfg['image_size']}"),
               ("Optimiser", f"Adam, lr {es_cfg['lr']:g} → {es_cfg['lr_decay']:g}", "none (no gradient training)"),
               ("Iterations", f"{es_cfg['total_iters']:,} / {em_cfg['total_iters']:,}  (paper 70,000)",
                f"coreset {r18_cfg['coreset_ratio']:.0%} / {pc_cfg['coreset_ratio']:.0%}"),
               ("Batch size", f"{es_cfg['batch_size']}  ·  weight decay {es_cfg['weight_decay']:g}",
                f"{r18_cfg['batch_size']} (feature extraction)"),
               ("Hardware", f"Colab {colab['gpu']}: {colab['times_s']['efficientad_s']} s / "
                            f"{colab['times_s']['efficientad_m']} s",
                f"i7-11800H CPU {r18_v2['fit_seconds']:.0f} s / T4 {colab['times_s']['patchcore']} s")]
    table_y = y + 0.75
    tbl = s.shapes.add_table(len(hp_rows), 3, Inches(0.7), Inches(table_y), Inches(11.93),
                             Inches(0.6 * len(hp_rows))).table
    for c, wdt in enumerate((2.4, 4.9, 4.63)):
        tbl.columns[c].width = Inches(wdt)
    for r, row in enumerate(hp_rows):
        tbl.rows[r].height = Inches(0.6)
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            rgb(cell.fill, INDIGO if r == 0 else (PANEL if r % 2 else PANEL2))
            cell.margin_left = cell.margin_right = Inches(0.12)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            para = cell.text_frame.paragraphs[0]
            para.text = val
            run = para.runs[0]
            run.font.name, run.font.size = FONT, Pt(17)
            run.font.bold = r == 0 or c == 0
            run.font.color.rgb = WHITE if r == 0 else (TEAL if c == 0 else WHITE)

    # ---------------------------------------------------------------- 14 evaluation metrics explained
    cm_s = s_img["confusion_matrix"]
    r18_px = m["patchcore_r18"]["pixel_level"]
    s = new_slide(prs, 14, "Evaluation Metrics", "How the models are scored (positive = defective)", (
        "Before the results, here is how we score the models. Positive means defective. A true positive is a "
        "caught defect, a true negative a good bottle that passes, a false positive a false alarm, and a false "
        "negative a missed defect, which is the worst mistake in a factory. The matrix shows EfficientAD-S's real "
        f"counts: {cm_s['tp']} caught, {cm_s['fn']} missed, {cm_s['fp']} false alarm, {cm_s['tn']} passed. "
        "Precision asks: of what I flagged, how much was really bad? Recall asks: of all real defects, how many "
        "did I catch? F1 is high only if both are. AUROC is the chance that a random defective sample scores "
        "higher than a random normal one, and it does not depend on the threshold. IoU and Dice measure how well "
        "the predicted defect mask overlaps the true one. In a factory recall usually matters most, because a "
        "missed defect reaches the customer."))
    text(s, 0.7, 1.55, 4.9, 0.4, "Confusion matrix · EfficientAD-S", size=17, bold=True, colour=WHITE)
    text(s, 2.05, 1.98, 1.75, 0.3, "Pred. defective", size=14, colour=MUTED, align=PP_ALIGN.CENTER)
    text(s, 3.9, 1.98, 1.75, 0.3, "Pred. normal", size=14, colour=MUTED, align=PP_ALIGN.CENTER)
    for r, (row_lab, cells_) in enumerate((("Defective", (("TP", cm_s["tp"], "caught", TEAL),
                                                          ("FN", cm_s["fn"], "missed", ROSE))),
                                           ("Normal", (("FP", cm_s["fp"], "false alarm", AMBER),
                                                       ("TN", cm_s["tn"], "passed", TEAL))))):
        y = 2.35 + r * 1.3
        text(s, 0.7, y, 1.3, 1.2, row_lab, size=15, colour=MUTED, anchor=MSO_ANCHOR.MIDDLE)
        for c, (abbr, n, word, col) in enumerate(cells_):
            x = 2.05 + c * 1.85
            box(s, x, y, 1.75, 1.2, fill=PANEL, line=col, line_w=1.75)
            text(s, x, y + 0.08, 1.75, 0.6, [[(f"{abbr} ", {"colour": col, "bold": True}),
                                              (str(n), {"bold": True})]], size=24, align=PP_ALIGN.CENTER)
            text(s, x, y + 0.72, 1.75, 0.35, word, size=14, colour=MUTED, align=PP_ALIGN.CENTER)
    box(s, 0.7, 5.1, 4.9, 1.55, fill=PANEL2, line=ROSE)
    text(s, 0.95, 5.2, 4.45, 1.4, [[("Factory view: ", {"bold": True, "colour": ROSE}),
                                    ("recall matters most. A missed defect (FN) reaches the customer; a false "
                                     "alarm (FP) only costs a manual re-check.", {})]], size=15,
         anchor=MSO_ANCHOR.MIDDLE)
    rows_m = [("Precision", "TP / (TP + FP)", "Of what I flagged, how much was really defective?", INDIGO),
              ("Recall", "TP / (TP + FN)", "Of all real defects, how many did I catch?", INDIGO),
              ("F1", "2·P·R / (P + R)", "High only if precision and recall are both high", INDIGO),
              ("AUROC", "area under ROC", "Chance a defect outscores a normal; 1 = perfect, 0.5 = random", AMBER),
              ("IoU", "overlap / union", "How well the predicted mask matches the true defect", TEAL),
              ("Dice", "2·overlap / (pred + true)", "Like IoU, always ≥ IoU", TEAL)]
    for i, (name, formula, meaning, col) in enumerate(rows_m):
        y = 1.6 + i * 0.84
        box(s, 5.85, y, 6.78, 0.74, fill=PANEL, line=LINE)
        box(s, 5.85, y, 0.1, 0.74, fill=col, shape=MSO_SHAPE.RECTANGLE)
        text(s, 6.1, y, 1.25, 0.74, name, size=16, bold=True, colour=col, anchor=MSO_ANCHOR.MIDDLE)
        text(s, 7.35, y, 2.3, 0.74, formula, size=15, colour=WHITE, anchor=MSO_ANCHOR.MIDDLE)
        text(s, 9.7, y, 2.83, 0.74, meaning, size=14, colour=MUTED, anchor=MSO_ANCHOR.MIDDLE)

    # ---------------------------------------------------------------- 15 metrics
    s = new_slide(prs, 15, "Result Analysis: Metrics", "Same 20 test images for all four models", (
        f"These are the results on the {v2info['test_images']} test images of v2. EfficientAD-S, our main model, "
        f"reaches {s_img['accuracy']:.0%} accuracy, precision, recall and F1, image AUROC {s_img['auroc']:.2f}, "
        f"and the best pixel IoU of all models, {s_px['iou']:.3f}, meaning its defect masks fit best. Both "
        "PatchCore baselines classify all 20 test images correctly because they need no training. EfficientAD-M "
        "had too few iterations and misses five defects. PatchCore's masks cover the defect but are too large, "
        "because of the coarse 28 by 28 grid and the blur. A common question: why is pixel AUROC about 0.98 but "
        f"IoU only about {m['patchcore_r18']['pixel_level']['iou']:.2f}? AUROC measures how well defect pixels are "
        "ranked above normal ones over all thresholds; IoU uses one fixed threshold, and at that threshold the "
        "masks spread around the defect. Tuning it on the test masks would be cheating. EfficientAD is slow here "
        "only because it runs on a CPU. With only 20 test images, one image changes AUROC by 0.05 to 0.10, so "
        "small differences are not significant."))
    big = [(f"{s_img['accuracy']:.2f}", "Accuracy", INDIGO), (f"{s_img['precision']:.2f}", "Precision", INDIGO),
           (f"{s_img['recall']:.2f}", "Recall", TEAL), (f"{s_img['f1']:.2f}", "F1 score", TEAL),
           (f"{s_img['auroc']:.2f}", "Image AUROC", AMBER), (f"{s_px['iou']:.3f}", "Pixel IoU (best)", AMBER)]
    text(s, 0.7, 1.55, 3.6, 0.4, "EfficientAD-S (main model)", size=18, bold=True, colour=WHITE)
    for i, (v, lab, col) in enumerate(big):
        cx, cy = 0.7 + (i % 2) * 1.82, 2.05 + (i // 2) * 1.5
        stat(s, cx, cy, 1.7, 1.35, v, lab, accent=col, value_size=30, label_size=14)
    picture(s, c_metrics, 4.55, 1.6, 8.1, 4.3, frame=LINE)
    rows_txt = []
    for key, label in MODELS:
        il = m[key]["image_level"]["confusion_matrix"]
        rows_txt.append(f"{label}: {il['tp'] + il['tn']}/20 correct")
    text(s, 4.55, 6.12, 8.1, 0.6, "   ·   ".join(rows_txt), size=14, colour=MUTED, align=PP_ALIGN.CENTER)

    # ---------------------------------------------------------------- 13 screenshots
    s = new_slide(prs, 13, "Result Analysis: Screenshots", "What the models see", (
        "Top left: all four models on the same broken bottle next to the ground truth. EfficientAD's heatmaps hug "
        "the crack, while PatchCore's cover a much larger area. Top right: EfficientAD-S's mask against the "
        "annotation, where yellow means correctly found. Bottom left: the ROC curves of all four models and the "
        "confusion matrix of EfficientAD-S, with one miss and one false alarm. Bottom right: the real training loss "
        "of EfficientAD, which falls steadily over the iterations."))
    cells = [
        (img_compare, "Same broken bottle, all 4 models vs ground truth: EfficientAD masks follow the crack, "
                      "PatchCore marks a much larger area."),
        (img_gtcomp, "EfficientAD-S: input and mask vs annotation (yellow = found), IoU 0.556."),
        (img_analytics, "ROC of all 4 models; EfficientAD-S confusion matrix 9 / 1 / 1 / 9."),
        (c_loss, f"EfficientAD-S loss: "
                 f"{meta['efficientad_s']['summary']['loss_history'][0]['loss_total']:.1f} → "
                 f"{meta['efficientad_s']['summary']['loss_history'][-1]['loss_total']:.2f} over "
                 f"{es_cfg['total_iters']:,} iterations.")]
    picture(s, cells[0][0], 0.7, 1.5, 11.93, 2.45, frame=TEAL)
    text(s, 0.7, 4.02, 11.93, 0.32, cells[0][1], size=14, colour=MUTED, align=PP_ALIGN.CENTER)
    for i, (img, cap) in enumerate(cells[1:]):
        cx = 0.7 + i * 4.04
        picture(s, img, cx, 4.48, 3.85, 1.72, frame=INDIGO if i != 2 else TEAL)
        text(s, cx, 6.28, 3.85, 0.55, cap, size=14, colour=MUTED, align=PP_ALIGN.CENTER)

    # ---------------------------------------------------------------- 14 future scope
    s = new_slide(prs, 14, "Future Scope", "Realistic next steps", (
        "The most important next step is to train EfficientAD with the full 70,000 iterations on a GPU, which "
        "should close the gap to PatchCore in detection, especially for the M model. We would also extend the "
        "evaluation to all 15 MVTec AD categories and a larger test set, so the numbers become more reliable. "
        "Finally, the models can be deployed on a GPU or edge device on a real production line, with images "
        "from the line flowing back into the Roboflow dataset."))
    future = [("Full EfficientAD training", "Paper protocol of 70,000 iterations on a GPU, especially for "
                                            "EfficientAD-M.", INDIGO),
              ("More product categories", "Extend from bottle to all 15 MVTec AD categories.", TEAL),
              ("Larger test set", "More test images per defect type for more reliable metrics.", AMBER),
              ("GPU / edge deployment", "Real-time inspection on a production line (EfficientAD is designed for "
                                        "millisecond GPU latency).", INDIGO),
              ("Data feedback loop", "Add new line images to the Roboflow project and retrain periodically.", TEAL)]
    for i, (head, body, col) in enumerate(future):
        y = 1.6 + i * 1.04
        box(s, 0.7, y, 11.93, 0.9, fill=PANEL, line=LINE)
        box(s, 0.7, y, 0.12, 0.9, fill=col, shape=MSO_SHAPE.RECTANGLE)
        text(s, 1.05, y, 3.9, 0.9, head, size=19, bold=True, colour=col, anchor=MSO_ANCHOR.MIDDLE)
        text(s, 5.0, y, 7.45, 0.9, body, size=16, anchor=MSO_ANCHOR.MIDDLE)

    # ---------------------------------------------------------------- 15 final verdict
    label = dict(MODELS)

    def winners(get, low=False):
        values = {k: get(m[k]) for k, _ in MODELS}
        best = min(values.values()) if low else max(values.values())
        return [k for k, v in values.items() if abs(v - best) < 1e-9], best

    criteria = [
        ("Detection (image AUROC, accuracy)", lambda x: x["image_level"]["auroc"], False, "{:.2f}"),
        ("Pixel AUROC (ranking defect pixels)", lambda x: x["pixel_level"]["auroc"], False, "{:.4f}"),
        ("Mask fit (pixel F1 / IoU)", lambda x: x["pixel_level"]["iou"], False, "IoU {:.3f}"),
        ("Speed on CPU (mean latency)", lambda x: x["latency_ms"]["mean"], True, "{:.0f} ms"),
    ]
    wn = {}
    for name, get, low, fmt in criteria:
        wn[name] = winners(get, low)
    overall = "patchcore_wrn50"
    assert overall in wn[criteria[0][0]][0] and wn[criteria[1][0]][0] == [overall], "verdict no longer matches data"
    assert wn[criteria[2][0]][0] == ["efficientad_s"] and wn[criteria[3][0]][0] == ["patchcore_r18"]
    wrn_i, wrn_p = m[overall]["image_level"], m[overall]["pixel_level"]
    s = new_slide(prs, 15, "Final Verdict", "Which model performs best, and why", (
        f"Here is the final verdict. Looking at every criterion, PatchCore with WideResNet-50 is the best performing "
        f"model overall: it classifies all 20 test images correctly and has the highest pixel AUROC, "
        f"{wrn_p['auroc']:.4f}. It wins because PatchCore needs no gradient training, so it reaches its full "
        f"strength immediately, and the larger backbone gives richer features than ResNet18. EfficientAD-S is the "
        f"best at outlining defects, with pixel IoU {s_px['iou']:.3f}, but it was trained for only "
        f"{es_cfg['total_iters']:,} of the paper's 70,000 iterations. If speed on a CPU matters most, PatchCore "
        f"ResNet18 is the choice."))
    text(s, 0.7, 1.55, 6.3, 0.4, "Winner per criterion (20 test images)", size=18, bold=True, colour=TEAL)
    for i, (name, get, low, fmt) in enumerate(criteria):
        keys, best = wn[name]
        y = 2.05 + i * 1.02
        col = AMBER if keys == [overall] or overall in keys else (INDIGO if keys == ["efficientad_s"] else TEAL)
        box(s, 0.7, y, 6.3, 0.88, fill=PANEL, line=LINE)
        box(s, 0.7, y, 0.12, 0.88, fill=col, shape=MSO_SHAPE.RECTANGLE)
        text(s, 1.0, y + 0.08, 5.9, 0.35, name, size=15, colour=MUTED)
        text(s, 1.0, y + 0.42, 5.9, 0.4, [[(" & ".join(label[k] for k in keys), {"bold": True, "colour": WHITE}),
                                            ("   " + fmt.format(best), {"bold": True, "colour": col})]], size=18)
    box(s, 7.3, 1.55, 5.33, 2.85, fill=PANEL, line=AMBER, line_w=2.0)
    box(s, 7.3, 1.55, 5.33, 0.75, fill=AMBER, radius=0.15)
    box(s, 7.3, 1.95, 5.33, 0.35, fill=AMBER, shape=MSO_SHAPE.RECTANGLE)
    text(s, 7.3, 1.55, 5.33, 0.75, "Best overall: PatchCore · WRN-50", size=21, bold=True, colour=BG,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    bullets(s, 7.55, 2.48, 4.9, 2.3, [
        f"{wrn_i['confusion_matrix']['tp'] + wrn_i['confusion_matrix']['tn']} / 20 correct, image AUROC "
        f"{wrn_i['auroc']:.2f}",
        f"Highest pixel AUROC: {wrn_p['auroc']:.4f}",
        "No training needed: full strength at once",
        "WideResNet-50 features richer than ResNet18"], size=17, bullet_colour=AMBER, gap=6)
    box(s, 7.3, 4.6, 5.33, 2.05, fill=PANEL2, line=INDIGO, line_w=1.5)
    text(s, 7.55, 4.75, 4.9, 0.4, "Main model: EfficientAD-S", size=19, bold=True, colour=INDIGO)
    text(s, 7.55, 5.22, 4.9, 1.35, f"Best defect outlines (pixel IoU {s_px['iou']:.3f}, F1 {s_px['f1']:.3f}) "
                                   f"but trained for only {es_cfg['total_iters']:,} of 70,000 iterations, so it "
                                   f"trails in detection ({s_img['auroc']:.2f}).", size=16)
    em_cm = m["efficientad_m"]["image_level"]["confusion_matrix"]
    text(s, 0.7, 6.18, 6.3, 0.45, f"Weakest: EfficientAD-M ({em_cfg['total_iters']:,} iterations, "
                                  f"{em_cm['tp'] + em_cm['tn']} / 20 correct)", size=15, colour=MUTED)

    # ---------------------------------------------------------------- 16 conclusion
    s = new_slide(prs, 16, "Conclusion", "Key takeaways", (
        "To conclude: we built our own dataset on Roboflow and showed, with v1 and v2, that augmentation does not "
        "help this kind of model. We trained EfficientAD on a GPU and compared it fairly with two PatchCore "
        "baselines. EfficientAD-S gives the most precise defect outlines, PatchCore gives perfect detection on our "
        "test set without any training, and everything is available in a working inspection platform. Thank you, "
        "I am happy to take questions."))
    takeaways = [
        ("Own dataset", f"Roboflow, {total_v2} images, 80/10/10; v1 vs v2: augmentation did not help.", INDIGO),
        ("Best localisation", f"EfficientAD-S: pixel IoU {s_px['iou']:.3f} vs "
                              f"{m['patchcore_wrn50']['pixel_level']['iou']:.3f} for the best PatchCore.", TEAL),
        ("Best detection", "Both PatchCore baselines: 20 / 20 correct without any training.", AMBER),
        ("Working platform", "Web app with dataset switch, live inspection, comparison and PDF reports.", INDIGO)]
    for i, (head, body, col) in enumerate(takeaways):
        y = 1.6 + i * 1.14
        box(s, 0.7, y, 7.3, 1.02, fill=PANEL, line=LINE)
        box(s, 0.7, y, 0.12, 1.02, fill=col, shape=MSO_SHAPE.RECTANGLE)
        text(s, 1.0, y + 0.08, 6.85, 0.4, head, size=19, bold=True, colour=col)
        text(s, 1.0, y + 0.5, 6.85, 0.45, body, size=16)
    box(s, 8.3, 1.6, 4.33, 4.44, fill=INDIGO, radius=0.06)
    text(s, 8.3, 2.1, 4.33, 1.2, "Thank You", size=40, bold=True, colour=WHITE, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, 8.3, 3.3, 4.33, 0.7, "Questions?", size=28, bold=True, colour=BG, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, 8.5, 4.3, 3.93, 1.4, [NAME, f"Reg. No. {REG}", f"Guide: {GUIDE}"], size=17, colour=WHITE,
         align=PP_ALIGN.CENTER, space_after=2)
    text(s, 0.7, 6.3, 11.93, 0.5, [[("Limitation: ", {"bold": True, "colour": AMBER}),
                                      ("EfficientAD used a reduced protocol "
                                       f"({es_cfg['total_iters']:,} / {em_cfg['total_iters']:,} of the paper's "
                                       f"70,000 iterations) and the test set has {v2info['test_images']} images.",
                                       {})]], size=15, colour=MUTED)

    prs.save(OUT)
    print(f"saved {OUT} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build()
