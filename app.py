"""Industrial AI Inspection Platform - Streamlit app.

Run:  streamlit run app.py   ->  http://localhost:8501
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image

import dataset as ds
import history
import model_registry as reg
import ui
import visualization as viz
from evaluation import build_model, evaluate, load_metrics, results_dir
from report import build_pdf

PROJECT = Path(__file__).resolve().parent
PAGES = ["Overview", "Live Inspection", "Model Laboratory", "Model Comparison", "Analytics",
         "Dataset Explorer", "Inspection History", "Reports", "Documentation", "Settings"]
ICONS = ["◈", "◎", "⚗", "⇄", "▤", "▦", "↺", "⎙", "☰", "⚙"]
DOCS = [("Page guide", "docs/PAGE_GUIDE.md"), ("README", "README.md"), ("Demo guide", "docs/DEMO_GUIDE.md"),
        ("Project explanation", "docs/PROJECT_EXPLANATION.md"), ("Viva Q&A", "docs/VIVA.md")]
MODES = ["Original", "Anomaly heatmap", "Predicted mask", "Heatmap overlay", "Ground-truth mask",
         "Ground-truth comparison"]
GT_MODES = {"Ground-truth mask", "Ground-truth comparison"}

st.set_page_config(page_title="Industrial AI Inspection Platform", page_icon="◎", layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_css()
st.session_state.setdefault("nav", "Overview")
st.session_state.setdefault("preds", {})
st.session_state.setdefault("current", None)
st.session_state.setdefault("threads", 4)
st.session_state.setdefault("heat_alpha", 0.45)
torch.set_num_threads(st.session_state["threads"])


# ============================================================== data access
@st.cache_data(show_spinner=False)
def category_info(root: str, category: str) -> dict:
    return ds.validate_category(Path(root), category)


@st.cache_data(show_spinner=False)
def test_index(root: str, category: str) -> dict[str, ds.Sample]:
    return {s.name: s for s in ds.test_samples(Path(root), category)}


@st.cache_resource(show_spinner="Loading model ...", max_entries=3)
def cached_model(key: str, category: str, checkpoint: str, stamp: float):
    return reg.load(key, category)


def mem_available_mb() -> int:
    with open("/proc/meminfo") as handle:
        for line in handle:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) // 1024
    return 0


def get_model(key: str, category: str):
    """Load a model, refusing to load the heavy WRN-50 when RAM is short."""
    status = reg.status(key, category)
    if not status["checkpoint_available"]:
        return None
    ckpt = Path(status["checkpoint"])
    if reg.SPECS[key].source == "main" and mem_available_mb() < 1500:
        cached_model.clear()  # unload previously used models first, then re-check
        gc.collect()
    if reg.SPECS[key].source == "main" and mem_available_mb() < 1500:
        st.warning(f"Only {mem_available_mb()} MB RAM available - not loading {reg.SPECS[key].label} "
                   "(needs ~1.2 GB). Close other applications and retry.")
        return None
    return cached_model(key, category, str(ckpt), ckpt.stat().st_mtime)


def available_keys(category: str) -> list[str]:
    return [k for k in reg.SPECS if reg.status(k, category)["checkpoint_available"]]


def go_to(page: str) -> None:
    st.session_state["nav"] = page


def predict(key: str, category: str, image_id: str, image: np.ndarray) -> dict | None:
    """Cached per (image, model): switching view modes never re-runs inference."""
    cache = st.session_state["preds"]
    if (image_id, key) not in cache:
        model = get_model(key, category)
        if model is None:
            return None
        cache[(image_id, key)] = model.predict(image)
        while len(cache) > 12:
            cache.pop(next(iter(cache)))
    return cache[(image_id, key)]


def current_image() -> tuple[np.ndarray | None, ds.Sample | None]:
    current = st.session_state["current"]
    if not current:
        return None, None
    if current["source"] == "mvtec":
        sample = test_index(ROOT, current["category"]).get(current["name"])
        return (ds.load_rgb(sample.path), sample) if sample else (None, None)
    return current["array"], None


def set_current_sample(category: str, name: str) -> None:
    st.session_state["current"] = {"source": "mvtec", "category": category, "name": name,
                                   "image_id": f"{category}/{name}", "label": f"{category}/{name}"}


# ================================================================ sidebar
DATASETS = dict(ds.available_datasets())
if not DATASETS:
    st.error("No dataset found. Convert the Roboflow export (scripts/convert_roboflow_export.py) or set MVTEC_ROOT.")
    st.stop()
if "dataset" not in st.session_state or st.session_state["dataset"] not in DATASETS:
    launch = Path(ds.LAUNCH_ROOT).resolve() if ds.LAUNCH_ROOT else None
    st.session_state["dataset"] = next((label for label, root in DATASETS.items()
                                        if launch and root.resolve() == launch), next(iter(DATASETS)))
os.environ["MVTEC_ROOT"] = str(DATASETS[st.session_state["dataset"]])  # read by dataset / model_registry
ROOT = str(ds.find_dataset_root())


def on_dataset_change() -> None:
    """Predictions and the selected image belong to one dataset: drop them when switching."""
    st.session_state["preds"] = {}
    st.session_state["current"] = None

CATEGORIES = ds.available_categories(Path(ROOT))
DS = ds.describe()
with st.sidebar:
    ui.html_block(f'<div style="display:flex;gap:10px;align-items:center;margin:4px 0 6px 2px">{ui.LOGO}'
                  '<div><div style="font-weight:700;color:#f1f5f9">Inspection Platform</div>'
                  f'<div style="font-size:.72rem;color:#64748b">Industrial AI · {DS["short"]}</div></div></div>')
    ui.html_block('<div class="nav-caption">Navigation</div>')
    st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed",
             format_func=lambda p: f"{ICONS[PAGES.index(p)]}  {p}")
    ui.html_block('<div class="nav-caption">Workspace</div>')
    st.selectbox("Dataset", list(DATASETS), key="dataset", on_change=on_dataset_change,
                 help="Roboflow v2 is the final dataset (all 4 models). Roboflow v1 has the 7 augmentations "
                      "(PatchCore · ResNet18 only, for the augmentation comparison).")
    CATEGORY = st.selectbox("Product category", CATEGORIES,
                            index=CATEGORIES.index("bottle") if "bottle" in CATEGORIES else 0, key="category")
    AVAILABLE = available_keys(CATEGORY) or ["patchcore_r18"]
    st.selectbox("Active model", AVAILABLE, key="model_key", format_func=lambda k: reg.SPECS[k].label,
                 help="Used by Live Inspection, Analytics and Reports.")
    MODEL_KEY = st.session_state["model_key"]
    ui.html_block(f'<div class="nav-caption">Status</div><div class="mono" style="padding-left:4px">'
                  f'RAM available {mem_available_mb():,} MB<br>torch threads {torch.get_num_threads()} · CPU</div>')

INFO = category_info(ROOT, CATEGORY)
STATUSES = {k: reg.status(k, CATEGORY) for k in reg.SPECS}
ACTIVE_STATUS = STATUSES[MODEL_KEY]
N_READY = sum(s["checkpoint_available"] for s in STATUSES.values())
ui.topbar(f"{DS['short']} · {CATEGORY}", reg.SPECS[MODEL_KEY].label,
          f"{N_READY}/{len(STATUSES)} models available" if ACTIVE_STATUS["checkpoint_available"] else
          "active model not built", ACTIVE_STATUS["checkpoint_available"])


def evaluated() -> list[dict]:
    return [s for s in STATUSES.values() if s["evaluated"]]


# ================================================================ helpers
def metric_cards(status: dict) -> str:
    m = status["metrics"]
    i, p, lat = m["image_level"], m["pixel_level"], m["latency_ms"]
    cells = [("Image AUROC", ui.fmt(i["auroc"])), ("Pixel AUROC", ui.fmt(p["auroc"])),
             ("Precision (img)", ui.fmt(i["precision"])), ("Recall (img)", ui.fmt(i["recall"])),
             ("F1 (img / px)", f"{i['f1']:.3f} / {p['f1']:.3f}"), ("IoU (px)", ui.fmt(p["iou"])),
             ("Latency", f"{lat['mean']:.0f} ms"), ("Peak RAM", f"{m.get('peak_rss_mb', 0):,.0f} MB")]
    return '<div class="metric-grid">' + "".join(
        f'<div><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in cells) + "</div>"


def model_card(status: dict, compact: bool = False) -> str:
    spec = status["spec"]
    colour = ui.MODEL_COLORS[spec.key]
    ck = ui.badge("checkpoint ✓", "green") if status["checkpoint_available"] else ui.badge("no checkpoint", "amber")
    ev = ui.badge("evaluated ✓", "cyan") if status["evaluated"] else ui.badge("not evaluated", "amber")
    if status["evaluated"]:
        body = metric_cards(status)
    else:
        reason = status["reason"] or "Checkpoint available but not evaluated yet - run it from the Model Laboratory."
        body = f'<div class="unavailable" style="margin-top:8px">Metrics unavailable. {reason}</div>'
    arch = "" if compact else f'<div class="arch">{spec.description}</div>'
    return (f'<div class="card model-card" style="border-top:3px solid {colour}"><h4>{spec.label}</h4>'
            f'<div style="margin:6px 0 4px 0">{ck}{ev}</div>{arch}{body}</div>')


def backbone_mb(key: str) -> float:
    names = {"patchcore_r18": "resnet18-f37072fd.pth", "patchcore_wrn50": "wide_resnet50_2-95faca4d.pth"}
    path = Path(torch.hub.get_dir()) / "checkpoints" / names.get(key, "-")
    return path.stat().st_size / 2**20 if path.is_file() else 0.0


def bar_chart(rows: list[dict], fields: list[tuple[str, str]], title: str, yrange=None, fmt=".4f",
              height: int = 300) -> go.Figure:
    fig = go.Figure()
    for field, name in fields:
        single = len(fields) == 1
        fig.add_trace(go.Bar(
            x=[r["short"] for r in rows], y=[r[field] for r in rows], name=name,
            marker_color=[r["colour"] for r in rows] if single else None,
            text=[format(r[field], fmt) for r in rows] if single else None, textposition="outside",
            cliponaxis=False, textfont=dict(size=12, color="#e2e8f0"),
            hovertemplate="%{x}<br>" + name + ": %{y:" + fmt + "}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.35)
    fig.update_xaxes(tickangle=0)
    if yrange:
        fig.update_yaxes(range=yrange)
    return ui.style_fig(fig, height, title, legend=len(fields) > 1)


def comparison_rows() -> list[dict]:
    """One row per *evaluated* model; unavailable models never enter charts or rankings."""
    rows = []
    for status in evaluated():
        m, key = status["metrics"], status["key"]
        rows.append({
            "key": key, "label": reg.SPECS[key].label, "colour": ui.MODEL_COLORS[key],
            "short": reg.SPECS[key].label.replace("PatchCore · ", "PatchCore<br>"),
            "image_auroc": m["image_level"]["auroc"], "pixel_auroc": m["pixel_level"]["auroc"],
            "img_precision": m["image_level"]["precision"], "img_recall": m["image_level"]["recall"],
            "img_f1": m["image_level"]["f1"], "px_precision": m["pixel_level"]["precision"],
            "px_recall": m["pixel_level"]["recall"], "px_f1": m["pixel_level"]["f1"],
            "px_iou": m["pixel_level"]["iou"], "lat_mean": m["latency_ms"]["mean"],
            "lat_median": m["latency_ms"]["median"], "lat_p95": m["latency_ms"]["p95"],
            "fps": m["latency_ms"]["fps"], "ckpt_mb": status["checkpoint_mb"] or 0.0,
            "backbone_mb": backbone_mb(key), "peak_rss_mb": m.get("peak_rss_mb"),
            "images": m["num_test_images"], "evaluated_at": m["evaluated_at"],
            "threshold_source": m["model"].get("calibration_source", m["thresholds"]["source"]),
            "input": m["model"].get("input_size", 224), "grid": m["pixel_eval_grid"],
        })
    return rows


def run_and_save(key: str, image: np.ndarray, sample: ds.Sample | None, image_id: str, label: str):
    """Inference + one history row.  Called only from the Run button, never on reruns."""
    result = predict(key, CATEGORY, image_id, image)
    if result is None:
        return None
    gt = ds.load_mask(sample, image.shape[:2]) if sample else None
    agreement = viz.mask_agreement(result["mask"], gt) if gt is not None and gt.any() else {}
    over = viz.overlay(image, result["anomaly_map"], result["mask"], result["pixel_threshold"], gt)
    out_dir = history.OUTPUT_DIR / "history"
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = out_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{key}.png"
    Image.fromarray(ui.display_size(over, 640)).save(overlay_path)
    history.add({
        "category": CATEGORY, "source": "mvtec_test" if sample else "upload", "image": label,
        "score": result["score"], "image_threshold": result["image_threshold"],
        "prediction": "defective" if result["is_defective"] else "normal",
        "defect_area_pct": 100.0 * float(result["mask"].mean()), "latency_ms": result["latency_ms"],
        "ground_truth": ("defective" if sample.label else "normal") if sample else "unknown",
        "overlay_path": str(overlay_path), "model": key, "image_id": image_id,
        "iou": agreement.get("iou"), "dice": agreement.get("dice"),
        "num_regions": len(viz.regions(result["mask"])),
    })
    return result


def plot(fig: go.Figure, container=st, image: bool = False) -> None:
    config = ui.plotly_config() if image else ui.chart_config()
    container.plotly_chart(fig, use_container_width=True, config=config)


# ================================================================== Overview
def page_overview() -> None:
    ui.page_header("Overview", "Welcome to the inspection floor",
                   "Unsupervised defect detection: every model is built only from defect-free products and flags "
                   f"whatever deviates. All figures come from saved evaluation runs on the {DS['name']} {DS['split']}.")
    rows = comparison_rows()
    best = max(rows, key=lambda r: (r["pixel_auroc"], r["image_auroc"])) if rows else None
    hist = history.load()
    cols = st.columns(4)
    total = INFO["train_images"] + INFO["valid_images"] + INFO["test_images"]
    split = f"{INFO['train_images']} train · " + (f"{INFO['valid_images']} valid · " if INFO["valid_images"] else "")
    cols[0].markdown(ui.kpi("Dataset images", f"{total}",
                            f"{split}{INFO['test_images']} test · {INFO['masks']} masks",
                            ui.CYAN), unsafe_allow_html=True)
    missing = [reg.SPECS[k].label for k, s in STATUSES.items() if not s["checkpoint_available"]]
    cols[1].markdown(ui.kpi("Models available", f"{N_READY} / {len(STATUSES)}",
                            f"{len(rows)} evaluated" + (f" · missing: {', '.join(missing)}" if missing else ""),
                            ui.BLUE), unsafe_allow_html=True)
    cols[2].markdown(ui.kpi("Best pixel AUROC", ui.fmt(best["pixel_auroc"]) if best else "—",
                            best["label"] if best else "no evaluation yet", ui.GREEN), unsafe_allow_html=True)
    flagged_count = int((hist["prediction"] == "defective").sum()) if len(hist) else 0
    cols[3].markdown(ui.kpi("Inspections run", f"{len(hist)}", f"{flagged_count} flagged defective", ui.AMBER),
                     unsafe_allow_html=True)

    st.markdown("### Model availability")
    cols = st.columns(4)
    for col, status in zip(cols, STATUSES.values()):
        col.markdown(model_card(status, compact=True), unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        st.markdown("### Performance · evaluated models")
        if rows:
            c1, c2 = st.columns(2)
            plot(bar_chart(rows, [("image_auroc", "Image AUROC"), ("img_f1", "Image F1")],
                           "Image level · detection", [0, 1.1]), c1)
            plot(bar_chart(rows, [("pixel_auroc", "Pixel AUROC"), ("px_f1", "Pixel F1"), ("px_iou", "IoU")],
                           "Pixel level · localisation", [0, 1.1]), c2)
            plot(bar_chart(rows, [("lat_mean", "mean"), ("lat_median", "median"), ("lat_p95", "p95")],
                           "Inference latency per image (ms, CPU)", fmt=".0f", height=270))
        else:
            ui.html_block('<div class="unavailable">No evaluated models yet.</div>')
    with right:
        st.markdown(f"### Dataset · {CATEGORY}")
        tests = test_index(ROOT, CATEGORY).values()
        counts = [("train/good", INFO["train_images"]), ("test/good", INFO["test_normal"])] + [
            (f"test/{d}", sum(1 for s in tests if s.defect_type == d)) for d in INFO["defect_types"]]
        fig = go.Figure(go.Bar(x=[c for _, c in counts], y=[n for n, _ in counts], orientation="h",
                               text=[c for _, c in counts], textposition="outside", cliponaxis=False,
                               marker_color=[ui.GREEN, ui.GREEN] + [ui.RED] * len(INFO["defect_types"]),
                               hovertemplate="%{y}: %{x} images<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        plot(ui.style_fig(fig, 240, legend=False))
        st.markdown("### Recent inspection activity")
        if len(hist):
            recent = hist.head(6).assign(image=hist["image"].str.replace("MVTec test: ", "", regex=False))
            st.dataframe(recent[["timestamp", "image", "prediction", "score"]], hide_index=True,
                         use_container_width=True, column_config={"score": st.column_config.NumberColumn(format="%.3f")})
        else:
            ui.html_block('<div class="disabled-panel">No inspections yet. Open <b>Live Inspection</b> to run one.</div>')
    flagged = hist[(hist["prediction"] == "defective") & hist["overlay_path"].notna()].head(4) if len(hist) else hist
    if len(flagged):
        st.markdown("### Latest anomaly examples")
        cols = st.columns(4)
        for col, (_, row) in zip(cols, flagged.iterrows()):
            if Path(row["overlay_path"]).is_file():
                col.image(row["overlay_path"], use_container_width=True)
                label = reg.SPECS.get(row["model"] or "patchcore_r18", reg.SPECS["patchcore_r18"]).label
                col.markdown(f'<div class="thumb-cap">{row["image"]} · score {row["score"]:.2f} · {label}</div>',
                             unsafe_allow_html=True)


# =========================================================== Live Inspection
def page_inspection() -> None:
    ui.page_header("Live Inspection", "Interactive inspection workspace",
                   f"Pick a {DS['name']} test image (ground truth available) or upload your own, run the active "
                   "model, then inspect the result in six visualisation modes, a before/after slider and a "
                   "synchronized side-by-side view.")
    status = STATUSES[MODEL_KEY]
    if not status["checkpoint_available"]:
        ui.html_block(f'<div class="unavailable">{status["reason"]}</div>')
        if MODEL_KEY == "patchcore_r18":
            build_button()
        return

    c1, c2, c3 = st.columns([1.1, 2.2, 1])
    with c1:
        source = ui.segmented("Source", ["MVTec test image", "Upload"], "insp_source",
                              format_func=lambda o: DS["test_label"] if o == "MVTec test image" else o)
    current = st.session_state["current"]
    if source == "Upload":
        upload = c2.file_uploader("Image (PNG / JPG)", type=["png", "jpg", "jpeg", "bmp"], key="insp_upload")
        if upload is not None:
            data = upload.getvalue()
            image_id = "upload:" + hashlib.sha1(data).hexdigest()[:12]
            if not current or current["image_id"] != image_id:
                st.session_state["current"] = {"source": "upload", "category": CATEGORY, "name": upload.name,
                                               "image_id": image_id, "label": f"upload: {upload.name}",
                                               "array": np.asarray(Image.open(upload).convert("RGB"))}
        elif current and current["source"] != "upload":
            st.session_state["current"] = None
    else:
        names = list(test_index(ROOT, CATEGORY))
        selected = current["name"] if current and current["source"] == "mvtec" and current["name"] in names else names[0]
        if st.session_state.get("insp_sample") != selected and st.session_state.get("_insp_synced") != selected:
            st.session_state["insp_sample"] = selected  # sync the widget with the cross-page selection
        st.session_state["_insp_synced"] = selected
        choice = c2.selectbox(f"{DS['test_label']} (defect type / file)", names, key="insp_sample")
        if not current or current.get("name") != choice or current["source"] != "mvtec":
            set_current_sample(CATEGORY, choice)
            st.session_state["_insp_synced"] = choice
    current = st.session_state["current"]
    image, sample = current_image()
    if image is None:
        ui.html_block('<div class="disabled-panel">Upload an image of the selected product type to begin.</div>')
        return
    c3.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    run = c3.button(f"▶  Run {reg.SPECS[MODEL_KEY].label}", type="primary", use_container_width=True, key="run_insp")
    if run:
        with st.spinner("Running inference ..."):
            result = run_and_save(MODEL_KEY, image, sample, current["image_id"], current["label"])
        if result is not None:
            st.toast(f"Saved to history · {'DEFECTIVE' if result['is_defective'] else 'NORMAL'} "
                     f"({result['score']:.3f})", icon="✅")
    result = st.session_state["preds"].get((current["image_id"], MODEL_KEY))
    if result is None:
        left, right = st.columns([2, 1])
        plot(ui.image_figure(ui.display_size(image), 480), left, image=True)
        truth = ("defective · " + sample.defect_type if sample.label else "normal") if sample else "not available (upload)"
        right.markdown(ui.kpi("Selected image", current["name"], f"ground truth: {truth}", ui.CYAN),
                       unsafe_allow_html=True)
        right.info("Press **Run** to analyse this image with the active model.")
        return
    render_workspace(image, sample, result, current)


def render_workspace(image: np.ndarray, sample: ds.Sample | None, result: dict, current: dict) -> None:
    has_gt = sample is not None
    gt = ds.load_mask(sample, image.shape[:2]) if has_gt else None
    pthr, ithr = result["pixel_threshold"], result["image_threshold"]
    amap, mask = result["anomaly_map"], result["mask"]
    regions = viz.regions(mask)
    agreement = viz.mask_agreement(mask, gt) if has_gt and gt.any() else None
    alpha = st.session_state["heat_alpha"]
    disp = ui.display_size(image)
    scale = disp.shape[1] / image.shape[1]
    heat_top = 2.0 * pthr

    main, side = st.columns([2.15, 1])
    with side:
        defective = bool(result["is_defective"])
        ui.html_block(f'<div class="verdict {"v-bad" if defective else "v-ok"}"><span>'
                      f'{"⚠ DEFECTIVE" if defective else "✓ NORMAL"}</span>'
                      f'<span style="font-size:.8rem;font-weight:500">{reg.SPECS[MODEL_KEY].label}</span></div>')
        if sample:
            truth = "defective · " + sample.defect_type if sample.label else "normal"
            ok = (sample.label == 1) == defective
            ui.html_block(f'<div style="margin:8px 0 4px 2px;font-size:.85rem;color:#94a3b8">Ground truth: '
                          f'<b style="color:#e2e8f0">{truth}</b> → '
                          f'{ui.badge("correct", "green") if ok else ui.badge("wrong", "red")}</div>')
        k1, k2 = st.columns(2)
        k1.metric("Anomaly score", f"{result['score']:.3f}")
        k2.metric("Image threshold", f"{ithr:.3f}")
        k1.metric("Defect area", f"{100 * float(mask.mean()):.2f} %")
        k2.metric("Latency", f"{result['latency_ms']:.0f} ms")
        k1.metric("Regions", len(regions))
        k2.metric("Pixel threshold", f"{pthr:.3f}")
        if agreement:
            k1.metric("IoU vs GT", ui.fmt(agreement["iou"], 3))
            k2.metric("Dice vs GT", ui.fmt(agreement["dice"], 3))
        elif has_gt:
            st.caption("IoU / Dice undefined: this image has no defect in its ground truth.")
        else:
            st.caption("IoU / Dice unavailable: uploaded images have no ground-truth annotation.")
        ratio = min(result["score"] / (2 * ithr), 1.0)
        ui.html_block(f'<div style="margin:10px 0 2px 0;font-size:.75rem;color:#94a3b8">Score vs. threshold</div>'
                      f'<div style="position:relative;height:10px;border-radius:5px;background:linear-gradient(90deg,'
                      f'rgba(34,197,94,.55) 0 50%,rgba(239,68,68,.55) 50% 100%)">'
                      f'<div style="position:absolute;left:calc({ratio * 100:.1f}% - 2px);top:-4px;width:4px;'
                      f'height:18px;background:#f8fafc;border-radius:2px"></div></div>'
                      f'<div class="legend" style="justify-content:space-between"><span>0</span>'
                      f'<span style="color:#fbbf24">threshold {ithr:.3f}</span><span>{2 * ithr:.2f}</span></div>')
        st.button("⇄  Compare all models on this image", on_click=go_to, args=("Model Comparison",),
                  use_container_width=True)
        st.button("▤  Open analytics", on_click=go_to, args=("Analytics",), use_container_width=True)
        st.button("⎙  Report with this inspection", on_click=go_to, args=("Reports",), use_container_width=True)

    with main:
        mode = ui.segmented("Visualisation mode", MODES, "viz_mode", default="Heatmap overlay")
        if mode in GT_MODES and not has_gt:
            ui.html_block('<div class="disabled-panel" style="padding:60px 16px">⊘ <b>Ground truth unavailable</b><br>'
                          f'Ground-truth masks exist only for {DS["name"]} test images. Uploaded images have no '
                          'annotation, so this mode is disabled. The model output is still shown in the other modes.</div>')
        elif mode == "Original":
            plot(ui.image_figure(disp, 540), image=True)
        elif mode == "Anomaly heatmap":
            plot(ui.image_figure(disp, 560, heat=amap, heat_range=(0.0, heat_top), threshold=pthr), image=True)
            ui.heat_legend(pthr, heat_top)
        elif mode == "Predicted mask":
            plot(ui.image_figure(ui.display_size(viz.coloured_mask(mask, (239, 68, 68))), 540, boxes=regions, scale=scale), image=True)
            ui.legend([(ui.RED, "predicted anomaly (model output)"), ("#fbbf24", "connected-region bounding box")])
        elif mode == "Heatmap overlay":
            over = viz.overlay(image, amap, mask, pthr, None, alpha)
            plot(ui.image_figure(ui.display_size(over), 540, boxes=regions, scale=scale), image=True)
            ui.heat_legend(pthr, heat_top)
            ui.legend([(ui.RED, "predicted mask outline"), ("#fbbf24", "region bounding box")])
        elif mode == "Ground-truth mask":
            plot(ui.image_figure(ui.display_size(viz.coloured_mask(gt, (34, 197, 94))), 540), image=True)
            ui.legend([(ui.GREEN, "ground-truth defect (human annotation - not a model output)")])
            if not gt.any():
                st.caption("Defect-free image: its ground-truth mask is empty.")
        else:  # Ground-truth comparison
            comparison = viz.gt_comparison(image, mask, gt)
            g = st.columns(4)
            g[0].image(ui.display_size(image, 480), caption="Original", use_container_width=True)
            g[1].image(ui.display_size(viz.coloured_mask(mask, (239, 68, 68)), 480), caption="Predicted mask (model)",
                       use_container_width=True)
            g[2].image(ui.display_size(viz.coloured_mask(gt, (34, 197, 94)), 480),
                       caption="Ground-truth mask (annotation)", use_container_width=True)
            g[3].image(ui.display_size(comparison, 480), caption="Prediction vs. ground truth", use_container_width=True)
            plot(ui.image_figure(ui.display_size(comparison), 520), image=True)
            ui.legend([(ui.RED, "predicted only (false positive)"), (ui.GREEN, "ground truth only (missed)"),
                       ("#facc15", "overlap (correctly found)")])
            if agreement:
                st.caption(f"IoU {agreement['iou']:.3f} · Dice {agreement['dice']:.3f} · overlap {agreement['tp_px']:,} px · "
                           f"false positive {agreement['fp_px']:,} px · missed {agreement['fn_px']:,} px")

        t1, t2, t3 = st.tabs(["Before / after slider", "Synchronized view", f"Defect regions ({len(regions)})"])
        with t1:
            options = ["Heatmap overlay", "Anomaly heatmap", "Predicted mask"] + (["Ground-truth comparison"] if has_gt else [])
            after_mode = st.radio("Compare the original with", options, horizontal=True, key="slider_mode")
            after = {"Heatmap overlay": lambda: viz.overlay(image, amap, mask, pthr, None, alpha),
                     "Anomaly heatmap": lambda: viz.heatmap(amap, pthr),
                     "Predicted mask": lambda: viz.coloured_mask(mask, (239, 68, 68)),
                     "Ground-truth comparison": lambda: viz.gt_comparison(image, mask, gt)}[after_mode]()
            slider_key = "cmp" + hashlib.sha1(f"{current['image_id']}{after_mode}{MODEL_KEY}".encode()).hexdigest()[:8]
            ui.compare_slider(image, after, "Original", after_mode, key=slider_key)
        with t2:
            right = viz.overlay(image, amap, mask, pthr, gt if has_gt else None, alpha)
            plot(ui.synced_figure(disp, ui.display_size(right),
                                  ("Original", "Model output" + (" · green = ground truth" if has_gt else ""))), image=True)
            st.caption("Zoom or pan either image and the other follows. Scroll to zoom, drag to pan, double-click to reset.")
        with t3:
            if regions:
                st.dataframe(pd.DataFrame([{
                    "region": f"R{i + 1}", "x": r["x0"], "y": r["y0"], "width": r["x1"] - r["x0"],
                    "height": r["y1"] - r["y0"], "area_px": r["area_px"], "area_%": round(r["area_pct"], 3),
                    "max_score": round(float(amap[r["y0"]:r["y1"], r["x0"]:r["x1"]].max()), 3)}
                    for i, r in enumerate(regions)]), hide_index=True, use_container_width=True)
                st.caption("Connected components (8-connectivity) of the predicted mask; specks under 0.05 % of the "
                           "image are ignored.")
            else:
                st.caption("No region exceeds the pixel threshold.")
    with st.expander("Technical details"):
        st.markdown(f"- Image `{current['label']}` ({image.shape[1]}×{image.shape[0]} px)\n"
                    f"- Model **{reg.SPECS[MODEL_KEY].label}**: {reg.SPECS[MODEL_KEY].description}\n"
                    "- Thresholds calibrated on normal validation images only (never on test data).\n"
                    "- The anomaly map is resized to the original resolution; mask = map ≥ pixel threshold.\n"
                    "- Heatmap colour scale is fixed per model (pixel threshold = middle colour), not per image.")


# ========================================================== Model Laboratory
def page_laboratory() -> None:
    ui.page_header("Model Laboratory", "Models, checkpoints and calibration",
                   "Inspect each model's configuration, memory bank and decision thresholds; build the ResNet18 model "
                   "or evaluate an available checkpoint. EfficientAD training is intentionally disabled.")
    key = ui.segmented("Model", list(reg.SPECS), "lab_model", default=MODEL_KEY,
                       format_func=lambda k: reg.SPECS[k].label)
    status, spec = STATUSES[key], reg.SPECS[key]
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(model_card(status), unsafe_allow_html=True)
        with st.expander("Strengths and limitations", expanded=True):
            st.markdown(f"**Strengths.** {spec.strengths}\n\n**Limitations.** {spec.limitations}")
    with right:
        st.markdown("### Checkpoint")
        st.markdown(f'<div class="mono">{status["checkpoint"]}</div>', unsafe_allow_html=True)
        if not status["checkpoint_available"]:
            ui.html_block(f'<div class="unavailable" style="margin-top:8px">{status["reason"]}</div>')
            if key == "patchcore_r18":
                build_button()
            return
        st.caption(f"{status['checkpoint_mb']} MB on disk + {backbone_mb(key):.0f} MB ImageNet backbone weights")
        model = get_model(key, CATEGORY)
        if model is None:
            return
        info = model.info
        st.markdown("### Memory bank & calibration")
        a, b = st.columns(2)
        a.metric("Memory bank", f"{info.get('memory_bank_size') or 0:,}", help="vectors kept by the coreset")
        b.metric("Patch vectors", f"{info.get('patch_vectors') or 0:,}")
        a.metric("Image threshold", f"{model.calibration.image_threshold:.4f}")
        b.metric("Pixel threshold", f"{model.calibration.pixel_threshold:.4f}")
        st.caption("Calibration: " + info.get("calibration_source", ""))
        scores = getattr(model.calibration, "validation_scores", None)
        if scores:
            fig = go.Figure(go.Scatter(x=scores, y=[0] * len(scores), mode="markers",
                                       marker=dict(size=11, color=ui.GREEN, line=dict(color="#0a0f1c", width=1)),
                                       hovertemplate="validation score %{x:.3f}<extra></extra>"))
            fig.add_vline(x=model.calibration.image_threshold, line=dict(color=ui.AMBER, dash="dash"),
                          annotation_text="threshold = max", annotation_font_color=ui.AMBER)
            fig.update_yaxes(visible=False)
            plot(ui.style_fig(fig, 150, f"{len(scores)} held-out normal images", legend=False))
    st.markdown("### Actions")
    a, b = st.columns(2)
    with a:
        label = (f"Re-run evaluation ({DS['split']})" if status["evaluated"] else f"Evaluate on the {DS['split']}")
        if st.button(label, key=f"eval_{key}"):
            if mem_available_mb() < 1200:
                st.error(f"Only {mem_available_mb()} MB RAM available - evaluation skipped to protect the system.")
            else:
                model = get_model(key, CATEGORY)  # None (with a warning) when RAM is too low
                if model is not None:
                    bar = st.progress(0.0, text="Evaluating ...")
                    evaluate(model, CATEGORY, model_key=key,
                             progress=lambda f: bar.progress(f, text=f"Evaluating ... {f:.0%}"))
                    st.toast("Evaluation saved", icon="✅")
                    st.rerun()
    with b:
        if key == "patchcore_r18":
            build_button(rebuild=True)
        else:
            st.caption("Main-platform checkpoints are used read-only and never retrained here.")


def build_button(rebuild: bool = False) -> None:
    label = "Rebuild memory bank (~4 min CPU)" if rebuild else "Build ResNet18 model (~4 min CPU)"
    if st.button(label, key=f"build_{rebuild}"):
        if mem_available_mb() < 1800:
            st.error(f"Only {mem_available_mb()} MB RAM available - build skipped to protect the system.")
            return
        bar = st.progress(0.0, text="Extracting features ...")
        log = st.empty()
        build_model(CATEGORY, progress=lambda f: bar.progress(min(f, 1.0), text="Building memory bank ..."),
                    log=lambda m: log.info(m))
        cached_model.clear()
        st.session_state["preds"] = {}
        st.toast("Model built - re-run the evaluation to refresh metrics", icon="✅")


# ========================================================== Model Comparison
def page_comparison() -> None:
    ui.page_header("Model Comparison", "Model comparison centre",
                   "PatchCore and EfficientAD side by side. Numbers appear only for models with a real checkpoint and a "
                   "saved evaluation; every evaluated model was scored by the same code on the same test images.")
    cols = st.columns(4)
    for col, status in zip(cols, STATUSES.values()):
        col.markdown(model_card(status), unsafe_allow_html=True)

    rows = comparison_rows()
    st.markdown("### Evaluation protocol")
    if rows:
        same = len({r["images"] for r in rows}) == 1
        st.dataframe(pd.DataFrame([{"model": r["label"], "test images": r["images"], "input": f"{r['input']} px",
                                    "pixel metrics": r["grid"], "thresholds": r["threshold_source"],
                                    "evaluated": r["evaluated_at"]} for r in rows]),
                     hide_index=True, use_container_width=True)
        st.caption(f"Dataset: {DS['name']} · {CATEGORY} · {DS['split']} ({INFO['test_images']} images: "
                   f"{INFO['test_normal']} normal, {INFO['test_defective']} defective). "
                   + ("All evaluated models used the identical image list and `evaluation.evaluate`. " if same else
                      "⚠ Models were evaluated on different numbers of images - do not compare them directly. ")
                   + "Models without a checkpoint are excluded from every chart and ranking.")

        st.markdown("### Metrics")
        c1, c2 = st.columns(2)
        plot(bar_chart(rows, [("image_auroc", "Image AUROC")], "Image-level AUROC", [0, 1.1]), c1)
        plot(bar_chart(rows, [("pixel_auroc", "Pixel AUROC")], "Pixel-level AUROC (axis from 0.9)", [0.9, 1.005]), c2)
        c1, c2 = st.columns(2)
        plot(bar_chart(rows, [("img_precision", "Image precision"), ("img_recall", "Image recall"),
                              ("px_precision", "Pixel precision"), ("px_recall", "Pixel recall")],
                       "Precision & recall", [0, 1.15], fmt=".3f"), c1)
        plot(bar_chart(rows, [("img_f1", "Image F1"), ("px_f1", "Pixel F1"), ("px_iou", "Pixel IoU")],
                       "F1 score & IoU", [0, 1.15], fmt=".3f"), c2)
        c1, c2, c3 = st.columns(3)
        plot(bar_chart(rows, [("lat_mean", "mean"), ("lat_median", "median"), ("lat_p95", "p95")],
                       "Latency (ms per image, CPU)", fmt=".0f"), c1)
        plot(bar_chart(rows, [("ckpt_mb", "memory bank / checkpoint"), ("backbone_mb", "backbone weights")],
                       "Model size on disk (MB)", fmt=".0f"), c2)
        plot(bar_chart([r for r in rows if r["peak_rss_mb"]], [("peak_rss_mb", "Peak RAM")],
                       "Peak RAM while evaluating (MB)", fmt=".0f"), c3)
        with st.expander("Full comparison table"):
            st.dataframe(pd.DataFrame(rows).drop(columns=["key", "colour"]), hide_index=True, use_container_width=True)
        if len(rows) > 1:
            fastest = min(rows, key=lambda r: r["lat_mean"])
            slowest = max(rows, key=lambda r: r["lat_mean"])
            best_px = max(rows, key=lambda r: r["pixel_auroc"])
            img_aurocs = {round(r["image_auroc"], 4) for r in rows}
            best_iou = max(rows, key=lambda r: r["px_iou"])
            best_img = max(rows, key=lambda r: r["image_auroc"])
            detection = (f"All evaluated models reach the same image AUROC ({rows[0]['image_auroc']:.3f}), so they "
                         "cannot be separated on detection here. " if len(img_aurocs) == 1 else
                         f"<b>{best_img['label']}</b> detects best (image AUROC {best_img['image_auroc']:.4f}). ")
            iou_note = ("" if best_iou is best_px else
                        f" <b>{best_iou['label']}</b> gives the tightest masks (pixel IoU {best_iou['px_iou']:.3f}).")
            missing = [s["spec"].label for s in STATUSES.values() if not s["checkpoint_available"]]
            ui.html_block(f'<div class="card" style="margin-top:6px"><b>Reading the comparison.</b> {detection}'
                          f'<b>{best_px["label"]}</b> has the highest pixel AUROC ({best_px["pixel_auroc"]:.4f}).'
                          f'{iou_note} <b>{fastest["label"]}</b> is fastest ({fastest["lat_mean"]:.0f} ms vs. '
                          f'{slowest["lat_mean"]:.0f} ms per image on CPU). Only one category with '
                          f'{INFO["test_images"]} test images was evaluated, so small differences are not significant.'
                          + (f' Not ranked (no checkpoint): {", ".join(missing)}.' if missing else '')
                          + ('' if not any(reg.SPECS[r["key"]].family == "EfficientAD" for r in rows) else
                             ' EfficientAD was trained with a reduced protocol, so it is weaker than in the paper.')
                          + '</div>')
    else:
        ui.html_block('<div class="unavailable">No model has been evaluated yet.</div>')

    st.markdown("### Side-by-side inference")
    image, sample = current_image()
    if image is None or st.session_state["current"]["category"] != CATEGORY:
        set_current_sample(CATEGORY, next(iter(test_index(ROOT, CATEGORY))))
        image, sample = current_image()
    current = st.session_state["current"]
    a, b, c = st.columns([1.5, 1.8, 1])
    a.markdown(f'<div class="mono" style="padding-top:30px">Image <b style="color:#e2e8f0">{current["label"]}</b>'
               f' · change it in Live Inspection or Dataset Explorer</div>', unsafe_allow_html=True)
    with b:
        view = ui.segmented("Show", ["Heatmap", "Predicted mask", "Overlay"], "cmp_view")
    c.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    run_all = c.button("▶  Run all available models", type="primary", use_container_width=True, key="run_all")
    gt = ds.load_mask(sample, image.shape[:2]) if sample else None
    keys = list(reg.SPECS)
    panels = st.columns(len(keys) + 2)
    panels[0].image(ui.display_size(image, 420), caption="Original", use_container_width=True)
    for panel, key in zip(panels[1:-1], keys):
        status = STATUSES[key]
        with panel:
            if not status["checkpoint_available"]:
                ui.html_block(f'<div class="disabled-panel" style="padding:44px 10px"><b>{reg.SPECS[key].label}</b>'
                              '<br>unavailable<br><span style="font-size:.75rem">trained checkpoint missing - no output '
                              'is simulated</span></div>')
                continue
            result = st.session_state["preds"].get((current["image_id"], key))
            if result is None and run_all:
                with st.spinner(reg.SPECS[key].label):
                    result = predict(key, CATEGORY, current["image_id"], image)
            if result is None:
                ui.html_block(f'<div class="disabled-panel" style="padding:44px 10px;border-color:rgba(34,211,238,.4)">'
                              f'<b>{reg.SPECS[key].label}</b><br>ready · press <i>Run all</i></div>')
                continue
            pthr = result["pixel_threshold"]
            shown = {"Heatmap": lambda: viz.heatmap(result["anomaly_map"], pthr),
                     "Predicted mask": lambda: viz.coloured_mask(result["mask"], (239, 68, 68)),
                     "Overlay": lambda: viz.overlay(image, result["anomaly_map"], result["mask"], pthr)}[view]()
            verdict = "DEFECTIVE" if result["is_defective"] else "NORMAL"
            st.image(ui.display_size(shown, 420), use_container_width=True,
                     caption=f"{reg.SPECS[key].label} · {verdict} · s={result['score']:.2f} · {result['latency_ms']:.0f} ms")
            if gt is not None and gt.any():
                agreement = viz.mask_agreement(result["mask"], gt)
                st.caption(f"IoU {agreement['iou']:.3f} · Dice {agreement['dice']:.3f}")
    with panels[-1]:
        if gt is not None:
            st.image(ui.display_size(viz.coloured_mask(gt, (34, 197, 94)), 420), use_container_width=True,
                     caption="Ground truth (annotation)")
        else:
            ui.html_block('<div class="disabled-panel" style="padding:44px 10px">Ground truth<br>not available for uploads</div>')
    st.caption("Each heatmap uses its own model's fixed scale (that model's pixel threshold is the middle colour), so the "
               "colours mean the same thing across models even though raw scores differ.")
    st.markdown("### Strengths and limitations")
    cols = st.columns(4)
    for col, spec in zip(cols, reg.SPECS.values()):
        with col.expander(spec.label, expanded=False):
            st.markdown(f"{spec.description}\n\n**Strengths.** {spec.strengths}\n\n**Limitations.** {spec.limitations}")


# ================================================================= Analytics
def confusion_fig(cm: dict, title: str) -> go.Figure:
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    fig = go.Figure(go.Heatmap(z=z, x=["pred normal", "pred defective"], y=["actual normal", "actual defective"],
                               colorscale=[[0, "#0f172a"], [1, "#0891b2"]], showscale=False,
                               text=[[f"{v:,}" for v in row] for row in z], texttemplate="%{text}",
                               textfont=dict(size=15), hovertemplate="%{y} / %{x}: %{z:,}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return ui.style_fig(fig, 300, title, legend=False)


def page_analytics() -> None:
    ui.page_header("Analytics", "Aggregate performance",
                   "Detection (image level) and localisation (pixel level) results of the saved evaluation runs, with "
                   "per-defect breakdowns and latency distributions.")
    rows = comparison_rows()
    if not rows:
        ui.html_block('<div class="unavailable">No evaluation available. Run one from the Model Laboratory.</div>')
        return
    keys = [r["key"] for r in rows]
    key = ui.segmented("Model", keys, "an_model", default=MODEL_KEY if MODEL_KEY in keys else keys[0],
                       format_func=lambda k: reg.SPECS[k].label)
    m = STATUSES[key]["metrics"]
    i, p, lat = m["image_level"], m["pixel_level"], m["latency_ms"]
    per_image = pd.read_csv(results_dir(CATEGORY, key) / "per_image.csv")
    cols = st.columns(6)
    for col, (label, value, colour) in zip(cols, [
            ("Image AUROC", ui.fmt(i["auroc"]), ui.CYAN), ("Image F1", ui.fmt(i["f1"]), ui.CYAN),
            ("Pixel AUROC", ui.fmt(p["auroc"]), ui.BLUE), ("Pixel F1", ui.fmt(p["f1"]), ui.BLUE),
            ("Pixel IoU", ui.fmt(p["iou"]), ui.BLUE), ("Median latency", f"{lat['median']:.0f} ms", ui.AMBER)]):
        col.markdown(ui.kpi(label, value, "", colour), unsafe_allow_html=True)
    st.caption(f"{m['num_test_images']} {DS['split']} images · thresholds from {m['thresholds']['source']} · "
               f"pixel metrics on a {m['pixel_eval_grid']} grid · evaluated {m['evaluated_at']}")

    st.markdown("### Image level · detection")
    c1, c2, c3 = st.columns([1, 1, 1.3])
    roc = go.Figure()
    for r in rows:
        mm = STATUSES[r["key"]]["metrics"]
        roc.add_trace(go.Scatter(x=mm["roc_curve"]["fpr"], y=mm["roc_curve"]["tpr"], mode="lines",
                                 name=f"{r['label']} ({mm['image_level']['auroc']:.3f})",
                                 line=dict(color=r["colour"], width=3 if r["key"] == key else 1.5)))
    roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance", line=dict(color="#475569", dash="dash")))
    roc.update_xaxes(title="false positive rate")
    roc.update_yaxes(title="true positive rate")
    plot(ui.style_fig(roc, 300, "ROC curve (all evaluated models)"), c1)
    plot(confusion_fig(i["confusion_matrix"], "Confusion matrix · images"), c2)
    hist = go.Figure()
    for label, colour, name in ((0, ui.GREEN, "normal"), (1, ui.RED, "defective")):
        part = per_image[per_image["label"] == label]
        hist.add_trace(go.Histogram(x=part["score"], name=f"{name} (n={len(part)})", marker_color=colour, opacity=0.75,
                                    nbinsx=30, hovertemplate="score %{x}<br>%{y} images<extra></extra>"))
    hist.add_vline(x=m["thresholds"]["image"], line=dict(color=ui.AMBER, dash="dash"),
                   annotation_text="threshold", annotation_font_color=ui.AMBER)
    hist.update_layout(barmode="overlay")
    hist.update_xaxes(title="image anomaly score")
    plot(ui.style_fig(hist, 300, "Score distribution"), c3)

    st.markdown("### Pixel level · localisation")
    c1, c2, c3 = st.columns([1, 1.3, 1])
    plot(confusion_fig(p["confusion_matrix"], "Confusion matrix · pixels"), c1)
    defects = per_image[per_image["label"] == 1]
    summary = defects.groupby("defect_type").agg(images=("image", "count"), detected=("predicted", "sum"),
                                                 mean_iou=("iou", "mean"), mean_score=("score", "mean")).reset_index()
    fig = go.Figure(go.Bar(x=summary["defect_type"], y=summary["mean_iou"], marker_color=ui.BLUE,
                           text=[f"{v:.3f}" for v in summary["mean_iou"]], textposition="outside", cliponaxis=False,
                           customdata=summary[["images", "detected"]],
                           hovertemplate="%{x}<br>mean IoU %{y:.3f}<br>detected %{customdata[1]} / "
                                         "%{customdata[0]}<extra></extra>"))
    fig.update_yaxes(range=[0, 1])
    plot(ui.style_fig(fig, 300, "Mean IoU per defect type", legend=False), c2)
    box = go.Figure(go.Box(y=defects["iou"], name="IoU", marker_color=ui.CYAN, boxpoints="all", jitter=0.4,
                           customdata=defects["image"], hovertemplate="%{customdata}<br>IoU %{y:.3f}<extra></extra>"))
    plot(ui.style_fig(box, 300, "IoU per defective image", legend=False), c3)

    st.markdown("### Latency")
    c1, c2 = st.columns([1.4, 1])
    lfig = go.Figure(go.Histogram(x=per_image["latency_ms"], marker_color=ui.AMBER, nbinsx=30,
                                  hovertemplate="%{x} ms<br>%{y} images<extra></extra>"))
    lfig.update_xaxes(title="ms per image (batch size 1, CPU)")
    plot(ui.style_fig(lfig, 260, "Latency distribution", legend=False), c1)
    c2.markdown(ui.kpi("Mean / median / p95", f"{lat['mean']:.0f} / {lat['median']:.0f} / {lat['p95']:.0f} ms",
                       f"{lat['fps']:.1f} images per second · {m['torch_threads']} CPU threads", ui.AMBER),
                unsafe_allow_html=True)
    st.markdown("### Per defect type")
    st.dataframe(summary, hide_index=True, use_container_width=True,
                 column_config={"mean_iou": st.column_config.ProgressColumn("mean IoU", min_value=0, max_value=1,
                                                                            format="%.3f"),
                                "mean_score": st.column_config.NumberColumn(format="%.3f")})
    with st.expander("Per-image results"):
        st.dataframe(per_image, hide_index=True, use_container_width=True)


def roboflow_versions(category: str) -> list[dict]:
    """PatchCore ResNet18 on Roboflow v1 (augmented) vs. v2 (no augmentation), from the saved runs."""
    rows = []
    for tag, label in ds.DATASET_CHOICES:
        metrics_file = PROJECT / "results" / tag / category / "patchcore_r18" / "metrics.json"
        model_file = PROJECT / "artifacts" / tag / category / "model.json"
        if not (metrics_file.is_file() and model_file.is_file()):
            continue
        m = json.loads(metrics_file.read_text())
        info = json.loads(model_file.read_text())["info"]
        rows.append({"version": label, "train images": info["train_images"],
                     "augmentation": "flip, flop, 90° rotate, crop 0-10 %, brightness ±10 %, exposure ±10 %, "
                                     "blur 1 px (x3)" if tag == "roboflow_v1" else "none",
                     "image AUROC": round(m["image_level"]["auroc"], 4),
                     "pixel AUROC": round(m["pixel_level"]["auroc"], 4),
                     "pixel IoU": round(m["pixel_level"]["iou"], 3),
                     "build time (s)": round(info["fit_seconds"]),
                     "active": tag == ds.dataset_tag()})
    return rows


# ========================================================== Dataset Explorer
def inspect_sample(name: str) -> None:
    set_current_sample(st.session_state["category"], name)
    st.session_state["insp_source"] = "MVTec test image"
    go_to("Live Inspection")


def page_dataset() -> None:
    ui.page_header("Dataset Explorer", f"{DS['name']} · {CATEGORY}",
                   f"{DS['source']}. Folder <span class='mono'>{ROOT}</span>. Only train/good builds the models; test images "
                   "and ground-truth masks are used for evaluation. Press Inspect on a test image to open it in the workspace.")
    cols = st.columns(4)
    for col, (label, value, sub, colour) in zip(cols, [
            ("Training images", INFO["train_images"],
             "train/good · defect-free" + (f" · {INFO['valid_images']} valid" if INFO["valid_images"] else ""),
             ui.GREEN),
            ("Test · normal", INFO["test_normal"], "test/good", ui.GREEN),
            ("Test · defective", INFO["test_defective"], ", ".join(INFO["defect_types"]), ui.RED),
            ("Ground-truth masks", INFO["masks"], "✓ all present and aligned" if INFO["valid"] else "issues found", ui.CYAN)]):
        col.markdown(ui.kpi(label, str(value), sub, colour), unsafe_allow_html=True)
    versions = roboflow_versions(CATEGORY)
    if len(versions) == 2:
        st.markdown("### Roboflow versions · augmentation study (PatchCore · ResNet18, same 20 test images)")
        st.dataframe(pd.DataFrame(versions), hide_index=True, use_container_width=True)
        v1, v2 = versions[1], versions[0]
        st.caption(f"Both versions: 80/10/10 split, Auto-Orient + Stretch to 224x224. Augmentation tripled the "
                   f"training set ({v2['train images']} → {v1['train images']}) but did not improve the model "
                   f"(pixel AUROC {v1['pixel AUROC']} vs {v2['pixel AUROC']}) and took "
                   f"{v1['build time (s)'] / max(v2['build time (s)'], 1):.1f}× longer, so v2 is the final dataset. "
                   "Switch versions with **Dataset** in the sidebar.")
    tabs = st.tabs(["Normal · train", "Normal · test"] + [f"Defect · {d}" for d in INFO["defect_types"]])
    samples = list(test_index(ROOT, CATEGORY).values())
    with tabs[0]:
        grid = st.columns(6)
        for index, sample in enumerate(ds.train_samples(Path(ROOT), CATEGORY)[:12]):
            grid[index % 6].image(ui.display_size(ds.load_rgb(sample.path), 260), caption=sample.path.name,
                                  use_container_width=True)
    for tab, defect in zip(tabs[1:], ["good"] + INFO["defect_types"]):
        with tab:
            if defect != "good":
                ui.legend([(ui.GREEN, "ground-truth defect outline (annotation)")])
            grid = st.columns(6)
            for index, sample in enumerate([s for s in samples if s.defect_type == defect][:12]):
                image = ds.load_rgb(sample.path)
                if sample.label:
                    gt = ds.load_mask(sample, image.shape[:2])
                    image = viz.overlay(image, np.zeros(gt.shape, np.float32), np.zeros_like(gt), 1.0, gt, alpha=0.0)
                with grid[index % 6]:
                    st.image(ui.display_size(image, 260), caption=sample.name, use_container_width=True)
                    st.button("Inspect", key=f"insp_{sample.name}", use_container_width=True,
                              on_click=inspect_sample, args=(sample.name,))


# ======================================================== Inspection History
def page_history() -> None:
    location = history.DB_PATH.relative_to(PROJECT) if history.DB_PATH.is_relative_to(PROJECT) else history.DB_PATH
    ui.page_header("Inspection History", "Inspection log", f"Stored locally in SQLite ({location}).")
    frame = history.load()
    if frame.empty:
        ui.html_block('<div class="disabled-panel">No inspections yet - run one in <b>Live Inspection</b>.</div>')
        return
    frame["model"] = frame["model"].fillna("patchcore_r18")
    known = frame[frame["ground_truth"] != "unknown"]
    cols = st.columns(4)
    cols[0].markdown(ui.kpi("Inspections", str(len(frame)), "", ui.CYAN), unsafe_allow_html=True)
    cols[1].markdown(ui.kpi("Flagged defective", str(int((frame["prediction"] == "defective").sum())), "", ui.RED),
                     unsafe_allow_html=True)
    cols[2].markdown(ui.kpi("Agreement with ground truth",
                            f"{(known['prediction'] == known['ground_truth']).mean():.0%}" if len(known) else "—",
                            f"{len(known)} with known ground truth", ui.GREEN), unsafe_allow_html=True)
    cols[3].markdown(ui.kpi("Median latency", f"{frame['latency_ms'].median():.0f} ms", "", ui.AMBER),
                     unsafe_allow_html=True)
    f1, f2 = st.columns(2)
    models = f1.multiselect("Model", sorted(frame["model"].unique()), format_func=lambda k: reg.SPECS[k].label)
    preds = f2.multiselect("Prediction", ["normal", "defective"])
    view = frame
    if models:
        view = view[view["model"].isin(models)]
    if preds:
        view = view[view["prediction"].isin(preds)]
    st.dataframe(view.drop(columns=["overlay_path"]), hide_index=True, use_container_width=True,
                 column_config={"score": st.column_config.NumberColumn(format="%.3f"),
                                "image_threshold": st.column_config.NumberColumn(format="%.3f"),
                                "defect_area_pct": st.column_config.NumberColumn("defect area %", format="%.2f"),
                                "latency_ms": st.column_config.NumberColumn(format="%.0f"),
                                "iou": st.column_config.NumberColumn(format="%.3f"),
                                "dice": st.column_config.NumberColumn(format="%.3f")})
    st.download_button("Download CSV", view.to_csv(index=False), "inspection_history.csv", "text/csv")
    thumbs = view[view["overlay_path"].notna()].head(6)
    if len(thumbs):
        st.markdown("### Latest overlays")
        grid = st.columns(6)
        for col, (_, row) in zip(grid, thumbs.iterrows()):
            if Path(row["overlay_path"]).is_file():
                col.image(row["overlay_path"], caption=f"{row['image']} · {row['prediction']}", use_container_width=True)


# =================================================================== Reports
def page_reports() -> None:
    ui.page_header("Reports", "PDF report",
                   "Project, model and dataset description, the active model's metrics, the model comparison table, the "
                   "current inspection (with ground-truth comparison when available) and limitations.")
    status = STATUSES[MODEL_KEY]
    if not status["checkpoint_available"]:
        ui.html_block(f'<div class="unavailable">{status["reason"]}</div>')
        return
    image, sample = current_image()
    current = st.session_state["current"]
    has_current = image is not None and (current["image_id"], MODEL_KEY) in st.session_state["preds"]
    c1, c2 = st.columns(2)
    c1.markdown(ui.kpi("Active model", reg.SPECS[MODEL_KEY].label,
                       "evaluated" if status["evaluated"] else "not evaluated", ui.CYAN), unsafe_allow_html=True)
    c2.markdown(ui.kpi("Current inspection", current["label"] if has_current else "none",
                       "will be included" if has_current else "run one in Live Inspection to include it",
                       ui.GREEN if has_current else ui.AMBER), unsafe_allow_html=True)
    st.write("")
    if st.button("Generate PDF report", type="primary", key="gen_report"):
        if get_model(MODEL_KEY, CATEGORY) is None:  # shows the low-RAM warning
            st.error("Report not generated: the active model could not be loaded. Free some RAM, or choose "
                     "PatchCore · ResNet18, and try again.")
            return
        with st.spinner("Generating report ..."):
            path = make_report(image, sample, current if has_current else None)
        st.toast("Report generated", icon="✅")
        st.success(f"Saved {path.relative_to(PROJECT) if path.is_relative_to(PROJECT) else path}")
        st.download_button("Download PDF", path.read_bytes(), path.name, "application/pdf")


def make_report(image, sample, current) -> Path:
    model = get_model(MODEL_KEY, CATEGORY)
    metrics = load_metrics(CATEGORY, MODEL_KEY)
    figures = []
    if metrics:
        figures = [viz.figure_to_png(viz.confusion_figure(metrics["image_level"]["confusion_matrix"],
                                                          "Image-level confusion matrix")),
                   viz.figure_to_png(viz.roc_figure(metrics["roc_curve"]["fpr"], metrics["roc_curve"]["tpr"],
                                                    metrics["image_level"]["auroc"]))]
    samples = []
    if current:
        r = st.session_state["preds"][(current["image_id"], MODEL_KEY)]
        gt = ds.load_mask(sample, image.shape[:2]) if sample else None
        agreement = viz.mask_agreement(r["mask"], gt) if gt is not None and gt.any() else None
        truth = f" (ground truth: {'defective' if sample.label else 'normal'})" if sample else " (no ground truth)"
        samples.append({
            "name": f"Current inspection: {current['label']}{truth}"
                    + (f" - IoU {agreement['iou']:.3f}, Dice {agreement['dice']:.3f}" if agreement else ""),
            "score": r["score"], "latency_ms": r["latency_ms"],
            "prediction": "defective" if r["is_defective"] else "normal", "image": image,
            "heatmap": viz.heatmap(r["anomaly_map"], r["pixel_threshold"]),
            "overlay": viz.gt_comparison(image, r["mask"], gt) if gt is not None
            else viz.overlay(image, r["anomaly_map"], r["mask"], r["pixel_threshold"])})
    tests = list(test_index(ROOT, CATEGORY).values())
    picks = [next(s for s in tests if s.label == 0)] + [next(s for s in tests if s.defect_type == d)
                                                        for d in INFO["defect_types"][:2]]
    for s in picks:
        img = ds.load_rgb(s.path)
        r = model.predict(img)
        gt = ds.load_mask(s, img.shape[:2])
        samples.append({"name": f"{CATEGORY}/{s.name} (ground truth: {'defective' if s.label else 'normal'})",
                        "score": r["score"], "latency_ms": r["latency_ms"],
                        "prediction": "defective" if r["is_defective"] else "normal", "image": img,
                        "heatmap": viz.heatmap(r["anomaly_map"], r["pixel_threshold"]),
                        "overlay": viz.overlay(img, r["anomaly_map"], r["mask"], r["pixel_threshold"], gt)})
    model_info = {**model.info, "image_size": model.info.get("input_size", 224), "label": reg.SPECS[MODEL_KEY].label,
                  "description": reg.SPECS[MODEL_KEY].description}
    path = history.OUTPUT_DIR / "reports" / f"report_{ds.dataset_tag() or 'mvtec_ad'}_{CATEGORY}_{MODEL_KEY}_{time.strftime('%Y%m%d_%H%M%S')}.pdf"
    unavailable = [f"{s['spec'].label}: {s['reason']}" for s in STATUSES.values() if not s["checkpoint_available"]]
    dataset_info = {**INFO, "dataset_name": DS["name"], "dataset_source": DS["source"], "split": DS["split"]}
    return build_pdf(path, dataset_info, metrics, model_info, samples, figures, comparison=comparison_rows(),
                     unavailable=unavailable)


# ================================================================== Settings
def page_settings() -> None:
    ui.page_header("Settings", "Runtime settings", "Session-level preferences and system information.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Inference")
        st.slider("CPU threads for inference", 1, 8, key="threads",
                  help="Lower values leave more CPU for other applications.")
        st.slider("Heatmap overlay opacity", 0.2, 0.8, step=0.05, key="heat_alpha")
        st.markdown("### Maintenance")
        if st.button("Clear cached predictions"):
            st.session_state["preds"] = {}
            st.toast("Prediction cache cleared", icon="✅")
        if st.button("Unload models from memory"):
            cached_model.clear()
            st.session_state["preds"] = {}
            st.toast("Models unloaded", icon="✅")
        confirm = st.checkbox("I want to delete the inspection history")
        if st.button("Clear inspection history", disabled=not confirm):
            history.clear()
            st.toast("History cleared", icon="✅")
    with c2:
        st.markdown("### System")
        st.markdown(f"""<div class="card mono" style="line-height:1.9">
            dataset root · {ROOT}<br>categories · {len(CATEGORIES)} available<br>
            device · CPU (CUDA {'available' if torch.cuda.is_available() else 'not available'})<br>
            torch · {torch.__version__}<br>RAM available · {mem_available_mb():,} MB<br>
            main platform · {reg.MAIN_PROJECT} ({'found' if reg.MAIN_PROJECT.is_dir() else 'missing'})</div>""",
                    unsafe_allow_html=True)
        st.markdown("### Safety")
        st.caption("EfficientAD training is disabled in this application. Builds and evaluations refuse to start when "
                   "available RAM is low, to protect other applications.")


# ============================================================= Documentation
def render_markdown_doc(path: Path) -> None:
    """Render a project markdown file: text, local images and Mermaid diagrams."""
    import re

    text = path.read_text(encoding="utf-8")
    parts = re.split(r"```mermaid\n(.*?)```", text, flags=re.S)
    for index, part in enumerate(parts):
        if index % 2 == 1:  # Mermaid source
            ui.mermaid(part)
            continue
        for chunk in re.split(r"(!\[[^\]]*\]\([^)]+\))", part):
            image = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", chunk)
            if image:
                target = (path.parent / image.group(2)).resolve()
                if target.is_file():
                    st.image(str(target), caption=image.group(1), use_container_width=True)
            elif chunk.strip():
                st.markdown(chunk)


def page_docs() -> None:
    ui.page_header("Documentation", "Project documents",
                   "The page guide, README, demo script and the project explanation for the report and viva. "
                   f"The same files are in <span class='mono'>{PROJECT}</span>.")
    tabs = st.tabs([name for name, _ in DOCS])
    for tab, (name, relative) in zip(tabs, DOCS):
        with tab:
            path = PROJECT / relative
            if not path.is_file():
                ui.html_block(f'<div class="unavailable">{relative} not found.</div>')
                continue
            st.caption(f"{path}")
            st.download_button(f"Download {path.name}", path.read_bytes(), path.name, "text/markdown",
                               key=f"dl_{name}")
            render_markdown_doc(path)


{"Overview": page_overview, "Live Inspection": page_inspection, "Model Laboratory": page_laboratory,
 "Model Comparison": page_comparison, "Analytics": page_analytics, "Dataset Explorer": page_dataset,
 "Inspection History": page_history, "Reports": page_reports, "Documentation": page_docs,
 "Settings": page_settings}[st.session_state["nav"]]()
