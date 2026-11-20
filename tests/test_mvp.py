"""End-to-end checks of the MVP on the real MVTec AD bottle category.

Run:  python -m pytest -q tests
Requires the built bottle model (python evaluation.py --category bottle).
"""

from __future__ import annotations

import sys
from pathlib import Path

import os
import tempfile

import numpy as np
import pytest
import torch

os.environ["MVP_HISTORY_DB"] = str(Path(tempfile.mkdtemp()) / "history.sqlite3")
# These checks use the official MVTec AD release (the app's default is the Roboflow dataset).
os.environ["MVTEC_ROOT"] = str(Path.home() / "fun" / "industrial-anomaly-detection" / "data" / "mvtec_ad")
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import dataset as ds  # noqa: E402
import visualization as viz  # noqa: E402
from evaluation import artifacts_dir, binary_metrics  # noqa: E402
from model import PatchCoreLite, greedy_coreset  # noqa: E402

torch.set_num_threads(4)
ROOT = ds.find_dataset_root()


@pytest.fixture(scope="module")
def model() -> PatchCoreLite:
    if not (artifacts_dir("bottle") / "memory_bank.pt").is_file():
        pytest.skip("bottle model not built")
    return PatchCoreLite.load(artifacts_dir("bottle"))


def test_dataset_loading_and_integrity() -> None:
    info = ds.validate_category(ROOT, "bottle")
    assert info["valid"], info["problems"]
    assert (info["train_images"], info["test_normal"], info["test_defective"], info["masks"]) == (209, 20, 63, 63)
    train, val = ds.split_train_validation(ds.train_samples(ROOT, "bottle"))
    assert len(train) == 188 and len(val) == 21 and not {s.path for s in train} & {s.path for s in val}
    assert all("/train/good/" in str(s.path) for s in train + val)


def test_pretrained_backbone_and_feature_shape() -> None:
    m = PatchCoreLite()
    assert not any(p.requires_grad for p in m.backbone.parameters())
    # ImageNet weights, not random: first conv weights of torchvision ResNet18 have a known mean.
    assert abs(float(m.backbone.conv1.weight.mean())) < 0.01 and float(m.backbone.conv1.weight.std()) > 0.1
    features = m.embed(torch.zeros(2, 3, 224, 224))
    assert features.shape == (2, 384, 28, 28)


def test_memory_bank_construction_small() -> None:
    m = PatchCoreLite()
    paths = [s.path for s in ds.train_samples(ROOT, "bottle")[:6]]
    info = m.fit(paths, ds.load_rgb)
    assert info["patch_vectors"] == 6 * 28 * 28
    assert info["memory_bank_size"] == int(6 * 28 * 28 * 0.1)
    assert m.memory_bank.shape == (info["memory_bank_size"], 384)


def test_greedy_coreset_is_farthest_point() -> None:
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [10.0, 1.0], [5.0, 0.0]])
    chosen = greedy_coreset(points, 3, projection_dim=128).tolist()
    assert len(set(chosen)) == 3
    # After the first point, the next pick is always the one farthest from the set.
    assert set(chosen) >= {0, 2} or set(chosen) >= {1, 3} or set(chosen) >= {0, 3}


def test_saved_model_is_complete(model: PatchCoreLite) -> None:
    assert model.memory_bank.shape[1] == 384 and model.memory_bank.shape[0] > 1000
    cal = model.calibration
    assert cal.num_validation_images == 21
    assert cal.image_threshold == pytest.approx(max(cal.validation_scores))


@pytest.mark.parametrize("defect_type", ["good", "broken_large"])
def test_inference_normal_and_defective(model: PatchCoreLite, defect_type: str) -> None:
    sample = next(s for s in ds.test_samples(ROOT, "bottle") if s.defect_type == defect_type)
    image = ds.load_rgb(sample.path)
    result = model.predict(image)
    assert result["anomaly_map"].shape == image.shape[:2] == (900, 900)
    assert result["mask"].shape == image.shape[:2]
    assert result["anomaly_map_small"].shape == (224, 224)
    assert result["latency_ms"] > 0
    if defect_type == "good":
        assert not result["is_defective"]
    else:
        assert result["is_defective"]
        gt = ds.load_mask(sample, image.shape[:2])
        # the defect region must be hotter than the rest of the image
        assert result["anomaly_map"][gt > 0].mean() > result["anomaly_map"][gt == 0].mean()


def test_heatmap_and_overlay(model: PatchCoreLite) -> None:
    sample = next(s for s in ds.test_samples(ROOT, "bottle") if s.label == 1)
    image = ds.load_rgb(sample.path)
    result = model.predict(image)
    heat = viz.heatmap(result["anomaly_map"], result["pixel_threshold"])
    over = viz.overlay(image, result["anomaly_map"], result["mask"], result["pixel_threshold"],
                       ds.load_mask(sample, image.shape[:2]))
    assert heat.shape == over.shape == image.shape and heat.dtype == over.dtype == np.uint8
    assert not np.array_equal(over, image)


def test_metric_calculations() -> None:
    m = binary_metrics(np.array([0, 0, 0, 1, 1, 1, 1]), np.array([0, 0, 1, 0, 1, 1, 1]))
    assert m["confusion_matrix"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 3}
    assert m["precision"] == pytest.approx(0.75) and m["recall"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(0.75) and m["iou"] == pytest.approx(3 / 5)
    assert m["specificity"] == pytest.approx(2 / 3)


def test_saved_metrics_are_consistent() -> None:
    import json

    path = PROJECT / "results" / "bottle" / "patchcore_r18" / "metrics.json"
    if not path.is_file():
        pytest.skip("evaluation not run")
    metrics = json.loads(path.read_text())
    cm = metrics["image_level"]["confusion_matrix"]
    assert sum(cm.values()) == metrics["num_test_images"] == 83
    assert cm["tn"] + cm["fp"] == 20 and cm["fn"] + cm["tp"] == 63
    pixel_cm = metrics["pixel_level"]["confusion_matrix"]
    assert sum(pixel_cm.values()) == 83 * 256 * 256


def test_pdf_report(model: PatchCoreLite, tmp_path: Path) -> None:
    from report import build_pdf

    sample = next(s for s in ds.test_samples(ROOT, "bottle") if s.label == 1)
    image = ds.load_rgb(sample.path)
    r = model.predict(image)
    samples = [{"name": sample.name, "score": r["score"], "latency_ms": r["latency_ms"], "prediction": "defective",
                "image": image, "heatmap": viz.heatmap(r["anomaly_map"], r["pixel_threshold"]),
                "overlay": viz.overlay(image, r["anomaly_map"], r["mask"], r["pixel_threshold"])}]
    import json

    metrics_path = PROJECT / "results" / "bottle" / "patchcore_r18" / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else None
    path = build_pdf(tmp_path / "r.pdf", ds.validate_category(ROOT, "bottle"), metrics,
                     {**model.info, "image_size": 224}, samples)
    data = path.read_bytes()
    assert data[:4] == b"%PDF" and len(data) > 50_000


# ------------------------------------------------------------ model registry
def test_efficientad_is_unavailable_and_never_ranked() -> None:
    import model_registry as reg

    for key in ("efficientad_s", "efficientad_m"):
        status = reg.status(key, "bottle")
        assert not status["checkpoint_available"] and not status["evaluated"] and status["metrics"] is None
        assert "checkpoint" in status["reason"].lower()
        with pytest.raises(FileNotFoundError):
            reg.load(key, "bottle")
    evaluated = [k for k in reg.SPECS if reg.status(k, "bottle")["evaluated"]]
    assert "efficientad_s" not in evaluated and "efficientad_m" not in evaluated


def test_all_evaluated_models_used_the_same_test_images() -> None:
    import pandas as pd

    import model_registry as reg

    lists = []
    for key in reg.SPECS:
        path = PROJECT / "results" / "bottle" / key / "per_image.csv"
        if path.is_file():
            lists.append(pd.read_csv(path)["image"].tolist())
    assert lists and all(images == lists[0] for images in lists)


def test_main_platform_adapter_reproduces_saved_scores() -> None:
    import pandas as pd

    import model_registry as reg

    if not reg.status("patchcore_wrn50", "bottle")["evaluated"]:
        pytest.skip("WRN-50 checkpoint not evaluated")
    model = reg.load("patchcore_wrn50", "bottle")
    saved = pd.read_csv(PROJECT / "results" / "bottle" / "patchcore_wrn50" / "per_image.csv").set_index("image")
    sample = next(s for s in ds.test_samples(ROOT, "bottle") if s.name == "broken_small/000.png")
    image = ds.load_rgb(sample.path)
    result = model.predict(image)
    assert result["anomaly_map"].shape == image.shape[:2]
    assert result["score"] == pytest.approx(saved.loc[sample.name, "score"], abs=1e-4)


def test_region_and_gt_helpers() -> None:
    pred = np.zeros((100, 100), np.uint8)
    pred[10:30, 10:30] = 1
    pred[60:65, 60:90] = 1
    gt = np.zeros_like(pred)
    gt[20:40, 20:40] = 1
    regions = viz.regions(pred)
    assert [(r["x0"], r["y0"], r["x1"], r["y1"]) for r in regions] == [(10, 10, 30, 30), (60, 60, 90, 65)]
    agreement = viz.mask_agreement(pred, gt)
    assert agreement["tp_px"] == 100 and agreement["iou"] == pytest.approx(100 / (550 + 400 - 100))
    image = np.full((100, 100, 3), 128, np.uint8)
    cmp = viz.gt_comparison(image, pred, gt)
    assert cmp[25, 25, 0] > cmp[25, 25, 2] and cmp[25, 25, 1] > cmp[25, 25, 2]  # overlap = yellow
    assert cmp[15, 15, 0] > cmp[15, 15, 1]  # predicted only = red
    assert cmp[35, 35, 1] > cmp[35, 35, 0]  # ground truth only = green


# ---------------------------------------------------------------- Streamlit
PAGES = ["Overview", "Live Inspection", "Model Laboratory", "Model Comparison", "Analytics",
         "Dataset Explorer", "Inspection History", "Reports", "Documentation", "Settings"]


def _app():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PROJECT / "app.py"), default_timeout=240)
    app.run()
    assert not app.exception, app.exception
    return app


def test_streamlit_pages_render(model: PatchCoreLite) -> None:
    app = _app()
    for page in PAGES:
        app.radio(key="nav").set_value(page).run()
        assert not app.exception, (page, app.exception)


def test_workspace_modes_and_cross_page_flow(model: PatchCoreLite) -> None:
    import history

    before = len(history.load())
    project_outputs = {p for d in ("history", "reports") for p in (PROJECT / "results" / d).glob("*")}
    app = _app()
    # Dataset Explorer -> Inspect -> Live Inspection with that image selected
    app.radio(key="nav").set_value("Dataset Explorer").run()
    app.button(key="insp_broken_large/000.png").click().run()
    assert app.session_state["nav"] == "Live Inspection"
    assert app.session_state["current"]["name"] == "broken_large/000.png"
    app.button(key="run_insp").click().run()
    assert not app.exception, app.exception
    assert any("DEFECTIVE" in m.value for m in app.markdown)
    assert len(history.load()) == before + 1
    assert any(m.label == "IoU vs GT" for m in app.metric)
    for mode in ["Original", "Anomaly heatmap", "Predicted mask", "Heatmap overlay", "Ground-truth mask",
                 "Ground-truth comparison"]:
        app.session_state["viz_mode"] = mode
        app.run()
        assert not app.exception, (mode, app.exception)
    assert len(history.load()) == before + 1  # re-renders never add history rows

    # Model Comparison on the same image: unavailable models shown as disabled, no outputs simulated
    app.radio(key="nav").set_value("Model Comparison").run()
    app.button(key="run_all").click().run()
    assert not app.exception, app.exception
    text = " ".join(m.value for m in app.markdown)
    assert "broken_large/000.png" in text and "trained checkpoint missing" in text
    assert ("bottle/broken_large/000.png", "patchcore_r18") in app.session_state["preds"]

    # Report containing the current inspection
    app.radio(key="nav").set_value("Reports").run()
    app.button(key="gen_report").click().run()
    assert not app.exception, app.exception
    assert any("Saved " in s.value and s.value.endswith(".pdf") for s in app.success)
    # outputs went next to the temporary test database, not into the project's results folder
    after = {p for d in ("history", "reports") for p in (PROJECT / "results" / d).glob("*")}
    assert after == project_outputs


def test_documentation_files_exist_and_render(model: PatchCoreLite) -> None:
    for relative in ("README.md", "docs/PAGE_GUIDE.md", "docs/DEMO_GUIDE.md", "docs/PROJECT_EXPLANATION.md", "docs/VIVA.md"):
        assert (PROJECT / relative).stat().st_size > 2000, relative
    app = _app()
    app.radio(key="nav").set_value("Documentation").run()
    assert not app.exception, app.exception
    assert [t.label for t in app.tabs] == ["Page guide", "README", "Demo guide", "Project explanation", "Viva Q&A"]
    text = " ".join(m.value for m in app.markdown)
    assert "Live Inspection" in text and "viva" in text.lower()
