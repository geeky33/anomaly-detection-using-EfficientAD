"""All models the app can show, their real availability, and a common predict API.

Models
------
* ``patchcore_r18``   - this project's PatchCore-style model (ResNet18), built here.
* ``patchcore_wrn50`` - PatchCore with WideResNet-50 from the main platform
  (``~/fun/industrial-anomaly-detection``), loaded read-only from its checkpoint.
* ``efficientad_s`` / ``efficientad_m`` - EfficientAD checkpoints of the main
  platform.  They are only used if a checkpoint file actually exists; nothing is
  ever trained or simulated here.

Every available model exposes ``predict(image) -> dict`` with the same keys as
:meth:`model.PatchCoreLite.predict` and a ``calibration`` with the image and
pixel thresholds, so evaluation and visualisation treat them identically.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import dataset as ds
from model import Calibration, PatchCoreLite

PROJECT = Path(__file__).resolve().parent
MAIN_PROJECT = Path.home() / "fun" / "industrial-anomaly-detection"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    family: str
    source: str  # "mvp" or "main"
    architecture: str  # main-platform architecture name
    description: str
    strengths: str
    limitations: str


SPECS: dict[str, ModelSpec] = {
    "patchcore_r18": ModelSpec(
        "patchcore_r18", "PatchCore · ResNet18", "PatchCore", "mvp", "patchcore",
        "Frozen ImageNet ResNet18, layer2+layer3 patch features (384-d), greedy coreset memory bank "
        "(10 %), nearest-neighbour distance. Built in this project.",
        "No training; ~4 min build on CPU; ~50 ms per image; strong image-level detection.",
        "Simplified PatchCore (lighter backbone, no 1024-d projection, no Eq. 7 re-weighting); coarse "
        "28x28 patch grid gives oversized masks.",
    ),
    "patchcore_wrn50": ModelSpec(
        "patchcore_wrn50", "PatchCore · WRN-50", "PatchCore", "main", "patchcore",
        "Frozen ImageNet WideResNet-50-2, layer2+layer3, 1024-d embeddings, exact greedy coreset (10 %). "
        "Built with the main platform code (on a Colab T4 for the Roboflow dataset).",
        "The backbone of the original PatchCore paper; finer 32x32 patch grid at 256 px input.",
        "~6x more backbone parameters than ResNet18: much slower on CPU; larger memory bank.",
    ),
    "efficientad_s": ModelSpec(
        "efficientad_s", "EfficientAD-S", "EfficientAD", "main", "efficientad_s",
        "Student-teacher PDN (distilled teacher) + autoencoder (Batzner et al., WACV 2024).",
        "Designed for millisecond latency on GPU; models logical anomalies via the autoencoder.",
        "Paper protocol needs 70,000 iterations (~32 h on this CPU). The Roboflow checkpoint uses a reduced "
        "protocol: 2,500 iterations on a Colab T4 GPU (6.6 min), so it is weaker than the paper model.",
    ),
    "efficientad_m": ModelSpec(
        "efficientad_m", "EfficientAD-M", "EfficientAD", "main", "efficientad_m",
        "Larger PDN variant of EfficientAD (paper Table 2).",
        "Higher accuracy than -S in the paper.",
        "Paper protocol needs ~225 h on this CPU. The Roboflow checkpoint uses a reduced protocol: 1,500 "
        "iterations on a Colab T4 GPU (10.8 min), so it is weaker than the paper model.",
    ),
}

UNAVAILABLE_REASON = {
    "efficientad_s": "No checkpoint for official MVTec AD: on this CPU the bottle run reached only ~925 of 3,000 "
                     "iterations in 48 min (~3 s/iteration). Trained on a GPU for the Roboflow dataset instead.",
    "efficientad_m": "No checkpoint for official MVTec AD (≈225 h on this CPU for the paper protocol). "
                     "Trained on a GPU for the Roboflow dataset instead.",
}


def checkpoint_path(key: str, category: str) -> Path:
    spec = SPECS[key]
    if spec.source == "mvp":
        return PROJECT / "artifacts" / ds.output_key(category) / "memory_bank.pt"
    return main_models_dir() / spec.architecture / category / "checkpoint.ckpt"


def main_models_dir() -> Path:
    """Where main-platform checkpoints for the active dataset live.

    Official MVTec AD: the main platform's own ``models`` folder.  Any other
    dataset (e.g. ``roboflow_v2``): ``artifacts/<tag>/main_models``, filled by
    ``scripts/import_colab_models.py``, so a checkpoint trained on MVTec AD is
    never presented as trained on that dataset.
    """
    tag = ds.dataset_tag()
    return PROJECT / "artifacts" / tag / "main_models" if tag else MAIN_PROJECT / "models"


def metrics_path(key: str, category: str) -> Path:
    return PROJECT / "results" / ds.output_key(category) / key / "metrics.json"


def file_size_mb(path: Path) -> float | None:
    return round(path.stat().st_size / 2**20, 1) if path.is_file() else None


def status(key: str, category: str) -> dict:
    """Real availability: checkpoint on disk? evaluated with this project's protocol?"""
    ckpt = checkpoint_path(key, category)
    metrics_file = metrics_path(key, category)
    metrics = json.loads(metrics_file.read_text()) if metrics_file.is_file() else None
    has_ckpt = ckpt.is_file()
    reason = None
    v2_ckpt = PROJECT / "artifacts" / "roboflow_v2" / "main_models" / SPECS[key].architecture / category / "checkpoint.ckpt"
    if not has_ckpt and ds.dataset_tag() == "roboflow_v1" and SPECS[key].source == "main" and v2_ckpt.is_file():
        reason = ("Trained for Roboflow v2 only (the final dataset, no augmentation). Select "
                  "'Roboflow v2' as the dataset in the sidebar to use this model. On v1 only PatchCore · "
                  "ResNet18 was built, for the augmentation comparison.")
    elif not has_ckpt and ds.dataset_tag() and SPECS[key].source == "main":
        reason = (f"Not trained on the {ds.dataset_tag()} dataset yet. Train it on a GPU with "
                  "colab/train_efficientad_colab.ipynb, then run scripts/import_colab_models.py.")
    elif not has_ckpt:
        reason = UNAVAILABLE_REASON.get(key, "No checkpoint has been built for this category yet.")
    return {
        "key": key,
        "spec": SPECS[key],
        "checkpoint": str(ckpt),
        "checkpoint_available": has_ckpt,
        "checkpoint_mb": file_size_mb(ckpt),
        "evaluated": metrics is not None,
        "metrics": metrics,
        "reason": reason,
    }


def all_status(category: str) -> list[dict]:
    return [status(key, category) for key in SPECS]


# --------------------------------------------------------------------- loaders
def _main_backend():
    """Import the main platform's backend package (named ``app``) read-only.

    This project has an ``app.py``; make sure ``import app`` resolves to the
    backend *package*, not to the Streamlit script.
    """
    backend = MAIN_PROJECT / "backend"
    if not backend.is_dir():
        raise FileNotFoundError(f"Main platform not found at {MAIN_PROJECT}")
    loaded = sys.modules.get("app")
    if loaded is not None and not hasattr(loaded, "__path__"):
        del sys.modules["app"]
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    # The backend reads its settings (models folder) once, at first import.
    os.environ["MODELS_DIR"] = str(main_models_dir())
    registry = importlib.import_module("app.services.model_registry")
    # Already imported for another dataset in this process: point it at the active one.
    registry.settings.models_dir = main_models_dir()
    inference = importlib.import_module("app.ml.inference")
    return registry, inference


class MainPlatformModel:
    """Wraps a main-platform checkpoint behind the PatchCoreLite predict API."""

    def __init__(self, key: str, category: str, device: str = "cpu") -> None:
        registry, inference = _main_backend()
        self._infer = inference.infer_image
        self.key = key
        self.architecture = SPECS[key].architecture
        self.device = device
        self.model, self.metadata = registry.load_model(self.architecture, category, device)
        self.input_size = inference.model_input_size(self.architecture, self.metadata)
        cal = self.metadata["calibration"]
        self.calibration = Calibration(
            image_threshold=float(cal["image_threshold"]),
            pixel_threshold=float(cal["pixel_threshold"]),
            num_validation_images=int(cal.get("num_validation_images", 0)),
        )
        summary = self.metadata.get("summary", {})
        self.info = {
            "category": category,
            "train_images": len(ds.train_samples(ds.find_dataset_root(), category))
            - self.calibration.num_validation_images,
            "memory_bank_size": summary.get("coreset_size"),
            "patch_vectors": summary.get("total_embeddings"),
            "feature_dim": summary.get("embedding_dim"),
            "fit_seconds": round(summary.get("fit_seconds", summary.get("duration_seconds", 0)), 1) or None,
            "protocol": self.metadata.get("protocol"),
            "iterations": self.metadata.get("training_config", {}).get("total_iters"),
            "input_size": self.input_size,
            "calibration_source": "main platform: 10 % of train/good (seed 42); image = max validation "
                                  "score, pixel = 99.5th percentile",
        }

    def predict(self, image: np.ndarray) -> dict:
        result = self._infer(self.model, self.architecture, image, self.input_size, self.device)
        amap = result.anomaly_map
        cal = self.calibration
        return {
            "score": result.anomaly_score,
            "anomaly_map": amap,
            "latency_ms": result.model_ms,
            "mask": (amap >= cal.pixel_threshold).astype(np.uint8),
            "is_defective": result.anomaly_score >= cal.image_threshold,
            "image_threshold": cal.image_threshold,
            "pixel_threshold": cal.pixel_threshold,
        }


def load(key: str, category: str, device: str = "cpu"):
    """Load an available model; raises FileNotFoundError when there is no checkpoint."""
    if not checkpoint_path(key, category).is_file():
        raise FileNotFoundError(UNAVAILABLE_REASON.get(key, f"No checkpoint for {key}/{category}."))
    if SPECS[key].source == "mvp":
        model = PatchCoreLite.load(PROJECT / "artifacts" / ds.output_key(category), device)
        model.info.setdefault("calibration_source",
                              "10 % of train/good (seed 0); image = max validation score, pixel = 99.5th percentile")
        model.info.setdefault("input_size", model.config.image_size)
        return model
    return MainPlatformModel(key, category, device)
