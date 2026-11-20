"""Dataset discovery, validation and loading (Roboflow export by default, or official MVTec AD).

Set MVTEC_ROOT to choose a dataset explicitly, e.g. the official release.

Expected layout of one category (official MVTec AD release)::

    <root>/<category>/train/good/*.png            defect-free training images
    <root>/<category>/test/good/*.png             defect-free test images
    <root>/<category>/test/<defect>/*.png         defective test images
    <root>/<category>/ground_truth/<defect>/<name>_mask.png

Only ``train/good`` is used to build the model.  The test split and the
ground-truth masks are used for evaluation only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)

# Places where the dataset is looked for, in order.  MVTEC_ROOT overrides all.
OFFICIAL_ROOTS = (
    Path(__file__).resolve().parent / "data" / "mvtec_ad",
    Path.home() / "fun" / "industrial-anomaly-detection" / "data" / "mvtec_ad",
    Path.home() / "datasets" / "mvtec_ad",
    Path.home() / "datasets" / "mvtec_anomaly_detection",
)
# The project's own Roboflow dataset (data/roboflow_v2) is the default; the
# official release is the fallback.
CANDIDATE_ROOTS = (
    Path(__file__).resolve().parent / "data" / "roboflow_v2",
    Path(__file__).resolve().parent / "data" / "mvtec_ad",
    Path.home() / "fun" / "industrial-anomaly-detection" / "data" / "mvtec_ad",
    Path.home() / "datasets" / "mvtec_ad",
    Path.home() / "datasets" / "mvtec_anomaly_detection",
)


@dataclass(frozen=True)
class Sample:
    """One image of the dataset."""

    path: Path
    split: str  # "train" or "test"
    defect_type: str  # "good" or the defect folder name
    mask_path: Path | None

    @property
    def label(self) -> int:
        """0 = normal, 1 = defective."""
        return int(self.defect_type != "good")

    @property
    def name(self) -> str:
        return f"{self.defect_type}/{self.path.name}"


def _is_category_dir(path: Path) -> bool:
    return (path / "train" / "good").is_dir() and (path / "test").is_dir()


def find_dataset_root() -> Path:
    """Return the first directory that contains at least one valid category."""
    env = os.environ.get("MVTEC_ROOT")
    candidates = [Path(env)] if env else []
    candidates += list(CANDIDATE_ROOTS)
    for root in candidates:
        if root.is_dir() and any(_is_category_dir(root / c) for c in CATEGORIES):
            return root
    raise FileNotFoundError(
        "MVTec AD not found. Set MVTEC_ROOT to the folder that contains the category "
        f"folders (looked in: {', '.join(str(c) for c in candidates)})."
    )


def dataset_tag() -> str:
    """Short name of the active dataset: "" for official MVTec AD, else the root folder name.

    Built models and results are stored under this tag, so a model built on one
    dataset (e.g. ``data/roboflow_v2``) is never reused for another.
    """
    root = find_dataset_root().resolve()
    if any(root == c.resolve() for c in OFFICIAL_ROOTS):
        return ""
    return root.name


def describe() -> dict:
    """Display names for the active dataset, used by the app and the PDF report."""
    tag = dataset_tag()
    if not tag:
        return {"name": "MVTec AD", "short": "MVTec AD", "split": "official test split",
                "test_label": "MVTec test image",
                "source": "Official MVTec AD release (Bergmann et al., CVPR 2019)"}
    if tag.startswith("roboflow"):
        version = tag.split("_", 1)[1] if "_" in tag else tag
        return {"name": f"Roboflow bottle-anomaly {version}", "short": f"Roboflow {version}",
                "split": "test split", "test_label": "Roboflow test image",
                "source": f"Roboflow project bottle-anomaly, version {version}: 200 MVTec AD bottle images, "
                          "split 80/10/10 (train only normal), Auto-Orient + Stretch to 224x224"
                          + ("" if version == "v2" else ", 7 augmentations on train (x3)")}
    return {"name": tag, "short": tag, "split": "test split", "test_label": "Test image",
            "source": f"Custom dataset at {find_dataset_root()}"}


def output_key(category: str) -> Path:
    """Relative folder for a category's artifacts/results, e.g. ``bottle`` or ``roboflow_v2/bottle``."""
    tag = dataset_tag()
    return Path(tag) / category if tag else Path(category)


DATA_DIR = Path(__file__).resolve().parent / "data"
LAUNCH_ROOT = os.environ.get("MVTEC_ROOT")  # what the app was started with (before any switching)

DATASET_CHOICES = (
    ("roboflow_v2", "Roboflow v2 · no augmentation (final)"),
    ("roboflow_v1", "Roboflow v1 · 7 augmentations"),
)


def available_datasets() -> list[tuple[str, Path]]:
    """Selectable datasets as (label, root): Roboflow v2, Roboflow v1, official MVTec AD."""
    found = [(label, DATA_DIR / tag) for tag, label in DATASET_CHOICES
             if any(_is_category_dir(DATA_DIR / tag / c) for c in CATEGORIES)]
    official = next((r for r in OFFICIAL_ROOTS if r.is_dir() and any(_is_category_dir(r / c) for c in CATEGORIES)),
                    None)
    if official:
        found.append(("Official MVTec AD (full release)", official))
    return found


def available_categories(root: Path) -> list[str]:
    return [c for c in CATEGORIES if _is_category_dir(root / c)]


def _pngs(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.png") if p.is_file())


def train_samples(root: Path, category: str) -> list[Sample]:
    return [Sample(p, "train", "good", None) for p in _pngs(root / category / "train" / "good")]


def test_samples(root: Path, category: str) -> list[Sample]:
    samples = []
    test_dir = root / category / "test"
    for defect_dir in sorted(d for d in test_dir.iterdir() if d.is_dir()):
        for path in _pngs(defect_dir):
            mask = None
            if defect_dir.name != "good":
                mask = root / category / "ground_truth" / defect_dir.name / f"{path.stem}_mask.png"
            samples.append(Sample(path, "test", defect_dir.name, mask))
    return samples


def valid_samples(root: Path, category: str) -> list[Sample]:
    """Normal images of an explicit ``valid`` split (Roboflow exports); empty for official MVTec AD.

    Only ``valid/good`` is returned: calibration sets thresholds from normal images.
    """
    return [Sample(p, "valid", "good", None) for p in _pngs(root / category / "valid" / "good")]


def split_train_validation(samples: list[Sample], fraction: float = 0.1, seed: int = 0):
    """Hold out a fraction of the *normal training* images for threshold calibration."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(samples))
    n_val = max(1, int(round(len(samples) * fraction)))
    val_idx = set(order[:n_val].tolist())
    train = [s for i, s in enumerate(samples) if i not in val_idx]
    val = [s for i, s in enumerate(samples) if i in val_idx]
    return train, val


def load_rgb(path: Path | str) -> np.ndarray:
    """RGB uint8 array (H, W, 3)."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_mask(sample: Sample, shape: tuple[int, int]) -> np.ndarray:
    """Binary ground-truth mask (H, W) in {0, 1}; all zeros for normal images."""
    if sample.mask_path is None:
        return np.zeros(shape, dtype=np.uint8)
    with Image.open(sample.mask_path) as mask:
        array = np.asarray(mask.convert("L"))
    return (array > 127).astype(np.uint8)


def validate_category(root: Path, category: str) -> dict:
    """Counts and integrity checks for one category."""
    train = train_samples(root, category)
    test = test_samples(root, category)
    defective = [s for s in test if s.label == 1]
    problems = []
    for sample in defective:
        if sample.mask_path is None or not sample.mask_path.is_file():
            problems.append(f"missing mask for {sample.name}")
            continue
        with Image.open(sample.path) as image, Image.open(sample.mask_path) as mask:
            if image.size != mask.size:
                problems.append(f"size mismatch {sample.name}: {image.size} vs {mask.size}")
    if not train:
        problems.append("no training images")
    return {
        "category": category,
        "root": str(root),
        "train_images": len(train),
        "valid_images": len(_pngs(root / category / "valid" / "good"))
        + sum(len(_pngs(d)) for d in (root / category / "valid").glob("*") if d.is_dir() and d.name != "good"),
        "test_images": len(test),
        "test_normal": len(test) - len(defective),
        "test_defective": len(defective),
        "defect_types": sorted({s.defect_type for s in defective}),
        "masks": sum(1 for s in defective if s.mask_path and s.mask_path.is_file()),
        "problems": problems,
        "valid": not problems,
    }
