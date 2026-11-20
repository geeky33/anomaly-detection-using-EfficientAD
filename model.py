"""PatchCore-style anomaly detection with a pretrained ResNet18 (CPU friendly).

What is implemented (and what is simplified compared with the PatchCore paper,
Roth et al., CVPR 2022):

1. **Backbone** - torchvision ResNet18 with ImageNet weights, frozen.  The
   paper uses WideResNet-50; ResNet18 is ~4x cheaper (simplification).
2. **Features** - ``layer2`` (128 ch, 28x28 for a 224 input) and ``layer3``
   (256 ch, 14x14).  Each map is averaged over a 3x3 neighbourhood
   ("locally aware patch features", paper Eq. 2), ``layer3`` is upsampled to
   28x28 and both are concatenated -> one 384-d vector per patch.
   (The paper additionally projects each layer to 1024-d; skipped.)
3. **Memory bank** - patch vectors of all normal training images.
4. **Coreset** - greedy k-center selection (paper Alg. 1) on a 128-d random
   projection keeps ``coreset_ratio`` of the bank (capped at ``max_bank``).
5. **Score** - Euclidean distance of each test patch to its nearest
   memory-bank vector.  Image score = maximum patch distance.  The paper's
   re-weighting of the image score (Eq. 7) is not implemented.
6. **Anomaly map** - patch distances upsampled to the input size, Gaussian
   smoothed (sigma = 4) and resized to the original image size.

Thresholds are calibrated on held-out *normal training* images only (see
``calibrate``); test labels and masks are never used for that.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Config:
    image_size: int = 224
    coreset_ratio: float = 0.1
    max_bank: int = 20_000
    projection_dim: int = 128
    smoothing_sigma: float = 4.0
    batch_size: int = 8
    seed: int = 0
    # Calibration (normal validation images only)
    image_threshold_rule: str = "max score of normal validation images"
    pixel_quantile: float = 0.995


@dataclass
class Calibration:
    image_threshold: float
    pixel_threshold: float
    validation_scores: list[float] = field(default_factory=list)
    num_validation_images: int = 0


class PatchCoreLite:
    """Pretrained ResNet18 + nearest-neighbour memory bank."""

    def __init__(self, config: Config | None = None, device: str = "cpu") -> None:
        self.config = config or Config()
        self.device = device
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.eval().requires_grad_(False)
        self.backbone = backbone.to(device)
        self.memory_bank: torch.Tensor | None = None
        self.calibration: Calibration | None = None
        self.info: dict = {}

    # ------------------------------------------------------------------ features
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """RGB uint8 (H, W, 3) -> normalised tensor (3, S, S)."""
        size = self.config.image_size
        resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        x = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy())

    @torch.no_grad()
    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        """(B, 3, S, S) -> patch features (B, 384, S/8, S/8)."""
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(batch.to(self.device)))))
        f2 = b.layer2(b.layer1(x))
        f3 = b.layer3(f2)
        f2 = F.avg_pool2d(f2, 3, stride=1, padding=1)
        f3 = F.avg_pool2d(f3, 3, stride=1, padding=1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], dim=1)

    def _batches(self, images: list[np.ndarray] | list[Path], loader=None):
        size = self.config.batch_size
        for start in range(0, len(images), size):
            chunk = images[start : start + size]
            arrays = [loader(item) if loader else item for item in chunk]
            yield torch.stack([self.preprocess(a) for a in arrays])

    # --------------------------------------------------------------------- fit
    @torch.no_grad()
    def fit(self, train_paths: list[Path], loader, progress=None) -> dict:
        """Build the memory bank from normal training images (batch by batch)."""
        start = time.perf_counter()
        chunks = []
        for index, batch in enumerate(self._batches(train_paths, loader)):
            features = self.embed(batch)  # (B, C, h, w)
            chunks.append(features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]).cpu())
            if progress:
                progress(min(1.0, (index + 1) * self.config.batch_size / len(train_paths)) * 0.6)
        bank = torch.cat(chunks)
        del chunks
        total = bank.shape[0]
        target = min(self.config.max_bank, max(1, int(total * self.config.coreset_ratio)))
        indices = greedy_coreset(bank, target, self.config.projection_dim, self.config.seed, progress)
        self.memory_bank = bank[indices].contiguous()
        self.info = {
            "train_images": len(train_paths),
            "patch_vectors": int(total),
            "memory_bank_size": int(self.memory_bank.shape[0]),
            "feature_dim": int(self.memory_bank.shape[1]),
            "fit_seconds": round(time.perf_counter() - start, 1),
        }
        return self.info

    # ----------------------------------------------------------------- predict
    @torch.no_grad()
    def patch_scores(self, batch: torch.Tensor) -> torch.Tensor:
        """Nearest-neighbour distance of every patch: (B, h, w)."""
        if self.memory_bank is None:
            raise RuntimeError("The memory bank is empty - build the model first.")
        features = self.embed(batch)
        b, c, h, w = features.shape
        queries = features.permute(0, 2, 3, 1).reshape(-1, c)
        bank = self.memory_bank.to(self.device)
        best = torch.empty(queries.shape[0], device=self.device)
        for start in range(0, queries.shape[0], 4096):  # bounded memory
            best[start : start + 4096] = torch.cdist(queries[start : start + 4096], bank).min(dim=1).values
        return best.reshape(b, h, w)

    def anomaly_map(self, scores: torch.Tensor) -> np.ndarray:
        """(h, w) patch distances -> smoothed (S, S) map."""
        size = self.config.image_size
        up = F.interpolate(scores[None, None], size=(size, size), mode="bilinear", align_corners=False)
        smoothed = cv2.GaussianBlur(up[0, 0].cpu().numpy(), (0, 0), self.config.smoothing_sigma)
        return smoothed.astype(np.float32)

    def predict(self, image: np.ndarray) -> dict:
        """Full inference for one RGB image.

        Returns the image score, the anomaly map at the original resolution,
        the binary mask (if calibrated), the prediction and the latency.
        """
        start = time.perf_counter()
        scores = self.patch_scores(self.preprocess(image)[None])[0]
        amap_small = self.anomaly_map(scores)
        amap = cv2.resize(amap_small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        score = float(scores.max())
        latency_ms = (time.perf_counter() - start) * 1000.0
        result = {"score": score, "anomaly_map": amap, "anomaly_map_small": amap_small, "latency_ms": latency_ms}
        if self.calibration:
            result["mask"] = (amap >= self.calibration.pixel_threshold).astype(np.uint8)
            result["is_defective"] = score >= self.calibration.image_threshold
            result["image_threshold"] = self.calibration.image_threshold
            result["pixel_threshold"] = self.calibration.pixel_threshold
        return result

    # --------------------------------------------------------------- calibrate
    def calibrate(self, val_paths: list[Path], loader) -> Calibration:
        """Thresholds from held-out NORMAL training images (no test data).

        * image threshold = highest image score among the validation images
          (every validation image is classified as normal);
        * pixel threshold = ``pixel_quantile`` quantile of all validation
          anomaly-map pixels (~0.5 % of normal pixels flagged).
        """
        scores, pixels = [], []
        for path in val_paths:
            image = loader(path)
            patch = self.patch_scores(self.preprocess(image)[None])[0]
            scores.append(float(patch.max()))
            pixels.append(self.anomaly_map(patch).ravel())
        all_pixels = np.concatenate(pixels)
        self.calibration = Calibration(
            image_threshold=float(max(scores)),
            pixel_threshold=float(np.quantile(all_pixels, self.config.pixel_quantile)),
            validation_scores=scores,
            num_validation_images=len(val_paths),
        )
        return self.calibration

    # ------------------------------------------------------------- persistence
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({"memory_bank": self.memory_bank}, directory / "memory_bank.pt")
        meta = {"config": asdict(self.config), "info": self.info,
                "calibration": asdict(self.calibration) if self.calibration else None,
                "backbone": "torchvision resnet18 IMAGENET1K_V1 (frozen)"}
        (directory / "model.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path, device: str = "cpu") -> "PatchCoreLite":
        meta = json.loads((directory / "model.json").read_text())
        model = cls(Config(**meta["config"]), device)
        model.memory_bank = torch.load(directory / "memory_bank.pt", map_location="cpu")["memory_bank"]
        model.info = meta["info"]
        if meta.get("calibration"):
            model.calibration = Calibration(**meta["calibration"])
        return model


@torch.no_grad()
def greedy_coreset(features: torch.Tensor, n: int, projection_dim: int = 128, seed: int = 0, progress=None) -> torch.Tensor:
    """Greedy k-center selection: repeatedly add the point farthest from the set.

    Distances are computed on a random 128-d projection (as in PatchCore) and
    updated incrementally, so memory stays O(N).
    """
    generator = torch.Generator().manual_seed(seed)
    if features.shape[1] > projection_dim:
        projection = torch.randn(features.shape[1], projection_dim, generator=generator) / projection_dim**0.5
        reduced = features @ projection
    else:
        reduced = features
    # Squared distances via ||a||^2 + ||b||^2 - 2ab: one matrix-vector product
    # per step (torch.cdist per step was ~5x slower).  The arg-max is the same
    # for squared and plain distances, so the selection is unchanged.
    norms = (reduced * reduced).sum(dim=1)
    first = int(torch.randint(len(reduced), (1,), generator=generator))
    selected = [first]
    min_dist = (norms + norms[first] - 2 * reduced @ reduced[first]).clamp_(min=0)
    for step in range(1, n):
        index = int(torch.argmax(min_dist))
        selected.append(index)
        torch.minimum(min_dist, (norms + norms[index] - 2 * reduced @ reduced[index]).clamp_(min=0), out=min_dist)
        if progress and step % 500 == 0:
            progress(0.6 + 0.4 * step / n)
    return torch.tensor(selected, dtype=torch.long)
