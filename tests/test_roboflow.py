"""Checks of the Roboflow dataset (the app's default) and its four models.

Run:  python -m pytest -q tests/test_roboflow.py
Requires data/roboflow_v2 (scripts/convert_roboflow_export.py) and the models
built/imported for it (evaluation.py, scripts/import_colab_models.py).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
ROBOFLOW = PROJECT / "data" / "roboflow_v2"
pytestmark = pytest.mark.skipif(not ROBOFLOW.is_dir(), reason="Roboflow dataset not converted")

CHECK = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import dataset as ds, model_registry as reg
info = ds.validate_category(ds.find_dataset_root(), "bottle")
status = {k: (s["checkpoint_available"], s["evaluated"]) for k, s in
          ((k, reg.status(k, "bottle")) for k in reg.SPECS)}
print(json.dumps({"tag": ds.dataset_tag(), "info": info, "status": status, "describe": ds.describe()}))
"""


def _run(env_root: str | None) -> dict:
    import json

    env = {k: v for k, v in os.environ.items() if k != "MVTEC_ROOT"}
    if env_root:
        env["MVTEC_ROOT"] = env_root
    env["MVP_HISTORY_DB"] = str(Path(tempfile.mkdtemp()) / "h.sqlite3")
    out = subprocess.run([sys.executable, "-c", CHECK, str(PROJECT)], env=env, cwd=PROJECT,
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def test_roboflow_is_default_dataset_with_80_10_10_split() -> None:
    result = _run(None)
    info = result["info"]
    assert result["tag"] == "roboflow_v2"
    assert info["valid"], info["problems"]
    assert (info["train_images"], info["valid_images"], info["test_images"]) == (160, 20, 20)
    assert (info["test_normal"], info["test_defective"], info["masks"]) == (10, 10, 10)
    assert "Roboflow" in result["describe"]["name"]


def test_all_four_models_available_and_evaluated_on_roboflow() -> None:
    status = _run(None)["status"]
    assert status == {k: [True, True] for k in
                      ("patchcore_r18", "patchcore_wrn50", "efficientad_s", "efficientad_m")}, status


def test_official_mvtec_still_selectable() -> None:
    result = _run(str(Path.home() / "fun" / "industrial-anomaly-detection" / "data" / "mvtec_ad"))
    assert result["tag"] == ""
    assert result["info"]["train_images"] == 209


SWITCH = r"""
import json, os, sys, tempfile
from pathlib import Path
os.environ["MVP_HISTORY_DB"] = str(Path(tempfile.mkdtemp()) / "h.sqlite3")
os.chdir(sys.argv[1]); sys.path.insert(0, sys.argv[1])
from streamlit.testing.v1 import AppTest
app = AppTest.from_file("app.py", default_timeout=600); app.run()
out = {}
for label in app.selectbox(key="dataset").options:
    app.selectbox(key="dataset").set_value(label).run()
    app.radio(key="nav").set_value("Dataset Explorer").run()
    text = " ".join(m.value for m in app.markdown)
    out[label] = {"exception": [str(e.value) for e in app.exception],
                  "models": list(app.selectbox(key="model_key").options),
                  "versions_card": "augmentation study" in text,
                  "kpi_train": next((m for m in app.markdown if "Training images" in m.value), None) is not None,
                  "root": os.environ["MVTEC_ROOT"]}
print(json.dumps(out))
"""


def test_dataset_selector_switches_between_v2_v1_and_official() -> None:
    import json

    env = {k: v for k, v in os.environ.items() if k != "MVTEC_ROOT"}
    out = subprocess.run([sys.executable, "-c", SWITCH, str(PROJECT)], env=env, cwd=PROJECT,
                         capture_output=True, text=True, check=True).stdout
    result = json.loads(out.strip().splitlines()[-1])
    labels = list(result)
    assert labels[0].startswith("Roboflow v2") and labels[1].startswith("Roboflow v1")
    v2, v1 = result[labels[0]], result[labels[1]]
    assert not v2["exception"] and not v1["exception"]
    assert len(v2["models"]) == 4 and v1["models"] == ["PatchCore · ResNet18"]
    assert v2["root"].endswith("roboflow_v2") and v1["root"].endswith("roboflow_v1")
    assert v2["versions_card"] and v1["versions_card"]
    if len(labels) > 2:
        assert not result[labels[2]]["exception"]
