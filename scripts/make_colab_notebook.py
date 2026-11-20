"""Write colab/train_efficientad_colab.ipynb: trains EfficientAD-S, EfficientAD-M and PatchCore WRN-50
on the Roboflow v2 bottle dataset on a Colab GPU, using the package built into
colab/efficientad_colab_package.zip.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "colab" / "train_efficientad_colab.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# EfficientAD-S / EfficientAD-M / PatchCore WRN-50 on the Roboflow bottle dataset

1. **Runtime → Change runtime type → T4 GPU → Save.**
2. **Runtime → Run all.**
3. When asked, upload `efficientad_colab_package.zip`.
4. At the end, `roboflow_v2_models.zip` downloads automatically. Copy it into the project's `colab/` folder.

The iteration counts below are a **reduced protocol** sized for about 10 minutes on a T4. The paper uses
70,000 iterations. Lower `ITERS_S` / `ITERS_M` if you have less time.
"""),
    code("""
ITERS_S = 2500   # EfficientAD-S iterations (paper: 70000)
ITERS_M = 1500   # EfficientAD-M iterations (paper: 70000)
"""),
    code("""
import torch, subprocess
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> T4 GPU, then run again."
print(torch.cuda.get_device_name(0))
"""),
    code("""
import os, zipfile
from google.colab import files
if not os.path.isdir("/content/iad"):
    uploaded = files.upload()  # choose efficientad_colab_package.zip
    name = next(iter(uploaded))
    zipfile.ZipFile(name).extractall("/content")
print(sorted(os.listdir("/content/iad")))
"""),
    code("""
!pip install -q pydantic-settings sqlalchemy python-dotenv python-multipart
"""),
    code("""
import os, sqlite3, subprocess, sys, time, json
P = "/content/iad"
ENV = dict(os.environ,
           DATASET_ROOT=f"{P}/data/roboflow_v2",
           PRETRAINING_DATASET_DIR=f"{P}/data/imagenet_subset",
           DATABASE_URL=f"sqlite:///{P}/results/db.sqlite3")
os.makedirs(f"{P}/results", exist_ok=True)
TIMES = {}

def progress(arch):
    try:
        con = sqlite3.connect(f"{P}/results/db.sqlite3")
        row = con.execute("select current_iter, total_iters, stage from training_jobs "
                          "where architecture=? order by id desc limit 1", (arch,)).fetchone()
        con.close()
        return row
    except Exception:
        return None

def train(arch, *extra):
    start = time.time()
    proc = subprocess.Popen([sys.executable, "scripts/train.py", arch, "bottle", "--device", "cuda", *extra],
                            cwd=P, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = 0
    while proc.poll() is None:
        time.sleep(2)
        if time.time() - last > 20:
            last = time.time()
            print(f"  {arch}: {time.time() - start:5.0f}s  {progress(arch)}", flush=True)
    output = proc.stdout.read()
    TIMES[arch] = round(time.time() - start)
    print(output[-1500:])
    assert proc.returncode == 0, f"{arch} failed"
    print(f"== {arch} done in {TIMES[arch]} s")
"""),
    code("""
train("patchcore")   # WideResNet-50 PatchCore baseline (no gradient training)
"""),
    code("""
train("efficientad_s", "--iters", str(ITERS_S))
"""),
    code("""
train("efficientad_m", "--iters", str(ITERS_M))
"""),
    code("""
# Sanity check: a useful model gives different scores to different images (std > 0).
for arch in ("patchcore", "efficientad_s", "efficientad_m"):
    meta = json.load(open(f"{P}/models/{arch}/bottle/metadata.json"))
    cal = meta.get("calibration", {})
    stats = cal.get("image_score_stats", {})
    print(f"{arch:14s} protocol={meta.get('protocol')}  iters={meta.get('total_iters', meta.get('training_config', {}).get('total_iters'))}"
          f"  image_threshold={cal.get('image_threshold'):.4f}  val score std={stats.get('std', float('nan')):.5f}")
print("wall times (s):", TIMES)
"""),
    code("""
import shutil
json.dump({"times_s": TIMES, "iters": {"efficientad_s": ITERS_S, "efficientad_m": ITERS_M},
           "gpu": torch.cuda.get_device_name(0)}, open(f"{P}/models/colab_run.json", "w"), indent=1)
shutil.rmtree("/content/out", ignore_errors=True)
shutil.copytree(f"{P}/models", "/content/out", ignore=shutil.ignore_patterns("pretrained"))
shutil.make_archive("/content/roboflow_v2_models", "zip", "/content/out", ".")
files.download("/content/roboflow_v2_models.zip")
"""),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 0}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1))
print(OUT)
