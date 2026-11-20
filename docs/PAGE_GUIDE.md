# Page Guide — Industrial AI Inspection Platform

This guide explains every page of the web application: what it is for, what you see, which controls
it has, and how to read the results. Start the app with:

```bash
cd /home/aarya/fun/industrial-anomaly-detection-mvp
~/miniconda3/bin/streamlit run app.py      # then open http://localhost:8501
```

All documents (this guide, the README, the demo guide and the project explanation) can also be read
inside the app on the **Documentation** page.

> **Dataset and models in the final version**
> * By default the app uses our **Roboflow dataset** (`data/roboflow_v2`): 200 MVTec AD bottle images,
>   160 train / 20 valid / 20 test.
> * All **four models are available and evaluated**:
>   * EfficientAD-S and EfficientAD-M (main model, trained on a Colab GPU with a reduced protocol)
>   * PatchCore ResNet18 and PatchCore WRN-50 (baseline)
> * Where this guide says "official test image", read "test image of the active dataset". The app shows
>   *Roboflow test image* or *MVTec test image* accordingly.
> * Set `MVTEC_ROOT` to the official MVTec AD folder to use the original release. EfficientAD is then
>   unavailable, because it was trained only for the Roboflow dataset.
> * **Sidebar → Dataset** switches the whole website between *Roboflow v2 · no augmentation (final)*
>   (4 models), *Roboflow v1 · 7 augmentations* (480 augmented training images, PatchCore ResNet18
>   only) and *Official MVTec AD*. Switching clears the selected image and cached predictions. The
>   **Dataset Explorer** shows a v1 vs v2 augmentation-study table.
> * Screenshots of the final version: `docs/screenshots/new_*.png` (v2) and `new_v1_*.png` (v1).

---

## Layout common to every page

**Top header.** The platform logo and title on the left. On the right, three status chips:
* **Dataset**: the active dataset and category (e.g. `Roboflow v2 · bottle`).
* **Model**: the active model (e.g. `PatchCore · ResNet18`).
* **Status**: how many of the four models have a checkpoint (green dot), or an amber dot when the
  active model is not built.

**Sidebar.**
* **Navigation**: the ten pages. The highlighted (cyan) entry is the open page.
* **Product category**: which MVTec AD category to work with. Only `bottle` has been built and evaluated.
* **Active model**: lists only models that have a checkpoint. It is used by Live Inspection,
  Analytics and Reports.
* **Status**: free RAM and the number of CPU threads used for inference.

**Colour meaning (same on every page)**

| Colour | Meaning |
|---|---|
| Green | normal / success / ground-truth (annotated) defect |
| Red | anomaly / defective / model prediction / error |
| Amber | warning / unavailable / thresholds |
| Cyan | information / selected item |

**Shared state.** The image you select, the active model and the predictions already computed are
remembered while you move between pages. A typical flow:
Dataset Explorer → Inspect → Live Inspection → Compare all models → Analytics → Reports.

---

## 1. Overview

**Purpose:** a one-screen summary of the dataset, the models and the latest activity.

**What you see**
1. **KPI cards**
   * *Dataset images*: training + test images of the category, with the mask count.
   * *Models available*: models with a checkpoint (4 / 4 on the Roboflow dataset).
   * *Best pixel AUROC*: the evaluated model that localises defects best.
   * *Inspections run*: rows in the inspection history, and how many were flagged defective.
2. **Model availability**: one card per model. Badges show *checkpoint ✓ / no checkpoint* and
   *evaluated ✓ / not evaluated*. Evaluated models show their metrics; unavailable models show an
   amber box explaining why. No numbers are ever shown for them.
3. **Performance charts** (evaluated models only):
   * *Image level · detection*: image AUROC and F1.
   * *Pixel level · localisation*: pixel AUROC, pixel F1 and IoU.
   * *Inference latency*: mean, median and 95th-percentile milliseconds per image.

   Hover over a bar to see its exact value.
4. **Dataset**: number of images per split and defect type (green = normal, red = defective).
5. **Recent inspection activity**: the last six inspections.
6. **Latest anomaly examples**: overlays of the most recent images flagged defective.

**Data source:** saved evaluation files `results/<category>/<model>/metrics.json` and the history
database. Nothing on this page runs a model.

---

## 2. Live Inspection — the inspection workspace

**Purpose:** run a model on one image and examine the result in detail.

### Choosing an image
* **Source → MVTec test image**: pick any official test image from the drop-down (`defect type /
  file name`, e.g. `broken_large/000.png`). These images have a known ground truth.
* **Source → Upload**: drag in your own PNG/JPG. Uploads have **no ground truth**.
* Images opened from the Dataset Explorer arrive here already selected.

Press **▶ Run <model>** to analyse the image. Every run is saved to the Inspection History once.
Switching views afterwards does not re-run the model or save again.

### Side panel (right)
| Item | Meaning |
|---|---|
| Verdict banner | **⚠ DEFECTIVE** (red) if the anomaly score ≥ image threshold, else **✓ NORMAL** (green) |
| Ground truth → correct / wrong | only for official test images: whether the verdict matches the real label |
| Anomaly score | the image score: the largest patch distance from normal |
| Image threshold | decision boundary, calibrated on normal training images only |
| Defect area | percentage of pixels in the predicted mask |
| Latency | time of the model forward pass for this image |
| Regions | number of separate connected areas in the predicted mask |
| Pixel threshold | value above which a pixel is marked defective |
| IoU / Dice vs GT | overlap between predicted mask and ground-truth mask (official defective images only) |
| Score vs. threshold bar | where the score lies: left half normal, right half anomalous |

The buttons below jump to **Model Comparison** (same image), **Analytics** and **Reports**.

### Visualisation modes
Use the segmented control above the image. Image views support **zoom** (scroll or toolbar),
**pan** (drag) and **reset** (double-click or the home icon).

1. **Original**: the unmodified image.
2. **Anomaly heatmap**: the score of every pixel (blue = normal, red = strongly anomalous). Hover
   to read the exact score. The colour bar is fixed per model: the pixel threshold sits in the
   middle colour, so a normal image stays blue.
3. **Predicted mask**: pixels above the pixel threshold, in red. Yellow boxes (R1, R2, …) mark
   each connected region.
4. **Heatmap overlay**: the heatmap blended over the photo, with the predicted mask outlined in red
   and region boxes in yellow.
5. **Ground-truth mask**: the human annotation of the defect, in green. This is *not* a model
   output. Normal images have an empty mask.
6. **Ground-truth comparison**: four panels (original, predicted mask, ground-truth mask, and the
   comparison) plus a zoomable comparison image:
   * **Red**: predicted but not a defect (false positive)
   * **Green**: a real defect the model missed
   * **Yellow**: overlap (correctly found)

   A line below gives IoU, Dice and the pixel counts of each colour.

For **uploaded** images, modes 5 and 6 show a disabled panel explaining that uploads have no annotation.

### Tabs below the image
* **Before / after slider**: the original on the left, a chosen view on the right (overlay,
  heatmap, mask, or ground-truth comparison for official images). Drag the cyan handle to compare,
  scroll to zoom (both sides zoom together), drag to pan, double-click to reset.
* **Synchronized view**: original and model output side by side; zooming or panning one moves the
  other identically.
* **Defect regions**: table of every predicted region with position, size, area (%) and maximum
  score. Specks smaller than 0.05 % of the image are ignored.

**Technical details** (expander): image size, model description and how the thresholds were chosen.

---

## 3. Model Laboratory

**Purpose:** inspect each model's internals and run maintenance actions.

**Controls:** the **Model** selector at the top (all four models).

**What you see**
* The model card (status badges, description, metrics if evaluated).
* **Strengths and limitations** of the model.
* **Checkpoint**: file path, size on disk and the size of the ImageNet backbone weights. For
  a model without a checkpoint for the active dataset, an amber box says so and explains how to train it.
* **Memory bank & calibration**:
  * *Memory bank*: vectors kept after coreset selection.
  * *Patch vectors*: vectors extracted from the normal training images.
  * The image and pixel thresholds, and where they came from.
  * For the ResNet18 model, a plot of the held-out normal validation scores. The image threshold
    (dashed amber line) equals their maximum.

**Actions**
* **Evaluate / Re-run evaluation**: scores the model on the full test split of the active dataset and saves new
  metrics. It refuses to start if free RAM is below 1.2 GB.
* **Rebuild memory bank** (ResNet18 only, ~4 min): rebuilds the model from the normal training images.
  It refuses to start below 1.8 GB free RAM.
* Main-platform checkpoints (WRN-50, EfficientAD-S/-M) are read-only in the app. EfficientAD is
  trained on a GPU with `colab/train_efficientad_colab.ipynb` and imported with
  `scripts/import_colab_models.py`, not from the app.

---

## 4. Model Comparison — comparison centre

**Purpose:** compare all models fairly, on numbers and on the same image.

**What you see**
1. **Four model cards**: PatchCore · ResNet18, PatchCore · WRN-50, EfficientAD-S, EfficientAD-M.
   Unavailable models show *no checkpoint / not evaluated* with the reason.
2. **Evaluation protocol table**: for each evaluated model: number of test images, input size,
   pixel-metric grid, how the thresholds were calibrated, and when it was evaluated. The caption
   confirms all models were evaluated on the identical image list.
3. **Metrics charts** (evaluated models only):
   * Image-level AUROC
   * Pixel-level AUROC (note: the axis starts at 0.9 to make the difference visible)
   * Precision & recall, image and pixel level
   * F1 & IoU
   * Latency: mean, median, p95
   * Model size on disk: memory bank plus backbone weights
   * Peak RAM while evaluating (measured)

   **Full comparison table** (expander) lists every number.
4. **Reading the comparison**: a short, data-driven summary: which model localises best, which is
   fastest, and a reminder that only one category was evaluated.
5. **Side-by-side inference**:
   * The image is the one selected in Live Inspection or the Dataset Explorer.
   * **Show** switches between heatmap, predicted mask and overlay.
   * **▶ Run all available models** runs every model that has a checkpoint.
   * One panel per model with verdict, score, latency, and IoU/Dice when ground truth exists.
   * A model without a checkpoint shows *unavailable — trained checkpoint missing — no output is simulated*.
     On the Roboflow dataset all four models run. EfficientAD-M takes ~6 s on CPU.
   * The last panel is the ground-truth mask.
   * Each heatmap uses its own model's scale (its pixel threshold is the middle colour).
6. **Strengths and limitations** of each model (expanders).

**Fairness rules:**
* Same test images and same evaluation code for every model.
* Thresholds come from normal training images only.
* Models without a checkpoint never appear in charts or rankings.

---

## 5. Analytics

**Purpose:** aggregate performance of one evaluated model over the whole test split.

**Controls:** the **Model** selector (evaluated models only).

**What you see**
* **KPI row**: image AUROC, image F1, pixel AUROC, pixel F1, pixel IoU, median latency.
* **Image level · detection**
  * *ROC curve*: true-positive vs. false-positive rate for all evaluated models. The selected
    model is drawn thicker; the dashed line is chance.
  * *Confusion matrix · images*: normal/defective vs. predicted normal/defective.
  * *Score distribution*: image scores of normal (green) and defective (red) test images, with the
    calibrated threshold (dashed amber). Images to the right of the line are flagged defective.
* **Pixel level · localisation**
  * *Confusion matrix · pixels*: counted over every pixel of every test image (256×256 grid).
  * *Mean IoU per defect type*: hover to see how many images of that type were detected.
  * *IoU per defective image*: box plot with every image as a dot (hover shows the file name).
* **Latency**: histogram of per-image latency and the mean / median / p95 summary.
* **Per defect type** table and **Per-image results** (expander): score, prediction, latency and IoU
  of every test image.

**Data source:** `results/<category>/<model>/metrics.json` and `per_image.csv`.

---

## 6. Dataset Explorer

**Purpose:** browse the data used by the models: the Roboflow dataset by default, or the official MVTec AD release.

**What you see**
* **KPI cards**: training images (normal), normal test images, defective test images (with defect
  types), ground-truth masks and an integrity check (all masks present and aligned).
* **Tabs**:
  * *Normal · train*: defect-free images used to build the models.
  * *Normal · test*: defect-free test images.
  * *Defect · <type>*: defective test images with the ground-truth defect outlined in green.
* **Inspect** under any test image opens it in Live Inspection, ready to run.

Only `train/good` images build the models. Test images and masks are used only for evaluation.

---

## 7. Inspection History

**Purpose:** a log of every inspection run in the app.

**What you see**
* **KPI cards**: number of inspections, how many were flagged defective, agreement with ground truth
  (official test images only) and median latency.
* **Filters**: by model and by prediction.
* **Table**: timestamp, category, source (test image or upload), image, model, score, threshold,
  prediction, defect area %, latency, ground truth, IoU, Dice, number of regions.
* **Download CSV**: exports the filtered table.
* **Latest overlays**: thumbnails of the saved overlay images.

Stored locally in `results/history.sqlite3`; overlays in `results/history/`.

---

## 8. Reports

**Purpose:** produce a PDF summary.

**What you see**
* *Active model* card, and whether it has been evaluated.
* *Current inspection* card: the image last inspected with the active model. If there is none, run
  one in Live Inspection to include it.
* **Generate PDF report**, then **Download PDF**.

**The PDF contains**
1. Project description
2. Model description and memory-bank settings
3. Dataset information
4. Evaluation metrics (image and pixel level) with confusion matrix and ROC curve
5. Model comparison table, plus a list of models that were not compared and why
6. The current inspection, including the ground-truth comparison for official images
7. Three sample inspections (one normal, two defective)
8. Limitations

Reports are saved in `results/reports/`.

---

## 9. Settings

**Purpose:** runtime preferences and maintenance.

* **CPU threads for inference**: fewer threads leave more CPU for other applications.
* **Heatmap overlay opacity**: how strongly the heatmap is blended over the photo.
* **Clear cached predictions**: forget the predictions kept in this browser session.
* **Unload models from memory**: frees RAM used by loaded models.
* **Clear inspection history**: tick the confirmation box first.
* **System**: dataset location, number of categories, device, PyTorch version, free RAM, and whether
  the main platform (source of the WRN-50 checkpoint) was found.

---

## 10. Documentation

**Purpose:** read the project documents without leaving the app. Tabs:
* **Page guide**: this document.
* **README**: project summary, results, models, commands.
* **Demo guide**: 5-minute demo script.
* **Project explanation**: theory for the report and the viva, including 32 viva questions with
  answers.

The same files are in the project folder:

```
/home/aarya/fun/industrial-anomaly-detection-mvp/README.md
/home/aarya/fun/industrial-anomaly-detection-mvp/docs/PAGE_GUIDE.md
/home/aarya/fun/industrial-anomaly-detection-mvp/docs/DEMO_GUIDE.md
/home/aarya/fun/industrial-anomaly-detection-mvp/docs/PROJECT_EXPLANATION.md
```
