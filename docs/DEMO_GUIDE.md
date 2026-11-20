# Demo Guide: run, features, where to find every result, 5-minute script

**Project:** Industrial Anomaly Detection and Localization using **EfficientAD** (with **PatchCore**
baseline) on MVTec AD, built on our own **Roboflow** dataset.

---

## 1. Start the website

```bash
cd /home/aarya/fun/industrial-anomaly-detection-mvp
unset MVTEC_ROOT                                 # make sure no old dataset override is set
~/miniconda3/bin/streamlit run app.py
```

Open **http://localhost:8501**.

* Start it **from the project folder**, otherwise the dark theme is not loaded.
* If an instance is already running, stop it (Ctrl+C) and start again.
* Close heavy programs before the demo. The app needs ~1.5 GB of free RAM to load WRN-50 or
  EfficientAD-M. When RAM is short, it unloads other models first and shows a warning instead of
  crashing.
* If port 8501 is busy: `~/miniconda3/bin/streamlit run app.py --server.port 8502`.

**Optional checks before the demo:**

```bash
~/miniconda3/bin/python -m pytest -q tests       # 22 tests, ~2 min (both datasets, all 4 models)
```

---

## 2. What the website has (features)

### Sidebar (on every page)

| Control | What it does |
|---|---|
| **Navigation** | The 10 pages |
| **Dataset** | Switches the whole website between datasets:<br>• **Roboflow v2 · no augmentation (final)**: default, all 4 models<br>• **Roboflow v1 · 7 augmentations**: 480 augmented training images, PatchCore ResNet18 only (the augmentation experiment)<br>• **Official MVTec AD**: full release, PatchCore models only<br>Switching clears the current image and cached predictions. |
| **Product category** | bottle |
| **Active model** | Model used by Live Inspection, Analytics and Reports. Only models that exist for the selected dataset are listed. |
| **Status** | Free RAM and CPU threads |

The **top bar** always shows the active dataset (e.g. `Roboflow v2 · bottle`), the active model, and
how many of the 4 models are available (4/4 on v2, 1/4 on v1).

### The 10 pages

| Page | What it shows |
|---|---|
| **Overview** | KPI cards (200 images = 160 train · 20 valid · 20 test, models available, best pixel AUROC, inspections run), one card per model with its metrics, performance and latency charts, dataset chart, recent inspections |
| **Live Inspection** | Pick a test image (or upload one), **Run** the active model: verdict NORMAL / DEFECTIVE, ground-truth badge (correct / wrong), anomaly score vs threshold, defect area, regions, latency, IoU and Dice vs ground truth. **6 visualisation modes:** original, anomaly heatmap, predicted mask, heatmap overlay, ground-truth mask, ground-truth comparison (red = false alarm, green = missed, yellow = found). Also a **before/after slider** and a **synchronized zoom** view. |
| **Model Laboratory** | Per model: checkpoint file and size, training protocol (e.g. EfficientAD-S: 2,500 iterations on a T4 GPU), thresholds, validation-score plot; buttons to re-evaluate / rebuild |
| **Model Comparison** | All 4 models side by side: AUROC, precision/recall, F1/IoU, latency, size, RAM charts; a "Reading the comparison" summary; **Run all available models** on the current image, giving 4 heatmaps next to the ground truth |
| **Analytics** | For the active model: ROC curve, image and pixel confusion matrices, score distribution vs threshold, IoU per defect type and per image, latency distribution |
| **Dataset Explorer** | Split counts and mask check; **v1 vs v2 augmentation study table**; image galleries (train, test normal, each defect type with ground-truth outline); **Inspect** opens an image in Live Inspection |
| **Inspection History** | Every inspection you ran (time, image, model, verdict, score, IoU), with filters and CSV export |
| **Reports** | **Generate PDF report** for the active model and dataset, then **Download PDF** |
| **Documentation** | README, page guide, this demo guide, project explanation, viva Q&A, readable inside the app |
| **Settings** | Threads, overlay opacity, clear caches / history, system info |

---

## 3. Where to find each result on the website

| Result you want to show | Where | How |
|---|---|---|
| Dataset split 80/10/10 (160 / 20 / 20) | **Overview** → first KPI card; **Dataset Explorer** | Dataset = Roboflow v2 |
| Augmented images (flip, rotate, brightness, blur…) | **Dataset Explorer** → *Normal · train* tab | Dataset = **Roboflow v1** (files `…_aug1`, `…_aug2`) |
| Augmentation helps or not (v1 vs v2) | **Dataset Explorer** → "Roboflow versions · augmentation study" table | either dataset |
| Final 4-model results table | **Overview** model cards, or **Model Comparison** → *Full comparison table* | Dataset = Roboflow v2 |
| EfficientAD vs PatchCore on one image | **Model Comparison** → **Run all available models** | v2; pick the image in Live Inspection first |
| Heatmap / mask / ground truth for one image | **Live Inspection** → Run → visualisation modes | any model |
| ROC curve, confusion matrix, errors per defect | **Analytics** | choose the model as *Active model* |
| Which images a model got wrong | **Analytics** → per-image table (score vs threshold) | e.g. EfficientAD-S misses `contamination_004` |
| Training details (iterations, GPU, time) | **Model Laboratory** | per model |
| PDF report | **Reports** → Generate → Download | the active model + dataset |

**Ready-made results (no website needed):**
* PDF reports in `results/reports/`:
  * `report_roboflow_v2_bottle_efficientad_s_*.pdf`
  * `report_roboflow_v2_bottle_efficientad_m_*.pdf`
  * `report_roboflow_v2_bottle_patchcore_r18_*.pdf`
  * `report_roboflow_v2_bottle_patchcore_wrn50_*.pdf`
  * `report_roboflow_v1_bottle_patchcore_r18_*.pdf`
* Screenshots in `docs/screenshots/new_*.png` (`new_v1_*` = Roboflow v1)
* Raw metrics in `results/roboflow_v2/bottle/<model>/metrics.json` and `per_image.csv`

---

## 4. Final numbers (Roboflow v2 test split: 10 normal + 10 defective)

| | EfficientAD-S | EfficientAD-M | PatchCore R18 | PatchCore WRN-50 |
|---|---|---|---|---|
| Image AUROC | 0.95 | 0.89 | **1.00** | **1.00** |
| Correct of 20 | 18 | 15 | 20 | 20 |
| Pixel AUROC | 0.981 | 0.941 | 0.988 | **0.992** |
| Pixel IoU | **0.514** | 0.417 | 0.347 | 0.375 |
| Latency (CPU) | 0.48 s | 5.7 s | **0.06 s** | 0.30 s |

Augmentation study (PatchCore R18):

| | Train images | Pixel AUROC | IoU | Build time |
|---|---|---|---|---|
| v1 (7 augmentations) | 480 | 0.9866 | 0.339 | 552 s |
| v2 (none) | 160 | **0.9875** | **0.347** | **222 s** |

---

## 5. Five-minute demo script

Before starting, open **Model Comparison** and press **Run all available models** once (warm-up).
Keep the Roboflow project open in a browser tab.

### 0:00 – 1:00 · Topic and dataset
* "Topic: anomaly detection and localisation with **EfficientAD**, with **PatchCore** as the baseline.
  Every model learns only from **good** products and flags whatever looks different."
* **Roboflow tab**: "We built the dataset from scratch: 200 bottle images, labelled normal or anomalous,
  split **80/10/10**. Train has **no defects**, by design."
* **Overview**: 200 images (160 / 20 / 20), **4 / 4 models**.
* Sidebar **Dataset → Roboflow v1**. Go to **Dataset Explorer**:
  * "Version 1 has the **7 augmentations**: flip, flop, 90° rotation, crop/zoom, brightness, exposure,
    blur. Training grows to 480 images. Here are the augmented copies."
  * Point at the **augmentation study** table: "It did not help (pixel AUROC 0.9866 vs 0.9875) and took
    2.5× longer, so **v2 is our final dataset**."
* Switch **Dataset → Roboflow v2** (back to 4/4 models).

### 1:00 – 2:30 · Live Inspection
1. Active model **EfficientAD-S**, image `broken_large/bottle_test_broken_large_002.png`, **Run**:
   * **DEFECTIVE**, score 0.263 vs threshold 0.202, ground truth **correct**, IoU ≈ 0.54–0.56.
2. Show *Anomaly heatmap*, *Predicted mask*, and *Ground-truth comparison* (yellow = found,
   red = false alarm, green = missed).
3. Switch to **PatchCore · ResNet18** and Run the same image: **DEFECTIVE** (3.17 vs 1.96), IoU 0.36.
   "PatchCore is confident, but its mask is much larger than the crack. EfficientAD's is tighter."
4. Before/after slider and synchronized zoom on the crack.

### 2:30 – 3:40 · Model Comparison
* **Run all available models**: four heatmaps next to the ground truth.
* Points to make:
  * "PatchCore needs no training, so it detects all 20 test images correctly."
  * "EfficientAD-S has the best localisation (IoU 0.51) after 6.6 min of GPU training."
  * "EfficientAD-M needed more iterations. It misses 5 defects."
  * "EfficientAD used a **reduced protocol**: 2,500 / 1,500 instead of 70,000 iterations. On this CPU the
    full protocol would take 32 h / 225 h, so we trained on a free Colab T4 GPU."

### 3:40 – 4:20 · Analytics
* Active model EfficientAD-S. Score distribution: the threshold comes from normal **validation**
  images only.
* The one miss is `contamination_004` (small and faint) and the one false alarm is `good_000`.

### 4:20 – 5:00 · Report
* **Reports → Generate PDF report → Download**. The PDF has the dataset (Roboflow, 80/10/10), the model's
  metrics, the 4-model comparison table, the inspection just run, and the limitations.
* "Ready-made reports for all models and both versions are in `results/reports/`."

---

## 6. If something goes wrong

| Symptom | Fix |
|---|---|
| Light (white) widgets | Start Streamlit from the project folder, or restart it |
| `AttributeError` after an update | Restart the app (helper modules are not hot-reloaded) |
| "Only … MB RAM available - not loading …" | Close other programs, then Run again (the app already unloaded the other models) |
| EfficientAD-M seems frozen | It takes ~6 s per image on CPU; wait for the spinner |
| Only 1/4 models available | The sidebar **Dataset** is on Roboflow v1; switch to **Roboflow v2** |
| Shows MVTec AD instead of Roboflow | `unset MVTEC_ROOT` and restart, or choose Roboflow v2 in the sidebar |
| Port 8501 busy | `~/miniconda3/bin/streamlit run app.py --server.port 8502` |
