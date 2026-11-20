# Viva Questions and Answers

**Industrial Anomaly Detection and Localization using EfficientAD (with PatchCore baseline) on MVTec AD**

All questions in one place. Questions 1–32 cover the theory and the PatchCore work (on the official
MVTec AD release). Questions 33–46 cover the final version: the Roboflow dataset (v1 and v2),
EfficientAD training on a GPU, and the four-model comparison. Questions 47–53 cover the final verdict
(which model is best and why) and the numbers shown in the presentation.

Background for each answer: [PROJECT_EXPLANATION.md](PROJECT_EXPLANATION.md) (sections 17–19 for the
final version). The slide-by-slide guide to the presentation is
[PRESENTATION_EXPLANATION.md](PRESENTATION_EXPLANATION.md).

**Key numbers to remember (Roboflow v2 test split, 20 images):**

| | EfficientAD-S | EfficientAD-M | PatchCore R18 | PatchCore WRN-50 |
|---|---|---|---|---|
| Correct (of 20) | 18 | 15 | 20 | 20 |
| Accuracy | 0.90 | 0.75 | 1.00 | 1.00 |
| Precision / Recall (image) | 0.90 / 0.90 | 1.00 / 0.50 | 1.00 / 1.00 | 1.00 / 1.00 |
| Image AUROC | 0.95 | 0.89 | 1.00 | 1.00 |
| Pixel AUROC | 0.981 | 0.941 | 0.988 | **0.992** |
| Pixel F1 | **0.679** | 0.589 | 0.515 | 0.545 |
| Pixel IoU | **0.514** | 0.417 | 0.347 | 0.375 |
| Mean latency (laptop CPU) | 479 ms | 5,744 ms | **60 ms** | 299 ms |
| Training | 2,500 it, T4, 6.6 min | 1,500 it, T4, 10.8 min | none (memory bank) | none (memory bank) |

Augmentation study (PatchCore R18): v1 (7 augmentations, 480 train images) pixel AUROC 0.9866;
v2 (none, 160 images) 0.9875.

**Final verdict:**
* **Best overall: PatchCore WRN-50.** 20/20 correct and the highest pixel AUROC (0.9924).
* **Best defect outlines: EfficientAD-S.** Pixel F1 0.679, IoU 0.514.
* **Fastest: PatchCore ResNet18.** 60 ms on CPU.
* **Weakest: EfficientAD-M.** 15/20 correct after only 1,500 iterations.

---

## Part A: Theory and PatchCore (official MVTec AD)

**1. What problem does your project solve?**
Automatic visual inspection: deciding whether a product image is defective and highlighting where
the defect is, using only defect-free images to build the model.

**2. Why not train a normal classifier (good vs. defective)?**
Defective examples are rare, expensive to collect and label, and new defect types appear over time.
A classifier only recognises the defect types it was trained on. Learning "normal" catches any
deviation.

**3. Is your method supervised or unsupervised?**
Unsupervised (one-class) for building the model: only normal images, no labels. Labels and masks
are used only to *evaluate* it.

**4. What is MVTec AD and why use it?**
The standard industrial anomaly detection benchmark: 15 categories, an official train/test split,
pixel-accurate masks. Using it makes results comparable and checkable.

**5. Which category did you use and how big is it?**
Bottle: 209 normal training images, 83 test images (20 normal, 63 defective: broken_large,
broken_small, contamination), 900×900 resolution.

**6. Explain PatchCore in one minute.**
Extract mid-level CNN features for every patch of the normal images and store them in a memory
bank, reduced with greedy coreset selection. For a test image, each patch's anomaly score is the
distance to its nearest neighbour in the bank; the map of these distances localises defects and its
maximum is the image score.

**7. Do you train ResNet18?**
No. We use torchvision's ImageNet weights and keep them frozen. Building the model only extracts
features and stores them.

**8. Why layer2 and layer3?**
layer1 is too low-level (edges, colour). layer4 is too coarse (7×7) and specialised to ImageNet
classes. layer2/layer3 describe local structure at a useful resolution and transfer well.

**9. What is the purpose of the 3×3 average pooling?**
Each patch vector then summarises its neighbourhood, which makes it robust to small shifts and
gives local context (the paper calls these "locally aware patch features").

**10. What is the memory bank and how large is yours?**
The set of normal patch vectors: 147,392 vectors from 188 images, reduced to 14,739 (10 %) by
coreset selection, each 384-dimensional.

**11. What is coreset subsampling and why is it greedy?**
Choosing a small subset that still covers the whole feature space. The greedy k-center rule always
adds the point farthest from the current subset, which guarantees the subset approximates every
original point within a bounded distance. Random sampling could miss rare normal appearances,
causing false alarms.

**12. Why project to 128 dimensions during coreset selection?**
To make the many distance computations cheaper. Random projections approximately preserve
distances (Johnson-Lindenstrauss), so the chosen points are almost the same.

**13. How is the anomaly score computed?**
For every test patch: the Euclidean distance to its nearest memory-bank vector. Image score = the
maximum over all patches.

**14. Why the maximum and not the mean?**
A defect usually occupies a small area. The mean would be dominated by the many normal patches and
hide it; the maximum reacts to the single most abnormal region.

**15. How do you get a pixel-level heatmap from a 28×28 grid?**
Bilinear upsampling to 224×224, Gaussian blur (σ = 4), then resize to the original 900×900 image.

**16. How did you choose the thresholds? Did you use the test set?**
No. 21 normal training images were held out from the memory bank. Image threshold = their maximum
score; pixel threshold = 99.5th percentile of their pixel scores. Test labels and masks were used
only to compute the final metrics.

**17. What is AUROC and why report it?**
The area under the ROC curve: the probability that a random defective sample scores higher than a
random normal one. It is threshold-independent, so it measures the scoring quality itself.

**18. Your image AUROC is 1.0. Does that mean the model is perfect?**
It means every defective test image scored higher than every normal one on this split. But
bottle is an easy category with only 83 test images, and at our fixed threshold one defective image
is still missed (recall 0.984). Harder categories would score lower.

**19. Pixel AUROC is 0.98 but pixel F1 is only 0.46. Why?**
AUROC looks at the ranking over all thresholds. F1 uses our single calibrated threshold, at which
the predicted masks are too large (precision 0.30, recall 1.00): the blur and the coarse 28×28 grid
spread the high scores around the defect. A stricter threshold would raise precision, but choosing
it from the test masks would be cheating.

**20. What does IoU 0.30 mean?**
On average the overlap between predicted and real defect pixels is 30 % of their union. The real
defect is almost entirely covered (recall ≈ 1.0), but the prediction also covers surrounding normal
glass.

**21. Precision vs. recall: which matters more in a factory?**
Usually recall (catching defects): a missed defect reaches the customer. Too many false alarms
(low precision) cost manual re-inspection time. The threshold sets the trade-off.

**22. What was your inference time?**
About 50 ms per image (median 48 ms, ≈ 20 images/s) on a laptop CPU with 6 threads. Building the
model takes ~4 minutes.

**23. How does your implementation differ from the original PatchCore?**
ResNet18 instead of WideResNet-50; no 1024-d per-layer projection; no score re-weighting; the full
image is resized to 224 instead of a 224 centre crop. So it is "PatchCore-style", not a reproduction.

**24. What happens if I upload a photo of a different object?**
It will very likely be called defective. The model only knows what a normal bottle looks like, so
everything else is far from the memory bank. One model per product type is needed.

**25. How would you improve the project?**
WideResNet-50 features; evaluate all 15 categories; compute pixel metrics and AU-PRO at full
resolution; FAISS for faster nearest-neighbour search; a larger calibration set; compare with
EfficientAD for speed/accuracy.

**26. How do you know the results are not fabricated?**
The metrics are computed by `evaluation.py` from the model's predictions on the official test
images and saved to `results/bottle/patchcore_r18/metrics.json` and `per_image.csv`. Rebuilding from scratch
reproduced identical thresholds and metrics, and the 17 automated tests re-check the pipeline.

---

**27. Why is EfficientAD shown as "unavailable"?**
It needs 70,000 training iterations per category: about 32 h (S) or 225 h (M) on this CPU. No
checkpoint exists, so the app shows the reason instead of numbers. Inventing or estimating its
results would be scientific misconduct.

**28. Is the comparison between the two PatchCore models fair?**
Yes, within its scope. Both use the same 83 images, the same evaluation code, the same pixel grid,
and thresholds from normal images only. One difference is stated openly: they use different random
validation splits (seed 0 vs. 42) and input sizes (224 vs. 256).

**29. What is the difference between IoU and Dice?**
Both measure mask overlap. IoU = |P∩G| / |P∪G|. Dice = 2|P∩G| / (|P|+|G|), which weighs the
overlap twice. Dice is always ≥ IoU (for broken_large/000: IoU 0.321, Dice 0.486).

**30. Why does the WRN-50 model localise better?**
A deeper, wider backbone gives richer patch features. It also uses a 256 px input (32×32 patch grid
instead of 28×28) and 1024-d embeddings, so boundaries are finer. The cost is ~6× more computation.

**31. What does the before/after slider show that a single overlay does not?**
The exact correspondence between the raw image and the model output at the same position. Dragging
the handle over a defect shows whether the hot region really sits on the visible damage.

**32. Why are ground-truth views disabled for uploaded images?**
Ground truth is a manual annotation that only exists for the official test images. Showing anything
there for an upload would mean presenting an invented annotation.

## Part B: Final version (Roboflow dataset, EfficientAD, four models)

**33. Why did you build the dataset on Roboflow instead of just using MVTec AD?**
The requirement was to build the dataset from scratch with our own split, preprocessing and
augmentation. We started from MVTec AD images, which keeps real ground-truth masks for evaluation, but
the selection, labelling, 80/10/10 split, preprocessing and augmentation were done by us on Roboflow.

**34. Why does the training split contain no defective images? Won't the model be unable to learn defects?**
It is not supposed to learn defects. Unsupervised anomaly detection learns what *normal* looks like and
flags deviations, so it can detect defect types it has never seen. A defect in training would become
part of "normal" and would then be missed. Defective images are used in validation and test.

**35. You applied 7 augmentations. Why does the app use the version without augmentation?**
We measured both. Augmentation slightly lowered pixel AUROC (0.9866 vs. 0.9875) and took 2.5× longer.
For anomaly detection, augmented normal images widen what counts as normal. A rotated or brightened
bottle becomes "normal", which can hide real defects. So we kept v1 as the experiment and use v2.

**36. Why is EfficientAD worse than PatchCore at detection in your results, when the paper says the opposite?**
We trained it for 2,500 (S) and 1,500 (M) iterations instead of 70,000, because that is what fits in
~10 minutes on a free GPU. PatchCore needs no training, so it is at full strength immediately. The
comparison shows the cost of training under a tight compute budget, not the methods' best possible
results.

**37. Then what is EfficientAD better at in your results?**
Localisation. EfficientAD-S has pixel IoU 0.51 and pixel precision 0.62. PatchCore has 0.35–0.38 for
both, because its masks are much larger than the defects.

**38. Why is EfficientAD-M so slow (5.7 s per image)?**
Latency was measured on the laptop CPU. EfficientAD is designed for GPU inference, where the paper
reports a few milliseconds per image. On a CPU, the medium-size networks are much slower.

**39. How do you know the GPU-trained checkpoints are really used and not made up?**
* The checkpoint files (`artifacts/roboflow_v2/main_models/<model>/bottle/checkpoint.ckpt`) and their
  metadata (iterations, device = cuda, training time) are on disk.
* The app loads them, and every number comes from running them on the 20 test images here.
* The Model Comparison page runs all four live on any image.

**40. You have two Roboflow versions. Which one does the website use?**
Both. The sidebar has a **Dataset** selector:
* **Roboflow v2** (no augmentation) is the default and final dataset. All four models were trained
  and evaluated on it.
* **Roboflow v1** (7 augmentations, 480 training images) holds the augmentation experiment. Only
  PatchCore ResNet18 was built on it, because the experiment showed augmentation does not help, so the
  expensive GPU models were trained on v2 only.

The Dataset Explorer shows the v1-vs-v2 comparison table on both.

**41. Why train EfficientAD only on v2 and not on v1 as well?**
The augmentation question was already answered with PatchCore: v1 was not better and took 2.5× longer.
GPU time was the bottleneck (about 10 minutes available). It was spent on the dataset version that
performed better and that Roboflow recommends for anomaly detection.

**42. What do your 80/10/10 splits contain exactly?**
* Train: 160 normal images.
* Validation: 10 normal + 10 defective. The normal ones set the thresholds.
* Test: 10 normal + 10 defective (3 broken_large, 4 broken_small, 3 contamination).

Validation and test use different images. No test image is used for training or thresholds.

**43. How are the thresholds chosen for EfficientAD?**
* EfficientAD's training code holds out 10 % of the normal training images (16 images).
* The image threshold is the highest anomaly score among them, so every held-out normal image passes.
* The pixel threshold is the 99.5th percentile of their anomaly-map pixels.

No defective image and no test image is used.

**44. Which model would you deploy?**
It depends on the requirement:
* **Detect defects on a CPU, now:** PatchCore ResNet18. It is perfect on our test images and takes 60 ms.
* **Defect outlines matter** (e.g. measuring defect size): EfficientAD-S. It has the best IoU (0.51),
  and on a GPU it reaches millisecond latency.
* With full training (70,000 iterations), EfficientAD should also improve at detection, as reported in
  the paper.

**45. What are the main limitations of your results?**
1. EfficientAD used a reduced protocol (4 % / 2 % of the paper's iterations).
2. Only 20 test images from one category (bottle), so differences of ±0.05 AUROC are not significant.
3. Latency was measured on a CPU, while EfficientAD is designed for a GPU.
4. PatchCore ResNet18 is a simplified PatchCore.

**46. Where can the examiner see the evidence?**
* Checkpoints and metadata (iterations, device, training time): `artifacts/roboflow_v2/main_models/`.
* Metrics and per-image scores: `results/roboflow_v2/bottle/<model>/`.
* PDF reports: `results/reports/`.
* The Colab notebook: `colab/train_efficientad_colab.ipynb`.
* Live: **Model Comparison → Run all available models** on any test image.

## Part C: Final verdict and the presentation numbers

**47. Which model performs best overall, and why?**
**PatchCore WRN-50.**
* It classifies all 20 test images correctly (image AUROC 1.00, accuracy 1.00), tied with PatchCore
  ResNet18.
* It has the highest pixel AUROC of all four models (0.9924), so it ranks defective pixels above normal
  ones best.

Why it wins:
1. PatchCore has no training loop. The memory bank is complete as soon as the normal features are
   stored, so the short compute budget did not weaken it.
2. WideResNet-50 gives richer features than ResNet18 (1024-d vs 384-d patch vectors).

**48. If PatchCore is best, why is EfficientAD your main model?**
* EfficientAD is the method the project set out to study. It is designed for millisecond GPU
  inference, and the paper reports it above PatchCore after full training.
* In our results it already gives the **best defect outlines**: EfficientAD-S has pixel F1 0.679 and
  IoU 0.514, against at most 0.545 and 0.375 for PatchCore.
* It was trained with a reduced protocol: 2,500 (S) and 1,500 (M) of the paper's 70,000 iterations. So
  the verdict is "best within our compute budget", not "the better method in general".

**49. Which model has the best pixel precision? Is it EfficientAD-S?**
No. **EfficientAD-M** has the highest pixel precision, 0.644 (EfficientAD-S has 0.618). But M's pixel
recall is only 0.543, so it misses much of each defect.

EfficientAD-S has the best balance of precision and recall: the highest pixel F1 (0.679) and IoU
(0.514). PatchCore has almost perfect pixel recall (0.9997) but low precision (0.35–0.38), because
its masks are much larger than the defects.

**50. What do accuracy 0.75 and recall 0.50 for EfficientAD-M mean?**
* M labels 15 of 20 test images correctly. It raises **no false alarms** (precision 1.00, all 10 normal
  bottles pass).
* It misses 5 of the 10 defects (recall 0.50).
* Its anomaly scores for those defects stayed below the threshold set on normal images. A larger network
  needs more iterations to separate the classes.

**51. The presentation shows IoU 0.556 for one image, but the table says EfficientAD-S has IoU 0.514. Which is right?**
Both are right; they measure different things.
* **0.514** is the pixel IoU over **all 20 test images**, computed on a 256×256 grid by the evaluation
  code.
* **0.556** is the IoU of **one image** (`broken_large_002`), computed by the website at the image's
  full 224×224 resolution.
* The evaluation code gives 0.545 for that same image, because of its 256×256 grid.

**52. How did you make sure the numbers in the presentation are real?**
* The presentation is generated by `make_presentation.py`. It reads every number directly from the
  saved result files: `results/roboflow_v2/bottle/<model>/metrics.json`, checkpoint metadata
  (iterations, loss history, training time) and the dataset folders.
* The verdict slide computes the winner of each criterion from those files. The script stops if the
  data no longer supports the verdict.
* The charts are drawn from the same values.

**53. Is the 20-image test set big enough to call a winner?**
Only with care. One image changes image AUROC by up to 0.05–0.10. So the clear findings are:
* The two PatchCore models detect perfectly.
* EfficientAD-S outlines defects best.
* EfficientAD-M needs more training.

The small gap between WRN-50 and ResNet18 in pixel AUROC (0.9924 vs 0.9875) is not significant on
this test set.

