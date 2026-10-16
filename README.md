# AI301 Assignment 2: Cats and Dogs++

Multi-label image classification (Kaggle In-Class competition "Cats and Dogs++").
Each photo gets five 0/1 labels: `Cat`, `Dog` and three named pets, `Moris`, `Motya`, `Biatrix`. The target metric is **macro F1**.

| Notebook | Content |
|---|---|
| [01_eda.ipynb](notebooks/01_eda.ipynb) | Data checks, label combinations, image sizes |
| [02_metric.ipynb](notebooks/02_metric.ipynb) | Macro F1 analysis, thresholds, PR AUC |
| [03_validation.ipynb](notebooks/03_validation.ipynb) | Near copies, adversarial validation, folds, first submissions |

The fold assignment is saved in [folds.csv](folds.csv), so every model uses the same splits.

---

## 1. EDA

**Data quality.** Train has 12,260 images and test has 18,673. The CSV files and image folders match exactly: no missing files, no extra files, no duplicate IDs. Labels are only 0 and 1, with no NaN.

**Labels.** Moris and Motya are always cats, and Biatrix is always a dog. That leaves only 8 label combinations in train:

| Combination | Photos |
|---|---|
| Cat | 4989 |
| Dog | 4984 |
| No labels | 2162 |
| Cat + Moris | 42 |
| Dog + Biatrix | 37 |
| Cat + Motya | 32 |
| Cat + Moris + Motya | 10 |
| Cat + Dog + Moris + Biatrix | 4 |

**Class imbalance.** Cat is 41.4% of train, Dog is 41.0%, and 17.6% of the photos have no label. The named pets are very rare: Moris 56 photos (0.46%), Motya 42 (0.34%), Biatrix 41 (0.33%). Since the metric is macro F1, these three rare labels make up 60% of the score, so the imbalance is the main difficulty of the task.

**Image size.** Every image in train and test is 450x375, and the size does not depend on the label. So the model cannot use size as a shortcut, and we don't need to resize anything to make the sizes equal.

**Look at the photos.** The photos with no label contain easy negatives (lobster, turkey, horses) and hard ones (wombat and other furry animals), which can trigger false Cat or Dog predictions. Generic cats and dogs look like web photos: many breeds, a large animal in the center. Moris, Motya and Biatrix look like phone photos from one home and from walks, and the pet is often small in the frame. Two risks follow from this:
- the model may learn the background (home, street) instead of the pet;
- strong cropping can cut the small pet out of the image, so augmentation should be gentle.

---

## 2. Metric analysis

### Why macro F1 fits this task
Accuracy or micro-averaged F1 would be dominated by Cat and Dog. We checked this with trivial predictions on train:

| Prediction | Cat | Dog | Moris | Motya | Biatrix | Macro F1 |
|---|---|---|---|---|---|---|
| all ones | 0.586 | 0.581 | 0.009 | 0.007 | 0.007 | 0.238 |
| perfect | 1 | 1 | 1 | 1 | 1 | 1.0 |
| perfect Cat/Dog, zeros for named pets | 1 | 1 | 0 | 0 | 0 | 0.4 |

Both perfect cases match a hand calculation, so our implementation is correct. The last row reveals the key point: a model that is perfect on Cat and Dog but ignores the named pets gets only 0.4. So we need to pay a lot of attention to minor classes.

### Weaknesses and edge cases
- **Very noisy for rare labels.** With about 40-60 photos per named pet, one more missed photo lowers the macro F1 by about 0.002-0.003 (a miss in Cat or Dog costs only 0.00002). We call two macro F1 scores *similar* if they differ by less than **0.005**, which is about 2 photos of a named pet. A smaller gain is not a real gain.
- **Depends on the threshold.** With the same model, moving the threshold changed macro F1 by 0.014 (see below). Without a fixed threshold, a better threshold can look like a better model.
- **Undefined F1.** If a label has no true and no predicted photos, F1 is 0/0 and sklearn returns 0, so even a perfect answer loses 0.2. It cannot happen in our folds (8-12 photos of each named pet per fold), but it happens in small batches. So we compute F1 only on the full validation set, never per batch.
- **Labels are scored separately.** Macro F1 does not see the rows, so impossible combinations (e.g. Biatrix = 1 but Dog = 0) are not punished. We tried the rule "a predicted named pet sets Cat/Dog to 1". It changed 3 photos of 12,260, gave +0.00002 macro F1 (far below 0.005), so we do not use it.

### Baseline used for the analysis
To study the metric we need some predictions, so we took a pretrained `convnext_base.fb_in22k_ft_in1k_384` (ImageNet only) as a frozen feature extractor (1024 numbers per image), and trained one logistic regression per label. All scores below are out-of-fold, using the folds from section 3.

| Threshold | Cat | Dog | Moris | Motya | Biatrix | Macro F1 |
|---|---|---|---|---|---|---|
| 0.5 | 0.997 | 0.997 | 0.869 | 0.853 | 1.000 | 0.943 |
| 0.25 | 0.997 | 0.996 | 0.917 | 0.914 | 0.976 | 0.960 |

Almost all of the loss comes from Moris and Motya (at 0.5 they miss 13 of 56 and 10 of 42 photos). The lower threshold helps because the named pets are rare and the model is shy to predict them.

Picking 0.25 on the same data we score is optimistic, so we also did a fair check: for each fold, take the best threshold from the other 4 folds and apply it only to the held-out fold. Macro F1 is then **0.957** (the thresholds chosen were 0.25 in four folds and 0.30 in one, so 0.25 is stable). We use **0.25**.

### Complementary metrics
- **PR AUC (per label, macro mean)**: no threshold, and it ignores the huge number of easy negatives.
- **Per-label F1**: shows *which* label is failing, since macro F1 hides it.
- **ROC AUC**: we checked it and do not use it. It is ≥ 0.999 for every label (even Moris beats Dog), because the false positive rate divides by about 12,000 negatives, so a few false alarms cost almost nothing.

| Metric | Cat | Dog | Moris | Motya | Biatrix | Macro |
|---|---|---|---|---|---|---|
| F1 at 0.25 | 0.997 | 0.996 | 0.917 | 0.914 | 0.976 | 0.960 |
| PR AUC | 1.000 | 1.000 | 0.981 | 0.977 | 1.000 | 0.991 |
| ROC AUC | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

PR AUC also tells us *why* we lose points. Biatrix has PR AUC 1.0 but F1 0.976: the ranking is perfect, so only the threshold is to blame. Moris has PR AUC 0.981: some wrong photos rank above real Moris photos, so only a better model can help.

### Proposed alternative target metric
**Macro PR AUC.** It keeps the 20% weight for each label (so the rare pets count), needs no threshold, and is not inflated by easy negatives. It lets us compare models without mixing in threshold tuning. We keep macro F1 as the official metric and report macro PR AUC next to it.

---

## 3. Validation

### Strategy
**Stratified 5-fold split** over the 8 label combinations (`StratifiedKFold`, `shuffle=True`, `random_state=42`), saved in `folds.csv`. Stratifying by the *combination* (not by each label) puts the rare pets evenly into every fold:

| Combination | f0 | f1 | f2 | f3 | f4 |
|---|---|---|---|---|---|
| Cat / Dog | ~998 | ~997 | ~998 | ~998 | ~998 |
| Cat + Moris | 8 | 8 | 9 | 9 | 8 |
| Cat + Motya | 7 | 6 | 6 | 6 | 7 |
| Dog + Biatrix | 8 | 8 | 7 | 7 | 7 |

Why this choice: images have no time or group information. EXIF is stripped in all 30,933 photos (no capture time), and there is no owner or series ID. So there is nothing to split by, and a stratified split is the natural option. The test set is a random-looking sample of the same data (see below), so this should correlate well with the leaderboard.

### Near-duplicates
Photos of the named pets may come in series, which would cause leakage between folds. With no EXIF time, we looked for them using the cosine similarity of the ConvNeXt embeddings. By eye, pairs of named-pet photos at cosine ≥ 0.95 show the same scene (this does not hold for all train photos, so we use it only for pets).

Only 22 of 125 named-pet photos have such a near copy, and 15 of them (68%) are in test, 7 (32%) in train. Test is 60% of all photos, so near copies are spread about as expected by chance. A validation fold with near copies in train therefore looks like test, so we **do not group near copies** into one fold. This is not proven; with 22 photos, one photo moves the percentage by about 5 points.

### Adversarial validation
We trained a logistic regression (same model type as the main baseline) on the embeddings to tell train (0) from test (1), with 5 stratified folds. The ROC AUC is **0.501** (folds between 0.494 and 0.509), so the model cannot separate train from test. No distribution shift is visible, and we keep the same split with no extra action.

### CV vs leaderboard

| Model | Threshold | CV macro F1 | Public LB |
|---|---|---|---|
| ConvNeXt-base embeddings + LogReg | 0.25 | 0.957 (fair check; 0.960 optimistic) | 0.92922 |
| ConvNeXt-base embeddings + LogReg | 0.5 | 0.943 | 0.92186 |

The leaderboard is lower than CV by about 0.03 (at 0.25) and 0.02 (at 0.5). The good news: the order of the two thresholds is the same on CV and LB, so the CV choice holds on unseen data.

**Why the gap?** One explanation (not confirmed) is that test has more named-pet photos than train. The model predicts about 2-3 times more named pets in test than the train share suggests:

| Label | Expected from train share | Predicted at 0.25 |
|---|---|---|
| Cat | 7733 | 7771 |
| Dog | 7653 | 7681 |
| Moris | 85 | 169 |
| Motya | 64 | 115 |
| Biatrix | 62 | 184 |

If most of these were false alarms, the LB score would be much lower than 0.92, so many of them are probably real pets. Since the model learned that these pets are very rare, it may miss some of them. With so few photos per pet, a gap of 0.02-0.03 is also not far from the noise level of the metric (1 photo ≈ 0.0025).

**Decision.** Compare models by CV macro F1 (and macro PR AUC), treat differences below 0.005 as noise, and expect the LB to be about 0.02-0.03 lower than CV. Use threshold 0.25 for now.

---

## AI tools

Claude Code was used to help with structuring and organizing the report. The conversation transcripts are in [transcripts/](transcripts/).
