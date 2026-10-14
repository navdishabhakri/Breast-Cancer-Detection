# Breast Cancer Diagnosis: Recall-First Classification

**Predicting whether a breast mass is malignant or benign from 30 diagnostic measurements, with an evaluation strategy built around one fact: a missed cancer costs far more than a false alarm.**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

> Course project for **SEA600** (Milestones 1 to 3). Group members: <!-- add names -->
> Educational work only. This is not a medical device and must not be used for real diagnosis.

---

## The problem

Given 30 features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass, classify the tumor as **malignant (1)** or **benign (0)**.

This is a supervised binary classification task with an asymmetric cost:

| Error | Meaning | Consequence |
|---|---|---|
| **False negative** | Cancer predicted as benign | Potentially life-threatening |
| **False positive** | Benign predicted as cancer | Extra testing, no direct harm |

So the project optimizes and reports **recall (sensitivity)** first, then precision, F1 and ROC-AUC. Accuracy alone is treated as misleading.

---

## Dataset

[Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic): 569 samples, 30 numeric features (mean, standard error and "worst" values of radius, texture, perimeter, area, smoothness, and so on), and a diagnosis label.

**Cleaning:** dropped the `id` column (no predictive value) and the empty `Unnamed: 32` column created by trailing commas in the CSV; encoded `diagnosis` as M = 1, B = 0. No missing values or duplicates needed handling.

**EDA findings that shaped the pipeline**
- Features differ enormously in scale (`area_mean` > 1000, `smoothness_mean` < 0.1), and most are right-skewed, so **standardization is required** for KNN, logistic regression and SVM.
- The correlation heatmap shows heavy multicollinearity (`radius_mean`, `perimeter_mean` and `area_mean` are nearly perfectly correlated), which motivated the feature engineering study.
- Benign cases outnumber malignant ones, so every split is **stratified**.

---

## Methodology

**Data hygiene**
- 80/20 stratified train/test split (`random_state=42`); the **test set is locked** and not used for training, tuning or model selection until the final evaluation.
- Model comparison uses **stratified 5-fold cross-validation** on the training set only.
- Scaling lives **inside sklearn `Pipeline`s**, so it is fit on each training fold only and never leaks validation data.
- Correlation filtering, PCA and SelectKBest are all fit on training data only.

**Evaluation toolkit:** accuracy, precision, recall, F1, ROC-AUC, ROC curves, precision-recall curves, and confusion matrices built from cross-validated predictions, plus resource cost (training time, per-sample inference time, serialized model size).

---

## Milestone 1: Baselines

| Model | Notes |
|---|---|
| **Logistic Regression** | Un-regularized (`penalty=None`), as a check on linear separability |
| **K-Nearest Neighbors** | Distance-based, non-parametric |

Both ran in scaled pipelines. Logistic regression holds higher precision than KNN at the same recall on the PR curve. KNN trains instantly but is slower and larger at inference because it stores the training set.

## Milestone 2: Advanced models

| Model | Notes |
|---|---|
| **Random Forest** | 100 bagged trees, default hyperparameters |
| **SVM (RBF kernel)** | Default hyperparameters, `probability=True` |

On cross-validation, **SVM was the strongest model**: CV accuracy 97.14%, F1 0.961, ROC-AUC 0.9949, and only 7 false negatives in out-of-fold predictions versus 14 for KNN. Random Forest was the most expensive to train and store.

## Milestone 3: Feature engineering, tuning and final evaluation

### Feature engineering study (applied to SVM)

| Approach | Result |
|---|---|
| Manual: drop features with correlation above 0.95 (30 to 23 features) | Recall maintained; accuracy and F1 slightly lower |
| PCA to 95% explained variance (10 components) | Matched the original 30 features, confirming heavy redundancy |
| SelectKBest, ANOVA F-test, k = 15 | **Worst:** recall fell from 0.9471 to 0.8941 |

The team's interpretation is that the ANOVA F-test scores features independently and discards combinations the RBF kernel can exploit. The original 30 features were kept for tuning.

### Hyperparameter tuning

`GridSearchCV` over `C` in {0.1, 1, 10, 50, 100} and `gamma` in {scale, auto, 0.01, 0.1, 1}, **scored on recall** with the same 5-fold stratified CV.

- **Best parameters: `C = 10`, `gamma = 'scale'`**
- CV recall improved from **0.9471 to 0.9647**
- `gamma = 1` collapsed the model (recall near zero), a clear example of an over-tight kernel memorizing individual points

### Final test-set evaluation (114 held-out samples, 42 malignant)

| Model | Test recall | Missed cancers (FN) |
|---|---|---|
| Logistic Regression | 88.10% | 5 |
| KNN | 90.48% | 4 |
| Random Forest | 92.86% | 3 |
| SVM (RBF), default | 92.86% | 3 |
| **SVM (RBF), tuned** | **92.86%** | **3** |

Random Forest, default SVM and tuned SVM all reached **97.37% accuracy and 100% precision** on the test set.

### Resource comparison

| Model | Train time | Inference per sample | Serialized size |
|---|---|---|---|
| Logistic Regression | fastest to deploy | fastest | 2.97 KB |
| Random Forest | 0.73 s | 0.00019 s | 320.49 KB |
| SVM (RBF), default | 0.068 s | 0.00009 s | 30.65 KB |
| SVM (RBF), tuned | n/a | near-instant | 24.29 KB |

---

## Recommendation

**Tuned SVM (RBF), `C = 10`, `gamma = 'scale'`.** Three models tied on the test set, so the choice rests on the other evidence:

1. Highest cross-validated recall (0.9647), a sturdier estimate than one 114-sample split
2. About 13x smaller than the Random Forest (24.29 KB vs 320.49 KB)
3. Fast inference suited to real-time decision support
4. No false alarms and only 3 missed cancers on the test set

---

## Limitations and next steps

Being direct about what this project does not show:

- **The decision threshold was never actually lowered.** The write-up argues for tuning the threshold to favor recall, but all reported results use the default 0.5. With 3 missed cancers still on the test set, threshold selection from the cross-validated precision-recall curve is the most obvious next step.
- **Test set is small.** With 42 malignant cases, the difference between 3 and 4 missed cancers is noise; confidence intervals or repeated stratified splits would give a fairer comparison.
- **Tuning and scoring used the same CV folds.** The 0.9647 CV recall is therefore slightly optimistic. Nested cross-validation would give an unbiased estimate.
- **Single dataset, no external validation.** Real deployment would need data from other institutions and calibration checks.
- **Interpretability.** Random Forest feature importances and SHAP values were planned but not completed; both matter for clinical trust.

---

## Run it yourself

```bash
git clone <your-repo-url>
cd <repo-name>
pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter
jupyter notebook
```

Download `data.csv` from the dataset link above and update the path in the first code cell (the notebook was written in Google Colab and reads `/content/data.csv`). Run the notebook top to bottom; each milestone builds on the objects defined in the previous one.

---

## Repository contents

```
├── <notebook-name>.ipynb   # full analysis: Milestones 1 to 3
├── data.csv                # Breast Cancer Wisconsin (Diagnostic)
└── README.md
```
