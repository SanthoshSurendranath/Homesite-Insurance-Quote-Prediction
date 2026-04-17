# Homesite Insurance Quote Conversion Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange?logo=scikit-learn) ![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-red) ![vecstack](https://img.shields.io/badge/vecstack-0.4.0-purple) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

This project builds a binary classification pipeline to predict whether a customer will convert an insurance quote into a purchased policy (`QuoteConversion_Flag`). Using a high-dimensional dataset of 595 anonymized coverage, geographic, personal, property, and sales features, the pipeline addresses significant class imbalance through **SMOTE oversampling** and evaluates multiple classifiers individually and through **ensemble stacking** to maximize predictive performance.

The dataset originates from the **Homesite Insurance** Kaggle competition and represents a real-world scenario where identifying likely purchasers enables targeted sales and underwriting strategies.

---

## Business Objective

Enable Homesite Insurance to identify prospective customers most likely to convert a quoted policy into a purchase, supporting:

- Prioritized sales team follow-up on high-conversion leads
- Optimized marketing spend by reducing outreach to low-probability customers
- Improved quote-to-policy conversion rates across customer segments

---

## Dataset

| Dataset | Records | Features | Target | Source |
|---|---|---|---|---|
| `RevisedHomesiteTrain1.csv` | 65,000 | 596 | `QuoteConversion_Flag` (0/1) | Google Drive |
| `RevisedHomesiteTest1.csv` | 173,836 | 595 | — (prediction target) | Google Drive |

### Feature Categories

The dataset contains 595 pre-processed and one-hot encoded features across five domains:

| Category | Example Features | Approx. Count |
|---|---|---|
| Coverage Fields | `CoverageField1A/B` through `CoverageField11A/B` | ~30 |
| Geographic Fields | `GeographicField1A/B` through `GeographicField64` | ~140 |
| Personal Fields | `PersonalField1` through `PersonalField83`, encoded variants | ~260 |
| Property Fields | `PropertyField1A/B` through `PropertyField39A/B` | ~80 |
| Sales Fields | `SalesField1A/B` through `SalesField15` | ~30 |

> **Class Imbalance Note:** The original training dataset is significantly imbalanced toward non-conversion (0), requiring SMOTE resampling before modelling to prevent classifier bias toward the majority class.

---

## Project Structure

```
HomesiteInsuranceQuotePrediction/
│
├── Homesite-Insurance-Quote-PredictionPart B.ipynb   # Full modelling pipeline
└── README.md
```

---

## Methodology

### 1. Data Loading
- Train and test datasets loaded from Google Drive via Google Colab
- Target column (`QuoteConversion_Flag`) separated from features
- Test set prepared without target for final predictions

### 2. Class Imbalance Handling — SMOTE
- **Synthetic Minority Oversampling Technique (SMOTE)** applied with `sampling_strategy=0.95`
- Resamples the minority class (converted quotes) to near-parity with the majority class
- All modelling performed on the resampled dataset to ensure fair class representation

### 3. Individual Classifiers
The following models were trained on the SMOTE-resampled data:

| Model | Purpose |
|---|---|
| Decision Tree Classifier | Interpretable baseline classifier |
| Random Forest Classifier | Ensemble of decision trees |
| K-Nearest Neighbours (KNN) | Instance-based learner |
| Multi-Layer Perceptron (MLP) | Neural network classifier |
| Gradient Boosting Classifier | Sequential boosting ensemble |

### 4. Ensemble Stacking (vecstack)
- **Base learners:** KNeighborsClassifier, MLPClassifier, RandomForestClassifier, DecisionTreeClassifier
- **Meta-learner:** GradientBoostingClassifier
- **Strategy:** Out-of-fold predictions (`oof_pred_bag`), 4-fold stratified cross-validation, shuffled with `random_state=0`
- Stacked feature matrix passed to GBC meta-learner for final classification

### 5. Prediction Output
- Final predictions generated for all 173,836 test records
- Outputs saved as CSV files to Google Drive, keyed by `QuoteNumber`
- Prediction probabilities extracted via `predict_proba` for confidence scoring

---

## Results

### Decision Tree on SMOTE-Resampled Data (50/50 test split)

| Metric | Class 0 (No Conversion) | Class 1 (Converted) | Overall |
|---|---|---|---|
| Precision | 0.92 | 0.91 | — |
| Recall | 0.91 | 0.92 | — |
| F1-Score | 0.92 | 0.91 | — |
| **Accuracy** | — | — | **91.6%** |

Confusion Matrix: [[24,184 / 2,257] / [2,075 / 22,904]]

### Ensemble Stacking — Cross-Validation Accuracy (4-Fold)

| Base Model | Mean CV Accuracy | Std Dev |
|---|---|---|
| KNeighborsClassifier | 72.1% | ±0.06% |
| MLPClassifier | 79.8% | ±6.45% |
| **RandomForestClassifier** | **93.9%** | ±0.07% |
| DecisionTreeClassifier | 91.9% | ±0.09% |

> Random Forest was the strongest and most stable base learner. The GBC meta-learner achieved 100% accuracy on the stacked holdout set.

---

## Output Files

| File | Model | Description |
|---|---|---|
| `your_output_file1.csv` | Decision Tree (SMOTE) | DT predictions on full test set |
| `MLP1.csv` | MLP Classifier | Neural network predictions |
| `KNN.csv` | KNN Classifier | Nearest-neighbour predictions |
| `GBC.csv` | Gradient Boosting | GBC predictions on test set |

All output files contain `QuoteNumber` and `QuoteConversion_Flag` columns.

---

## Technologies & Libraries

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Classification models, metrics, cross-validation |
| `imbalanced-learn` | SMOTE oversampling for class imbalance |
| `vecstack` | Model stacking utility |
| `mlxtend` | StackingClassifier support |
| `Google Colab` | Cloud-based execution environment |
| `Google Drive` | Dataset storage and output export |

---

## Setup & Usage

### Prerequisites

```bash
pip install scikit-learn imbalanced-learn vecstack mlxtend pandas numpy
```

### Running the Notebook

1. Mount Google Drive in Colab:
   ```python
   from google.colab import drive
   drive.mount('/gdrive')
   ```

2. Place the following files in your Google Drive root:
   - `RevisedHomesiteTrain1.csv`
   - `RevisedHomesiteTest1.csv`

3. Run all cells in `Homesite-Insurance-Quote-PredictionPart B.ipynb` sequentially.

4. Output prediction CSV files will be saved automatically to Google Drive.

---

## Key Findings

- **SMOTE effectively resolved class imbalance**, enabling classifiers to achieve balanced precision and recall across both classes rather than defaulting to majority-class predictions.
- **Random Forest was the strongest and most stable base learner** in the stacking ensemble, achieving ~93.9% cross-validation accuracy with very low variance (±0.07%).
- **MLP showed high variance across folds** (~6.45% std dev), indicating sensitivity to initialization and the high-dimensional sparse feature space — further tuning recommended.
- **Gradient Boosting as meta-learner** yielded the best final ensemble performance, demonstrating that combining diverse base learners via stacking improves over any single model.
- The **595-feature space** is highly sparse post one-hot encoding; dimensionality reduction (PCA, feature selection) is a recommended next step to improve generalisation and reduce training time on production deployments.

---

## Author

Santhosh Surendranath \
Data Scientist


---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
