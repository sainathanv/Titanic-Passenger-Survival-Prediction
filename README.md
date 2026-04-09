# Titanic Passenger Survival Prediction
### A Classical Machine Learning Classification Project

---

## Overview

This project builds and compares **five classical machine learning classifiers** to predict whether a passenger survived a ship disaster, using demographic and ticket information. It is a **binary classification** problem — the model predicts either `0` (did not survive) or `1` (survived).

The project covers the complete ML pipeline: data loading, exploratory data analysis, preprocessing, feature engineering, model training, hyperparameter tuning, evaluation, and cross-validation.

---

## Problem Statement

Given passenger attributes such as age, gender, ticket class, and fare, predict survival outcome.

| Property | Detail |
|---|---|
| Task | Binary Classification |
| Target variable | `Survived` (0 = did not survive, 1 = survived) |
| Dataset size | 891 passengers, 12 columns |
| Class balance | ~38% survived, ~62% did not (mildly imbalanced) |

---

## Dataset

The dataset file is `passengers.csv` with the following columns:

| Column | Type | Description |
|---|---|---|
| `PassengerId` | int | Unique row identifier (dropped) |
| `Pclass` | int | Ticket class: 1 = First, 2 = Second, 3 = Third |
| `Name` | string | Passenger name (dropped) |
| `Sex` | string | Gender: male / female |
| `Age` | float | Age in years (~177 missing values) |
| `SibSp` | int | Number of siblings/spouses aboard |
| `Parch` | int | Number of parents/children aboard |
| `Ticket` | string | Ticket number (dropped) |
| `Fare` | float | Ticket fare paid |
| `Cabin` | string | Cabin number (~687 missing, converted to binary) |
| `Embarked` | string | Port of embarkation: C / Q / S (2 missing) |
| `Survived` | int | **Target**: 0 = did not survive, 1 = survived |

---

## Project Structure

```
Titanic Passenger Survival Prediction/
│
├── 603_ml_project.ipynb     ← Main Jupyter notebook (all code + answers)
├── passengers.csv           ← Dataset (place here before running)
└── README.md                ← This file
```

---

## Models Implemented

| # | Model | Key Hyperparameter Tuned |
|---|---|---|
| 1 | Logistic Regression | Regularisation strength `C` |
| 2 | K-Nearest Neighbours (KNN) | Number of neighbours `k` |
| 3 | Support Vector Machine (SVM) | Kernel type + `C` + `gamma` |
| 4 | Decision Tree | `max_depth` + `min_samples_split` |
| 5 | Random Forest | `n_estimators` + `max_depth` + `max_features` |

---

## Workflow

```
Load Data
    ↓
Exploratory Data Analysis (EDA)
    ↓
Data Cleaning & Feature Engineering
    ↓
Build Preprocessing Pipeline
    ↓
Train / Validation / Test Split  (60% / 20% / 20%)
    ↓
For each of 5 models:
    Tune hyperparameters on Validation set
        ↓
    Refit best config on Train set
        ↓
    Evaluate ONCE on Test set
    ↓
5-Fold Cross-Validation on best model
    ↓
Final Comparison Table + ROC Curves
```

---

## Feature Engineering

Three new features are created before modelling:

| Feature | Formula | Rationale |
|---|---|---|
| `HasCabin` | 1 if Cabin is known, else 0 | Cabin knowledge correlates with higher class |
| `FamilySize` | `SibSp + Parch + 1` | Total family size including the passenger |
| `IsAlone` | 1 if `FamilySize == 1` | Solo travellers may have lower survival odds |

**Columns dropped:** `PassengerId`, `Name`, `Ticket`, `Cabin`

**Missing value handling:**
- `Age` — filled with **median per (Sex, Pclass) group** for maximum accuracy
- `Embarked` — filled with the **mode** (Southampton, 'S') since only 2 rows are affected

---

## Preprocessing Pipeline

A `scikit-learn` `Pipeline` wraps a `ColumnTransformer` that applies:

- `StandardScaler` → all numeric features (critical for KNN and SVM)
- `OneHotEncoder(handle_unknown='ignore')` → all categorical features (`Sex`, `Embarked`)

Using a Pipeline prevents **data leakage** — the scaler is fitted only on training data and never sees validation or test data in advance.

---

## Evaluation Metrics

Each model is evaluated on the test set using five metrics:

| Metric | What it measures |
|---|---|
| Accuracy | Overall percentage of correct predictions |
| Precision | Of passengers predicted to survive, how many actually did |
| Recall | Of passengers who actually survived, how many did we catch |
| F1-Score | Harmonic mean of Precision and Recall (best for imbalanced data) |
| ROC AUC | How well the model ranks survivors above non-survivors |

---

## Data Split

```python
# Step 1: Hold out 20% as final test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Step 2: Split remaining 80% into 75% train / 25% validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

# Final proportions: ~60% train / 20% validation / 20% test
```

The test set is used **exactly once per model** — never for hyperparameter tuning.

---

## How to Run

### Option A — Google Colab (recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `603_ml_project.ipynb` via **File → Upload notebook**
3. Run the first cell — it will show a file picker to upload `passengers.csv`
4. Go to **Runtime → Run all**

### Option B — VS Code / Local Python

1. Clone or download this repository
2. Place `passengers.csv` in the same folder as the notebook
3. Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

4. Replace the Colab upload cell with this local loading code:

```python
import os

notebook_dir = r"D:\Your\Project\Folder\Path"   # ← change this to your folder path
csv_path = os.path.join(notebook_dir, 'passengers.csv')
df = pd.read_csv(csv_path)

print("Dataset loaded successfully!")
print("Shape:", df.shape)
```

5. Open the notebook in VS Code and run all cells

---

## Requirements

| Library | Purpose |
|---|---|
| `numpy` | Numerical array operations |
| `pandas` | Data loading and manipulation |
| `matplotlib` | Base plotting library |
| `seaborn` | Statistical visualisations |
| `scikit-learn` | All ML models, pipelines, and metrics |

Install all at once:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Key Results (Typical Output)

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.81 | ~0.79 | ~0.74 | ~0.76 | ~0.87 |
| KNN | ~0.79 | ~0.76 | ~0.71 | ~0.73 | ~0.83 |
| SVM | ~0.82 | ~0.80 | ~0.75 | ~0.77 | ~0.88 |
| Decision Tree | ~0.78 | ~0.74 | ~0.72 | ~0.73 | ~0.78 |
| Random Forest | ~0.83 | ~0.81 | ~0.76 | ~0.78 | ~0.89 |

> Note: Exact values depend on hyperparameter tuning results on your specific data split.

---

## Key Findings

- **Gender** is the strongest single predictor — female passengers survived at ~74% vs ~19% for males
- **Passenger class** is the second strongest signal — 1st class: ~63%, 2nd: ~47%, 3rd: ~24%
- **Fare** is positively correlated with survival — higher fare = higher class = more survival
- **Random Forest** consistently achieves the best F1-Score and ROC AUC
- **Logistic Regression** is the best choice when interpretability matters — its coefficients directly explain each feature's effect on survival probability

---

## Report Questions Answered

The notebook contains inline Markdown answers to all **43 report questions** (Q1–Q43) covering data understanding, EDA, each model's tuning process, cross-validation analysis, and final reflection on limitations and improvements.

---

## Limitations

- Small dataset (891 samples) limits generalisation
- Simple imputation for Age (group median) — more sophisticated methods exist
- Class imbalance not explicitly addressed (no SMOTE or class weighting)
- Limited hyperparameter search space — GridSearchCV would be more exhaustive

---

## Possible Improvements

- Extract passenger titles from the `Name` column (Mr, Mrs, Master, Miss) as a feature
- Use `GridSearchCV` for more systematic hyperparameter tuning
- Apply SMOTE or `class_weight='balanced'` to handle class imbalance
- Try Gradient Boosting models (XGBoost, LightGBM) which typically outperform Random Forest on tabular data
- Bin `Age` into meaningful groups (child / adult / elderly)

---

## Author

Sainathan V
