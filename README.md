# Credit Risk Detection

Predicting whether a loan will default, using an EDA notebook and five classification models, all built as scikit-learn Pipelines.

## 1. Problem

Lenders lose money when borrowers default. The goal is to predict, from a borrower's profile and loan details, whether a loan will **default** (`loan_status`: 0 = repaid, 1 = default), so that risky applications can be flagged early. This is a binary classification problem with imbalanced classes (about 22% defaults), so recall, F1 and ROC AUC matter more than accuracy.

## 2. Data

- **Source:** Credit Risk Dataset (`credit_risk_dataset.csv`), Kaggle
- **Size:** 32,581 rows and 12 columns (32,416 after cleaning)
- **Target:** `loan_status`, imbalanced at roughly 22% defaults

| Column | Description |
|---|---|
| `person_age` | Borrower age in years |
| `person_income` | Annual income |
| `person_home_ownership` | RENT, OWN, MORTGAGE or OTHER |
| `person_emp_length` | Years employed |
| `loan_intent` | Purpose of the loan |
| `loan_grade` | Lender risk grade, A (safest) to G (riskiest) |
| `loan_amnt` | Loan amount |
| `loan_int_rate` | Interest rate (%) |
| `loan_percent_income` | Loan amount divided by income |
| `cb_person_default_on_file` | Past default on credit bureau record (Y/N) |
| `cb_person_cred_hist_length` | Credit history length in years |
| `loan_status` | **Target**: 1 = default |

## 3. Approach

### 3.1 Exploratory data analysis
- Inspected shape, dtypes, summary statistics and unique values.
- Removed **165 duplicate rows**.
- Plotted a correlation heatmap, the target distribution, category counts, default rate per category, numeric features split by target, and scatter plots.

### 3.2 Missing values
- `person_emp_length`: 2.74% missing. `loan_int_rate`: 9.55% missing.
- Missingness in `loan_int_rate` is about 8 to 11% in every loan grade, so it does not appear to depend on grade.
- Filled with the **median**. The final models do this inside the Pipeline, so medians are learned from training data only.

### 3.3 Outliers
- Rows with age above 100 were removed.
- Employment length above 60 years was treated as an error and set to missing.
- Income was **capped at the 99th percentile** (225,000), which affected 322 rows.

### 3.4 Preprocessing Pipeline
| Columns | Transformation |
|---|---|
| Numeric features | Median imputation, then `StandardScaler` |
| `person_home_ownership`, `loan_intent` | `OneHotEncoder` |
| `loan_grade` | `OrdinalEncoder` (A to G) |
| `cb_person_default_on_file` | `OrdinalEncoder` (N, Y) |

The data is split 80/20 with a stratified `train_test_split`.

### 3.5 Models
Logistic Regression, Gaussian Naive Bayes, Random Forest, Gradient Boosting and SVM (RBF). Class imbalance is handled with `class_weight="balanced"` (or balanced sample weights for Gradient Boosting).

## 4. Results

**5-fold stratified cross-validation (training data)**

| Model | Recall | F1 | ROC AUC |
|---|---|---|---|
| Logistic Regression | 0.779 | 0.615 | 0.863 |
| Naive Bayes | 0.673 | 0.615 | 0.838 |
| Random Forest | 0.710 | 0.823 | 0.933 |
| Gradient Boosting | 0.722 | 0.827 | 0.937 |
| SVM | 0.771 | 0.739 | 0.909 |

**Held-out test set**

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Gradient Boosting | 0.901 | 0.754 | 0.809 | 0.781 | **0.942** |
| Random Forest | 0.934 | 0.980 | 0.712 | 0.824 | 0.937 |
| SVM | 0.883 | 0.718 | 0.767 | 0.742 | 0.909 |
| Logistic Regression | 0.796 | 0.522 | 0.786 | 0.627 | 0.869 |
| Naive Bayes | 0.821 | 0.576 | 0.683 | 0.625 | 0.841 |

**Takeaways**
- **Gradient Boosting** has the best ROC AUC and the highest recall, so it catches the most defaults.
- **Random Forest** has the highest precision and accuracy but misses more defaults.
- **Logistic Regression** is a simple, interpretable baseline that still reaches a recall of about 0.79.
- Accuracy alone is misleading with about 78% non-defaults, so recall, F1 and ROC AUC are the main metrics.

## 5. Limitations
- `loan_grade` and `loan_int_rate` are set by the lender, so they partly encode the lender's own risk assessment.
- Outlier cleaning and the income cap were computed on the full dataset before the split, which is a small leak for a baseline.
- Capped income no longer exactly matches `loan_percent_income`.
- No hyperparameter tuning or probability-threshold tuning has been done yet.

---

## Possible next steps
- Hyperparameter tuning with `GridSearchCV` or `RandomizedSearchCV`.
- Tune the decision threshold to trade precision against recall.
- Try the model without `loan_grade` and `loan_int_rate` to reduce lender-assigned signal.
- Add feature importance or SHAP for interpretability.

## Tech stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn

## How to run
```bash
git clone https://github.com/<your-username>/credit-risk-detection.git
cd credit-risk-detection
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Credit_Risk_Detection.ipynb
```
Download `credit_risk_dataset.csv` from Kaggle and place it next to the notebook. Update the path in the first data-loading cell, which currently points to `/content/credit_risk_dataset.csv` (Google Colab).

## Project structure
```
.
├── Credit_Risk_Detection.ipynb
├── credit_risk_dataset.csv
└── README.md
```
