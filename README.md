# Credit Card Default Prediction

## 📌 Project Overview

This project uses **Machine Learning** to predict whether a credit card customer is likely to default on their payment in the next month.

The project uses customer information such as credit limit, demographic details, repayment status, bill amounts, and previous payment amounts. The notebook performs data cleaning, exploratory data analysis, preprocessing, class balancing, feature selection, scaling, and classification using **Logistic Regression**.

## 🎯 Objective

The main objective is to build a machine learning classification model that predicts whether a customer will default on their credit card payment in the next month.

- `0` → No Default
- `1` → Default

## 📊 Dataset

The project uses the `creditcard.csv` dataset.

- **Rows:** 30,000
- **Columns:** 25
- **Target:** `default payment next month`

### Main Features

- `LIMIT_BAL` – Credit limit
- `SEX` – Customer gender
- `EDUCATION` – Education level
- `MARRIAGE` – Marital status
- `AGE` – Customer age
- `PAY_0` to `PAY_6` – Previous repayment status
- `BILL_AMT1` to `BILL_AMT6` – Previous bill amounts
- `PAY_AMT1` to `PAY_AMT6` – Previous payment amounts

## 🔍 Exploratory Data Analysis

The project includes:

- Dataset inspection
- Missing-value checking
- Duplicate-value checking
- Correlation analysis
- Correlation heatmap
- Boxplots for numerical features
- Target-class distribution analysis

The dataset contains **no missing values** and **no duplicate rows**.

The original target distribution was:

| Class | Count |
|---|---:|
| No Default | 23,364 |
| Default | 6,636 |

This shows that the target classes are imbalanced.

## 🛠️ Data Preprocessing

### 1. Column Renaming

The repayment, bill, and payment columns were renamed to more descriptive month-based names.

Examples:

- `PAY_0` → `SEP_PAY`
- `PAY_2` → `AUG_PAY`
- `BILL_AMT1` → `SEP_BILL`
- `PAY_AMT1` → `SEP_PAYMENT`
- `default payment next month` → `target`

### 2. Categorical Value Conversion

Categorical values were converted into readable labels.

Examples:

- Gender → `M`, `F`
- Education → `UG`, `PG`, `HIGH_SCHOOL`, `OTHERS`
- Marriage → `MARRIED`, `SINGLE`, `OTHERS`
- Target → `YES`, `NO`

### 3. Outlier Handling

An **IQR-based outlier treatment** was applied to numerical columns.

### 4. Encoding

- `LabelEncoder` was used for the target and gender.
- `OneHotEncoder` was used for education and marriage categories.

### 5. Class Balancing

Since the target classes were imbalanced, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied.

After SMOTE:

- Class 0 → 23,364
- Class 1 → 23,364
- Total → 46,728 records

### 6. Feature Selection

`SelectKBest` with `f_classif` was used to select the top **25 features**.

Important selected features include:

- `SEP_PAY`
- `AUG_PAY`
- `JUL_PAY`
- `JUN_PAY`
- `MAY_PAY`
- `APR_PAY`
- `LIMIT_BAL`
- `SEP_PAYMENT`
- `AUG_PAYMENT`
- `JUL_PAYMENT`
- `JUN_PAYMENT`
- `MAY_PAYMENT`
- Education features
- Marriage features

### 7. Feature Scaling

`StandardScaler` was used to standardize the selected features before model training.

## 🤖 Machine Learning Model

### Logistic Regression

The project uses **Logistic Regression** as the classification model.

The balanced dataset was divided into:

- **80% Training Data**
- **20% Testing Data**
- **Random State:** 40

Dataset shapes:

- Training features: `(37382, 25)`
- Testing features: `(9346, 25)`

## 📈 Model Performance

The Logistic Regression model achieved the following results on the test set:

| Metric | Score |
|---|---:|
| Accuracy | 68.29% |
| Precision | 68.96% |
| Recall | 68.03% |
| F1 Score | 68.49% |

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| 0 | 0.68 | 0.69 | 0.68 |
| 1 | 0.69 | 0.68 | 0.68 |

The model provides reasonably balanced performance across both classes after applying SMOTE.

## 💻 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Jupyter Notebook / Google Colab

## 📁 Project Structure

```text
Credit-Card-Default-Prediction/
│
├── credit card project.ipynb
├── creditcard.csv
└── README.md
```

> **Note:** Check the original dataset's terms of use before redistributing the dataset in a public repository.

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Credit-Card-Default-Prediction
```

### 2. Install the required libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

### 3. Open the notebook

```bash
jupyter notebook
```

Then open the project `.ipynb` file and run the cells.

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Inspection
   ↓
Missing & Duplicate Value Check
   ↓
Exploratory Data Analysis
   ↓
Feature Renaming
   ↓
Categorical Encoding
   ↓
Outlier Treatment
   ↓
SMOTE Class Balancing
   ↓
Feature Selection
   ↓
Standard Scaling
   ↓
Train-Test Split
   ↓
Logistic Regression
   ↓
Model Evaluation
```

## 🚀 Future Improvements

- Compare multiple classification algorithms such as Random Forest, Decision Tree, XGBoost, and SVM.
- Use cross-validation for more reliable evaluation.
- Perform hyperparameter tuning.
- Evaluate ROC-AUC and PR-AUC.
- Create a confusion matrix and feature-importance analysis.
- Build a simple web application for real-time prediction.

## 📌 Conclusion

This project demonstrates an end-to-end machine learning workflow for predicting credit card payment default.

After preprocessing the data and balancing the target classes using SMOTE, the selected features were scaled and used to train a Logistic Regression classifier. The model achieved approximately **68.29% accuracy** on the test data.

## 👩‍💻 Project Type

**Machine Learning / Data Science Project**

This project demonstrates practical skills in:

- Data preprocessing
- Exploratory Data Analysis
- Feature engineering
- Handling imbalanced datasets
- Feature selection
- Machine learning classification
- Model evaluation
