
---

# 📢 Advertising Sales Prediction Model

## 🧠 Project Overview

This project aims to develop a **Linear Regression** model to predict product sales based on advertising budgets across different media channels. The goal is to understand how investments in **TV**, **Radio**, and **Newspaper** advertising affect sales performance.

---

## 📂 Dataset

The dataset, stored in `Dataset/advertising (1).csv`, includes the following features:

* **TV**: Advertising budget spent on TV (in thousands of dollars)
* **Radio**: Advertising budget spent on Radio (in thousands of dollars)
* **Newspaper**: Advertising budget spent on Newspaper (in thousands of dollars)
* **Sales**: Units sold (in thousands) — *Target variable*

---

## 🔧 Workflow

### 1. **Data Preprocessing**

* Loaded dataset using **pandas**
* Separated independent variables (`TV`, `Radio`, `Newspaper`) and the target variable (`Sales`)

### 2. **Model Building**

* Performed an 80/20 **train-test split** using `train_test_split`
* Initialized and trained a **Linear Regression** model using `scikit-learn`

### 3. **Model Evaluation**

* Made predictions on the test set
* Evaluated the model using:

  * **Mean Squared Error (MSE)**
  * **Root Mean Squared Error (RMSE)**
  * **R-squared Score (R²)**

### 4. **Results Output**

* Evaluation results including coefficients, intercept, and metrics were saved to a file named `model_evaluation.txt`

---

## 📈 Sample Output

An example of the output saved in `model_evaluation.txt`:

```
Model Evaluation Results:
Model Coefficients: [3.928 2.828 -0.104]
Intercept: 2.939
Mean Squared Error (MSE): 1.93
Root Mean Squared Error (RMSE): 1.39
R-squared (R2): 0.90
```

> 📌 *Note: Actual values will vary depending on the dataset.*

---

## ✅ Requirements

Make sure the following libraries are installed before running the project:

* `pandas`
* `numpy`
* `scikit-learn`

You can install them via pip:

```bash
pip install pandas numpy scikit-learn
```

---

## 🚀 How to Run

1. Place the dataset in the specified folder: `Dataset/advertising (1).csv`
2. Run the Python script.
3. View results in `model_evaluation.txt`

---

## 🏁 Conclusion

This project demonstrates how **linear regression** can effectively model the relationship between advertising spending and sales. The insights can guide businesses in allocating advertising budgets to maximize returns.

---
