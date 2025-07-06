# CODSOFT Internship Projects

This repository contains projects completed as part of the **CODSOFT Internship Program**. The projects demonstrate applications of machine learning in real-world scenarios using classification models, data preprocessing, and evaluation techniques.

---

## 📁 Projects Overview

### 🔹 Task 1: Movie Genre Classification

**Objective:**  
Classify movies into genres based on textual features such as title and description.

**Dataset:**  
Kaggle movie dataset containing fields like title, genre, and description.

**Technologies Used:**
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Multinomial Naive Bayes Classifier

**Steps:**
1. **Data Preprocessing:**
   - Removed missing/null values.
   - Converted genres to one-hot encoding.
   - Used `TfidfVectorizer` for converting text to numerical features.
2. **Model Training:**
   - Used `MultinomialNB` (Naive Bayes) classifier.
3. **Evaluation:**
   - Accuracy Score
   - Confusion Matrix
   - Classification Report

**Results:**  
Achieved satisfactory classification accuracy on test data. The model effectively learned to distinguish between genres based on textual content.

---

### 🔹 Task 2: Credit Card Fraud Detection

**Objective:**  
Detect fraudulent credit card transactions from anonymized customer transaction data.

**Dataset:**  
Kaggle Credit Card Fraud Detection dataset (contains features from PCA).

**Technologies Used:**
- Random Forest Classifier
- Logistic Regression
- K-Nearest Neighbors
- StandardScaler for feature scaling
- Class balancing (due to imbalance in fraudulent and non-fraudulent transactions)

**Steps:**
1. **Data Preprocessing:**
   - Scaled features using `StandardScaler`.
   - Checked for class imbalance.
2. **Model Training:**
   - Trained Logistic Regression, Random Forest, and KNN classifiers.
3. **Evaluation:**
   - Accuracy Score
   - Confusion Matrix
   - ROC AUC Score

**Results:**  
Random Forest performed best, effectively identifying fraudulent cases despite class imbalance.

---

### 🔹 Task 3: Customer Churn Prediction

**Objective:**  
Predict if a customer is likely to churn based on their usage and service behavior.

**Dataset:**  
Telecom customer churn dataset from Kaggle or similar.

**Technologies Used:**
- Label Encoding for categorical features
- Logistic Regression
- Random Forest Classifier
- Train-Test Split using Scikit-learn
- Seaborn and Matplotlib for visualization

**Steps:**
1. **Data Preprocessing:**
   - Handled missing values.
   - Encoded categorical variables.
   - Visualized correlations and churn distributions.
2. **Model Training:**
   - Trained Logistic Regression and Random Forest models.
3. **Evaluation:**
   - Confusion Matrix
   - Classification Report
   - ROC AUC Score

**Results:**  
The models accurately predicted customer churn, helping identify key contributing factors such as contract type, tenure, and service features.

---

## 🧠 Models Comparison Summary

| Task                      | Model(s) Used                    | Best Performer           |
|---------------------------|----------------------------------|--------------------------|
| Movie Genre Classification | Multinomial Naive Bayes         | MultinomialNB            |
| Credit Card Fraud Detection | RF, LR, KNN                    | Random Forest            |
| Customer Churn Prediction  | LR, RF                          | Random Forest            |

---

## 📌 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/MohdUmarSiddiq/CODSOFT.git
   cd CODSOFT
   ```
2. Run any of the notebooks using Jupyter:
   ```bash
   jupyter notebook
   ```
## 📚 Folder Structure
  ```bash
  CODSOFT/
  ├── Task 1 - Movie Genre Classification.ipynb
  ├── Task 2 - Credit Card Fraud detection.ipynb
  ├── Task 3 - Customer Churn Prediction.ipynb
  ├── README.md
  ```
## 🔗 Author 
Mohd Umar Siddiqui \
Intern @ CODSOFT \
[GitHub Profile](https://github.com/yourusername)
