# Customer Churn Prediction

## Project Overview

Customer churn refers to when a customer stops using a company's service.
This project builds a **Machine Learning model** to predict whether a customer will leave a subscription-based service.

The model uses historical customer data such as **usage behavior, demographics, and service information** to identify customers likely to churn.

Predicting churn helps companies:

* Retain valuable customers
* Improve customer satisfaction
* Reduce revenue loss

---

## Dataset

The dataset contains customer information including:

* Customer demographics
* Subscription details
* Usage behavior
* Billing information
* Churn status

### Example Features

* Age
* Gender
* Tenure
* Monthly Charges
* Usage
* Contract Type
* Churn (Target Variable)

**Target Variable**

Churn

* `0` → Customer stays
* `1` → Customer leaves

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

---

## Machine Learning Algorithms

The following models are used to predict churn:

1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier

These models are trained and compared to find the best-performing algorithm.

---

## Project Workflow

1. Data Collection
2. Data Preprocessing
3. Feature Encoding
4. Train-Test Split
5. Model Training
6. Model Evaluation

---

## Model Evaluation Metrics

The models are evaluated using:

* Accuracy
* Confusion Matrix
* Classification Report

These metrics help measure how well the model predicts customer churn.

---

## How to Run the Project

### 1 Install Required Libraries

pip install pandas numpy scikit-learn matplotlib seaborn

### 2 Run the Python Script

python churn_prediction.py

### 3 Provide Dataset

Place the dataset file:

churn.csv

in the project folder.

---

## Project Structure

Customer-Churn-Prediction
│
├── churn.csv
├── churn_prediction.py
└── README.md

---

## Expected Output

The program trains multiple machine learning models and prints their accuracy scores.

Example:

Logistic Regression Accuracy: 0.82
Random Forest Accuracy: 0.88
Gradient Boosting Accuracy: 0.86

---

## Applications

Customer churn prediction can be used in:

* Telecom companies
* Subscription platforms
* Banking services
* SaaS businesses
* Online streaming platforms

---

## Conclusion

This project demonstrates how machine learning can be used to analyze customer behavior and predict churn. Companies can use such models to take proactive steps to retain customers and improve business performance.

---

## Future Improvements

* Hyperparameter tuning
* Deep learning models
* Dashboard visualization
* Real-time prediction system
