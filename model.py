# Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score


# Load dataset
data = pd.read_csv("Churn_Modelling.csv")

print("Dataset Shape:", data.shape)
print(data.head())


# -------------------------------
# Data Preprocessing
# -------------------------------

# Convert categorical columns properly
label_encoder_geo = LabelEncoder()
label_encoder_gen = LabelEncoder()

data['Geography'] = label_encoder_geo.fit_transform(data['Geography'])
data['Gender'] = label_encoder_gen.fit_transform(data['Gender'])


# Separate features and target
# Drop useless columns
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
X = data.drop("Exited", axis=1)
y = data["Exited"]


# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -------------------------------
# Model 1 : Logistic Regression
# -------------------------------

lr_model = LogisticRegression()

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\nLogistic Regression Accuracy:", accuracy_score(y_test, lr_pred))


# -------------------------------
# Model 2 : Random Forest
# -------------------------------

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf_pred))


# -------------------------------
# Model 3 : Gradient Boosting
# -------------------------------

gb_model = GradientBoostingClassifier()

gb_model.fit(X_train, y_train)

gb_pred = gb_model.predict(X_test)

print("\nGradient Boosting Accuracy:", accuracy_score(y_test, gb_pred))


# -------------------------------
# Evaluation
# -------------------------------

print("\nConfusion Matrix (Random Forest)")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report")
print(classification_report(y_test, rf_pred))