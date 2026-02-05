# Student Lifestyle & Depression Prediction Project
# Supervised Machine Learning - Binary Classification
# Model: Logistic Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# 1. Load Dataset

df = pd.read_csv("student_lifestyle_100k.csv")

# 2. Data Preprocessing

# Drop irrelevant column
df.drop("Student_ID", axis=1, inplace=True)

# Encode target variable
df["Depression"] = df["Depression"].map({True: 1, False: 0})

# Encode Gender
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

# One-hot encode Department
df = pd.get_dummies(df, columns=["Department"], drop_first=True)

# 3. Feature & Target Split
X = df.drop("Depression", axis=1)
y = df["Depression"]

# 4. Train-Test Split (Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Model Training
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)
model.fit(X_train, y_train)

# 7. Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 8. Evaluation
print("\n========== MODEL PERFORMANCE ==========")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_prob):.3f}")

# 9. Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance)


# 10. Confusion Matrix Visualization
plt.figure(figsize=(6,4))
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
