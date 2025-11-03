# Install libraries
!pip install scikit-learn pandas numpy matplotlib seaborn

# Import dependencies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load dataset (you can replace the URL with your uploaded CSV)
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

# Convert to DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Simulate “issue priority” categories (High/Medium/Low)
df['priority'] = pd.cut(df['mean area'], bins=3, labels=['Low', 'Medium', 'High'])

# Encode labels
encoder = LabelEncoder()
df['priority_encoded'] = encoder.fit_transform(df['priority'])

# Split data
X = df[data.feature_names]
y = df['priority_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("✅ Model Evaluation Results")
print("---------------------------")
print(f"Accuracy: {accuracy:.2f}")
print(f"F1-Score: {f1:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

✅ Model Evaluation Results
---------------------------
Accuracy: 1.00
F1-Score: 1.00

Classification Report:
              precision    recall  f1-score   support

         Low       1.00      1.00      1.00         1
      Medium       1.00      1.00      1.00        92
        High       1.00      1.00      1.00        21

    accuracy                           1.00       114
   macro avg       1.00      1.00      1.00       114
weighted avg       1.00      1.00      1.00       114
