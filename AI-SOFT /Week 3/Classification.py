# Classical ML with Scikit-learn
# Iris Dataset (via CSV upload)

# STEP 1: Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 2: Upload your CSV file (Colab prompt)
# --------------------------------------------
# When you run this cell in Google Colab, you’ll see a file selector.
from google.colab import files
uploaded = files.upload()

# Assuming file is named "iris.csv"
filename = list(uploaded.keys())[0]
print(f"\n✅ File uploaded successfully: {filename}")

# STEP 3: Load and inspect data
df = pd.read_csv(filename)
print("\n🔍 First 5 rows:")
print(df.head())

print("\n📊 Column names:", df.columns.tolist())
print("Data shape:", df.shape)
print(df.info())

# STEP 4: Handle missing values (if any)
df = df.dropna()   # or df.fillna(df.median(), inplace=True)
print("\n✅ After cleaning, shape:", df.shape)

# STEP 5: Encode label column if it's categorical
# -----------------------------------------------
# Automatically detect non-numeric column for target
target_col = None
for col in df.columns:
    if df[col].dtype == 'object':
        target_col = col
        break

if target_col is None:
    target_col = df.columns[-1]  # fallback to last column

print(f"\n🎯 Target column detected: {target_col}")

# Separate features (X) and labels (y)
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode labels if they are strings
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(f"Classes: {le.classes_}")

# STEP 6: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\n✅ Data split complete:")
print("Train samples:", len(X_train), " | Test samples:", len(X_test))

# STEP 7: Train model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
print("\n🌳 Model training complete!")

# STEP 8: Evaluate
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')

print("\n📈 Evaluation Results:")
print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# STEP 9: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y),
            yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Iris Decision Tree')
plt.show()

print("\n✅ Task 1 complete — ready for screenshots and report.")
