# 📦 وارد کردن کتابخانه‌ها
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 📂 خواندن داده‌ها
df = pd.read_csv('./Social_Network_Ads.csv')

# ❌ حذف ستون غیرضروری
df.drop('User ID', axis=1, inplace=True)

# 🧠 تبدیل ستون Gender به عددی (0 و 1)
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

# 🧩 جدا کردن X و y
X = df.drop('Purchased', axis=1)
y = df['Purchased']

# 📚 تقسیم داده‌ها به train و test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ⚙️ تعریف مدل Logistic Regression
clf = LogisticRegression(class_weight='balanced')

# ✅ آموزش و ارزیابی روی داده‌های train/test
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n=== Single Train/Test Evaluation ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

# ====================================================
# 🎯 K-Fold Cross Validation
# ====================================================

print("\n=== K-Fold Cross Validation ===")

data_split = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in data_split.split(X, y):
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X.iloc[train_idx], y.iloc[train_idx])
    train_acc = clf.score(X.iloc[train_idx], y.iloc[train_idx])
    val_acc = clf.score(X.iloc[val_idx], y.iloc[val_idx])
    print(f"Train accuracy = {train_acc:.3f}, Validation accuracy = {val_acc:.3f}")

# ====================================================
# 🧪 تست چند مدل (اختیاری)
# ====================================================

models = [
    LogisticRegression(class_weight='balanced'),
    LogisticRegression()
]


for model in models:
    model.fit(X_train, y_train)
    print("Test Accuracy:", model.score(X_test, y_test))
    print("Train Accuracy:", model.score(X_train, y_train))
    print("----")
