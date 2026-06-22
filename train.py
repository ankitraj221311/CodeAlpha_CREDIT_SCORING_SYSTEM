# ==========================
# IMPORT LIBRARIES
# ==========================

import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from imblearn.over_sampling import SMOTE


# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("datas/german_credit_data.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)


# ==========================
# HANDLE MISSING VALUES
# ==========================

df.fillna("Unknown", inplace=True)


# ==========================
# ENCODE CATEGORICAL FEATURES
# ==========================

encoders = {}

for col in df.select_dtypes(include="object").columns:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    encoders[col] = le


# Save encoders
joblib.dump(
    encoders,
    "models/encoders.pkl"
)

print("\nEncoders Saved Successfully!")


# ==========================
# FEATURES & TARGET
# ==========================

X = df.drop("target", axis=1)

y = df["target"]


# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================
# FEATURE SCALING
# ==========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("Scaler Saved Successfully!")


# ==========================
# HANDLE IMBALANCED DATA
# ==========================

smote = SMOTE(
    random_state=42
)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print(pd.Series(y_train).value_counts())


# ==========================
# TRAIN MODEL
# ==========================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("\nModel Training Complete!")


# ==========================
# PREDICTIONS
# ==========================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# ==========================
# EVALUATION
# ==========================

print("\n========== MODEL PERFORMANCE ==========")

print(
    f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}"
)

print(
    f"Precision : {precision_score(y_test, y_pred):.4f}"
)

print(
    f"Recall    : {recall_score(y_test, y_pred):.4f}"
)

print(
    f"F1 Score  : {f1_score(y_test, y_pred):.4f}"
)

print(
    f"ROC AUC   : {roc_auc_score(y_test, y_prob):.4f}"
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================
# SAVE MODEL
# ==========================

joblib.dump(
    model,
    "models/model.pkl"
)

print("\nModel Saved Successfully!")


# ==========================
# FEATURE IMPORTANCE
# ==========================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")

print(
    importance.head(10)
)


# ==========================
# TARGET DISTRIBUTION
# ==========================

print("\nTarget Distribution:")

print(
    df["target"].value_counts()
)

print("\nTraining Completed Successfully!")