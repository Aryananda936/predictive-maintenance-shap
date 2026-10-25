import pandas as pd
import numpy as np
import joblib
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from lightgbm import LGBMClassifier


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("preprocessed_ai4i.csv")

print("========== DATASET LOADED ==========")
print("Dataset shape:", df.shape)


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop(
    ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"],
    axis=1
)

y = df["Machine failure"]

# Clean feature names
X.columns = [
    re.sub(r"[^A-Za-z0-9_]+", "_", col).strip("_")
    for col in X.columns
]


# ==========================================
# 3. SELECT TOP TWO FEATURES
# ==========================================

top_two_features = [
    "Torque_Nm",
    "Tool_wear_min"
]

X = X[top_two_features]

print("\nSelected features:")
print(X.columns.tolist())


# ==========================================
# 4. DEFINE RANDOM STATES
# ==========================================

random_states = list(range(1, 11))

results = []


# ==========================================
# 5. TRAIN AND TEST OVER TEN RANDOM STATES
# ==========================================

for random_state in random_states:

    print("\n====================================")
    print("RANDOM STATE:", random_state)
    print("====================================")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y
    )

    model = LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=random_state,
        verbosity=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "Random State": random_state,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })

    print("Accuracy :", f"{results[-1]['Accuracy']:.4f}")
    print("Precision:", f"{results[-1]['Precision']:.4f}")
    print("Recall   :", f"{results[-1]['Recall']:.4f}")
    print("F1 Score :", f"{results[-1]['F1 Score']:.4f}")
    print("ROC-AUC  :", f"{results[-1]['ROC-AUC']:.4f}")


# ==========================================
# 6. RESULTS FOR ALL TEN RUNS
# ==========================================

results_df = pd.DataFrame(results)

print("\n========== TOP TWO MODEL: ALL RUNS ==========")
print(results_df.to_string(index=False))


# ==========================================
# 7. MEAN AND STANDARD DEVIATION
# ==========================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

print("\n========== TOP TWO MODEL: SUMMARY ==========")

summary = results_df[metrics].agg(["mean", "std"])

print(summary.round(4))


# ==========================================
# 8. SAVE RESULTS
# ==========================================

results_df.to_csv(
    "top2_model_all_runs.csv",
    index=False
)

summary.to_csv(
    "top2_model_summary.csv"
)


# ==========================================
# 9. TRAIN FINAL TOP TWO MODEL
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

final_model = LGBMClassifier(
    objective="binary",
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    verbosity=-1
)

final_model.fit(X_train, y_train)

joblib.dump(
    final_model,
    "lightgbm_top2_model.pkl"
)

print("\n========== FINAL MODEL SAVED ==========")
print("File: lightgbm_top2_model.pkl")
print("Results: top2_model_all_runs.csv")
print("Summary: top2_model_summary.csv")
