import pandas as pd
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

X = df.drop(
    ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"],
    axis=1
)

y = df["Machine failure"]

X.columns = [
    re.sub(r"[^A-Za-z0-9_]+", "_", col).strip("_")
    for col in X.columns
]


# ==========================================
# 2. DEFINE FEATURE SETS
# ==========================================

feature_sets = {
    "Six Features": X.columns.tolist(),
    "Two Features": ["Torque_Nm", "Tool_wear_min"]
}

random_states = range(1, 11)
results = []


# ==========================================
# 3. TRAIN AND TEST BOTH MODELS
# ==========================================

for random_state in random_states:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y
    )

    for model_name, features in feature_sets.items():

        model = LGBMClassifier(
            objective="binary",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=random_state,
            verbosity=-1
        )

        model.fit(X_train[features], y_train)

        y_pred = model.predict(X_test[features])
        y_prob = model.predict_proba(X_test[features])[:, 1]

        results.append({
            "Random State": random_state,
            "Model": model_name,
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


# ==========================================
# 4. DISPLAY ALL RESULTS
# ==========================================

results_df = pd.DataFrame(results)

print("\n========== RESULTS FOR ALL RANDOM STATES ==========")
print(results_df.round(4).to_string(index=False))


# ==========================================
# 5. CALCULATE MEAN AND STANDARD DEVIATION
# ==========================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

summary = results_df.groupby("Model")[metrics].agg(
    ["mean", "std"]
)

print("\n========== COMPARATIVE MODEL PERFORMANCE ==========")
print(summary.round(4).to_string())


# ==========================================
# 6. SAVE RESULTS
# ==========================================

results_df.to_csv("model_comparison_all_runs.csv", index=False)
summary.to_csv("model_comparison_summary.csv")

print("\nResults saved successfully!")
print("1. model_comparison_all_runs.csv")
print("2. model_comparison_summary.csv")