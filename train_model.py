import pandas as pd
import matplotlib.pyplot as plt
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from lightgbm import LGBMClassifier


# ==========================================
# LOAD PREPROCESSED DATA
# ==========================================

df = pd.read_csv("preprocessed_ai4i.csv")

print("========== DATASET LOADED ==========")
print("Dataset shape:", df.shape)


# ==========================================
# SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop(
    ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"],
    axis=1
)

y = df["Machine failure"]


# ==========================================
# CLEAN FEATURE NAMES FOR LIGHTGBM
# ==========================================

X.columns = [
    re.sub(r'[^A-Za-z0-9_]+', '_', col).strip('_')
    for col in X.columns
]


# ==========================================
# RANDOM STATES
# ==========================================

random_states = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# ==========================================
# STORE RESULTS
# ==========================================

results = []


# ==========================================
# RUN MODEL 10 TIMES
# ==========================================

for random_state in random_states:

    print("\n==========================================")
    print("RANDOM STATE:", random_state)
    print("==========================================")

    # ------------------------------------------
    # TRAIN-TEST SPLIT
    # ------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y
    )

    print("Training data:", X_train.shape)
    print("Testing data:", X_test.shape)


    # ------------------------------------------
    # CREATE LIGHTGBM MODEL
    # ------------------------------------------

    model = LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=random_state
    )


    # ------------------------------------------
    # TRAIN MODEL
    # ------------------------------------------

    print("Training LightGBM...")

    model.fit(X_train, y_train)


    # ------------------------------------------
    # PREDICTION
    # ------------------------------------------

    y_pred = model.predict(X_test)

    # Probability of Machine Failure = 1
    y_prob = model.predict_proba(X_test)[:, 1]


    # ------------------------------------------
    # MODEL EVALUATION
    # ------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # ROC-AUC uses predicted probabilities
    roc_auc = roc_auc_score(y_test, y_prob)


    # ------------------------------------------
    # STORE RESULTS
    # ------------------------------------------

    results.append({
        "Random State": random_state,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })


    # ------------------------------------------
    # PRINT CURRENT RESULT
    # ------------------------------------------

    print("\nPerformance:")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("ROC-AUC  :", roc_auc)


# ==========================================
# RESULTS TABLE
# ==========================================

results_df = pd.DataFrame(results)

print("\n\n==========================================")
print("RESULTS FOR ALL 10 RANDOM STATES")
print("==========================================")

print(results_df.to_string(index=False))


# ==========================================
# AVERAGE PERFORMANCE
# ==========================================

average_accuracy = results_df["Accuracy"].mean()
average_precision = results_df["Precision"].mean()
average_recall = results_df["Recall"].mean()
average_f1 = results_df["F1 Score"].mean()
average_roc_auc = results_df["ROC-AUC"].mean()


print("\n\n==========================================")
print("AVERAGE PERFORMANCE OVER 10 RUNS")
print("==========================================")

print("Average Accuracy :", average_accuracy)
print("Average Precision:", average_precision)
print("Average Recall   :", average_recall)
print("Average F1 Score :", average_f1)
print("Average ROC-AUC  :", average_roc_auc)


# ==========================================
# SAVE FINAL MODEL
# ==========================================

# Train one final model using random_state=42
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
    random_state=42
)

final_model.fit(X_train, y_train)

joblib.dump(final_model, "lightgbm_model.pkl")

print("\n==========================================")
print("FINAL MODEL SAVED")
print("==========================================")
print("Saved as: lightgbm_model.pkl")
# ==========================================
# LIGHTGBM FEATURE IMPORTANCE
# ==========================================

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== LIGHTGBM FEATURE IMPORTANCE ==========")
print(feature_importance)
plt.figure(figsize=(8, 5))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("LightGBM Feature Importance")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()