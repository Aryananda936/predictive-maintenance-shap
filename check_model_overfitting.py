import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
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
# 2. DEFINE MODELS
# ==========================================

feature_sets = {
    "Six Features": X.columns.tolist(),
    "Two Features": ["Torque_Nm", "Tool_wear_min"]
}

random_states = range(1, 11)

results = []


# ==========================================
# 3. TRAIN AND EVALUATE
# ==========================================

for random_state in random_states:

    print("\n" + "=" * 65)
    print("RANDOM STATE:", random_state)
    print("=" * 65)

    # Same split for both models
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

        # Train using the selected features
        model.fit(X_train[features], y_train)

        # Predictions on training data
        train_pred = model.predict(X_train[features])

        # Predictions on unseen test data
        test_pred = model.predict(X_test[features])

        # Calculate accuracy
        train_accuracy = accuracy_score(
            y_train, train_pred
        )

        test_accuracy = accuracy_score(
            y_test, test_pred
        )

        # Store results
        results.append({
            "Random State": random_state,
            "Model": model_name,
            "Train Accuracy": train_accuracy,
            "Test Accuracy": test_accuracy,
            "Accuracy Gap": train_accuracy - test_accuracy,
            "Test Precision": precision_score(
                y_test, test_pred, zero_division=0
            ),
            "Test Recall": recall_score(
                y_test, test_pred, zero_division=0
            ),
            "Test F1": f1_score(
                y_test, test_pred, zero_division=0
            )
        })

        print(f"\nModel: {model_name}")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy : {test_accuracy:.4f}")
        print(
            f"Accuracy Gap     : "
            f"{train_accuracy - test_accuracy:.4f}"
        )


# ==========================================
# 4. DISPLAY ALL RESULTS
# ==========================================

results_df = pd.DataFrame(results)

print("\n\n========== ALL RESULTS ==========")
print(results_df.round(4).to_string(index=False))


# ==========================================
# 5. AVERAGE RESULTS
# ==========================================

metrics = [
    "Train Accuracy",
    "Test Accuracy",
    "Accuracy Gap",
    "Test Precision",
    "Test Recall",
    "Test F1"
]

summary = results_df.groupby("Model")[metrics].agg(
    ["mean", "std"]
)

print("\n\n========== OVERFITTING CHECK SUMMARY ==========")
print(summary.round(4).to_string())


# ==========================================
# 6. CONFUSION MATRICES
# ==========================================

# Use random state 42 for a separate illustrative diagnostic split.
# This is not one of the ten random states above.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

for model_name, features in feature_sets.items():

    model = LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        verbosity=-1
    )

    model.fit(X_train[features], y_train)

    test_pred = model.predict(X_test[features])

    print(f"\n========== {model_name}: CONFUSION MATRIX ==========")
    print(confusion_matrix(y_test, test_pred))

    print(f"\n========== {model_name}: CLASSIFICATION REPORT ==========")
    print(
        classification_report(
            y_test,
            test_pred,
            zero_division=0
        )
    )


# ==========================================
# 7. SAVE RESULTS
# ==========================================

results_df.to_csv(
    "overfitting_check_all_runs.csv",
    index=False
)

summary.to_csv(
    "overfitting_check_summary.csv"
)

print("\nDiagnostic results saved successfully!")