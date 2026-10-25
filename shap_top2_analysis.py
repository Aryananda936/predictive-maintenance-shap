import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import re
import numpy as np


# ==========================================
# 1. LOAD THE TWO-FEATURE MODEL
# ==========================================

model = joblib.load("lightgbm_top2_model.pkl")

print("========== MODEL LOADED ==========")
print("Two-feature LightGBM model loaded successfully!")


# ==========================================
# 2. LOAD PREPROCESSED DATA
# ==========================================

df = pd.read_csv("preprocessed_ai4i.csv")

print("\n========== DATASET LOADED ==========")
print("Dataset shape:", df.shape)


# ==========================================
# 3. PREPARE FEATURES AND TARGET
# ==========================================

X = df.drop(
    ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"],
    axis=1
)

y = df["Machine failure"]

# Clean feature names to match model training
X.columns = [
    re.sub(r"[^A-Za-z0-9_]+", "_", col).strip("_")
    for col in X.columns
]


# ==========================================
# 4. SELECT THE TWO FEATURES
# ==========================================

top_two_features = [
    "Torque_Nm",
    "Tool_wear_min"
]

X_top2 = X[top_two_features].copy()

print("\n========== SELECTED FEATURES ==========")
print(X_top2.columns.tolist())


# ==========================================
# 5. CREATE SHAP EXPLAINER
# ==========================================

explainer = shap.TreeExplainer(model)

# Explain a manageable sample of the dataset
X_sample = X_top2.sample(
    n=min(1000, len(X_top2)),
    random_state=42
)

shap_values = explainer.shap_values(X_sample)


# ==========================================
# 6. HANDLE SHAP OUTPUT
# ==========================================

if isinstance(shap_values, list):
    shap_values_failure = shap_values[1]

elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    # Some SHAP versions return:
    # samples x features x classes
    shap_values_failure = shap_values[:, :, 1]

else:
    shap_values_failure = shap_values


# ==========================================
# 7. SHAP FEATURE IMPORTANCE
# ==========================================

shap_importance = pd.DataFrame({
    "Feature": X_sample.columns,
    "Mean Absolute SHAP": np.abs(
        shap_values_failure
    ).mean(axis=0)
})

shap_importance = shap_importance.sort_values(
    by="Mean Absolute SHAP",
    ascending=False
)

print("\n========== TOP TWO MODEL: SHAP IMPORTANCE ==========")
print(shap_importance.to_string(index=False))


# ==========================================
# 8. SHAP SUMMARY PLOT
# ==========================================

shap.summary_plot(
    shap_values_failure,
    X_sample,
    show=False
)

plt.title("SHAP Summary - Two-Feature LightGBM Model")
plt.tight_layout()

plt.savefig(
    "shap_top2_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# 9. SHAP BAR PLOT
# ==========================================

shap.summary_plot(
    shap_values_failure,
    X_sample,
    plot_type="bar",
    show=False
)

plt.title("SHAP Feature Importance - Two-Feature Model")
plt.tight_layout()

plt.savefig(
    "shap_top2_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# 10. EXPLAIN ONE MACHINE
# ==========================================

sample_index = 0

sample = X_top2.iloc[[sample_index]]

prediction = model.predict(sample)[0]

probability = model.predict_proba(sample)[0][1]

print("\n========== INDIVIDUAL MACHINE ==========")

print("Machine measurements:")
print(sample.to_string(index=False))

print(
    "Prediction:",
    "MACHINE FAILURE" if prediction == 1
    else "NO MACHINE FAILURE"
)

print(f"Failure probability: {probability:.2%}")


# ==========================================
# 11. INDIVIDUAL SHAP EXPLANATION
# ==========================================

sample_shap = explainer.shap_values(sample)

if isinstance(sample_shap, list):
    sample_shap_failure = sample_shap[1][0]

elif isinstance(sample_shap, np.ndarray) and sample_shap.ndim == 3:
    sample_shap_failure = sample_shap[0, :, 1]

else:
    sample_shap_failure = sample_shap[0]


print("\n========== INDIVIDUAL SHAP CONTRIBUTIONS ==========")

for feature, value, shap_value in zip(
    sample.columns,
    sample.iloc[0].values,
    sample_shap_failure
):

    direction = (
        "toward machine failure"
        if shap_value > 0
        else "away from machine failure"
    )

    print(
        f"{feature}: value={value}, "
        f"SHAP={shap_value:.4f} "
        f"-> {direction}"
    )


# ==========================================
# 12. INDIVIDUAL WATERFALL PLOT
# ==========================================

explanation = explainer(sample)

if isinstance(explanation.values, np.ndarray) and explanation.values.ndim == 3:
    explanation = explanation[:, :, 1]

shap.plots.waterfall(
    explanation[0],
    max_display=2,
    show=False
)

plt.tight_layout()

plt.savefig(
    "shap_top2_waterfall.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# 13. COMPLETION
# ==========================================

print("\n==========================================")
print("TWO-FEATURE SHAP ANALYSIS COMPLETED!")
print("==========================================")

print("Generated files:")
print("1. shap_top2_summary.png")
print("2. shap_top2_feature_importance.png")
print("3. shap_top2_waterfall.png")