import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import re


# ==========================================
# 1. LOAD THE SIX-FEATURE MODEL
# ==========================================

model = joblib.load("lightgbm_model.pkl")

print("Six-feature LightGBM model loaded successfully!")


# ==========================================
# 2. LOAD THE DATASET
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

print("\nFeatures used:")
print(X.columns.tolist())


# ==========================================
# 3. LIGHTGBM FEATURE IMPORTANCE
# ==========================================

# Gain measures the improvement in the model's
# objective function contributed by feature splits.

lgbm_importance = model.booster_.feature_importance(
    importance_type="gain"
)

lgbm_df = pd.DataFrame({
    "Feature": X.columns,
    "LightGBM Importance": lgbm_importance
})

lgbm_df["LightGBM Rank"] = (
    lgbm_df["LightGBM Importance"]
    .rank(ascending=False, method="min")
    .astype(int)
)


# ==========================================
# 4. CALCULATE SHAP VALUES
# ==========================================

explainer = shap.TreeExplainer(model)

# Use a reproducible sample to calculate global importance.
X_sample = X.sample(
    n=min(2000, len(X)),
    random_state=42
)

shap_values = explainer.shap_values(X_sample)


# Handle different SHAP output formats.
if isinstance(shap_values, list):

    shap_values_failure = shap_values[1]

elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:

    shap_values_failure = shap_values[:, :, 1]

else:

    shap_values_failure = shap_values


# ==========================================
# 5. SHAP FEATURE IMPORTANCE
# ==========================================

mean_abs_shap = np.abs(shap_values_failure).mean(axis=0)

shap_df = pd.DataFrame({
    "Feature": X_sample.columns,
    "Mean Absolute SHAP": mean_abs_shap
})

shap_df["SHAP Rank"] = (
    shap_df["Mean Absolute SHAP"]
    .rank(ascending=False, method="min")
    .astype(int)
)


# ==========================================
# 6. COMBINE BOTH RANKINGS
# ==========================================

comparison = pd.merge(
    lgbm_df,
    shap_df,
    on="Feature"
)

comparison = comparison.sort_values("SHAP Rank")

print("\n========== FEATURE IMPORTANCE COMPARISON ==========")

print(
    comparison[
        [
            "Feature",
            "LightGBM Importance",
            "LightGBM Rank",
            "Mean Absolute SHAP",
            "SHAP Rank"
        ]
    ].to_string(index=False)
)


# ==========================================
# 7. SAVE RESULTS
# ==========================================

comparison.to_csv(
    "feature_importance_comparison.csv",
    index=False
)

print("\nSaved: feature_importance_comparison.csv")


# ==========================================
# 8. VISUAL COMPARISON OF RANKINGS
# ==========================================

plot_df = comparison.sort_values("SHAP Rank")

plt.figure(figsize=(10, 6))

positions = np.arange(len(plot_df))
bar_height = 0.35

plt.barh(
    positions - bar_height / 2,
    plot_df["SHAP Rank"],
    height=bar_height,
    label="SHAP Rank"
)

plt.barh(
    positions + bar_height / 2,
    plot_df["LightGBM Rank"],
    height=bar_height,
    label="LightGBM Rank"
)

plt.yticks(positions, plot_df["Feature"])

plt.xlabel("Importance Rank (1 = Most Important)")
plt.ylabel("Feature")

plt.title("SHAP vs LightGBM Feature Importance Ranking")

plt.gca().invert_yaxis()
plt.legend()
plt.tight_layout()

plt.savefig(
    "feature_importance_rank_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# 9. COMPLETION
# ==========================================

print("\n==========================================")
print("FEATURE IMPORTANCE COMPARISON COMPLETED!")
print("==========================================")