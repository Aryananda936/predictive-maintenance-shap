import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import re


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = joblib.load("lightgbm_model.pkl")

print("========== MODEL LOADED ==========")
print("LightGBM model loaded successfully!")


# ==========================================
# LOAD PREPROCESSED DATA
# ==========================================

df = pd.read_csv("preprocessed_ai4i.csv")

print("\n========== DATASET LOADED ==========")
print("Dataset shape:", df.shape)


# ==========================================
# PREPARE FEATURES
# ==========================================

X = df.drop(
    ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"],
    axis=1
)


# ==========================================
# CLEAN FEATURE NAMES
# ==========================================

X.columns = [
    re.sub(r'[^A-Za-z0-9_]+', '_', col).strip('_')
    for col in X.columns
]

print("\n========== FEATURES ==========")
print(X.columns.tolist())


# ==========================================
# CREATE SHAP EXPLAINER
# ==========================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)


# ==========================================
# HANDLE BINARY CLASSIFICATION
# ==========================================

if isinstance(shap_values, list):
    shap_values_for_failure = shap_values[1]
else:
    shap_values_for_failure = shap_values


# ==========================================
# SHAP SUMMARY PLOT
# ==========================================

plt.figure()

shap.summary_plot(
    shap_values_for_failure,
    X,
    show=False
)

plt.title("SHAP Feature Impact on Machine Failure")

plt.tight_layout()

plt.savefig(
    "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# SHAP FEATURE IMPORTANCE
# ==========================================

shap_importance = pd.DataFrame({
    "Feature": X.columns,
    "Mean Absolute SHAP": abs(shap_values_for_failure).mean(axis=0)
})

shap_importance = shap_importance.sort_values(
    by="Mean Absolute SHAP",
    ascending=False
)

print("\n========== SHAP FEATURE IMPORTANCE ==========")
print(shap_importance)


# ==========================================
# SHAP BAR PLOT
# ==========================================

plt.figure()

shap.summary_plot(
    shap_values_for_failure,
    X,
    plot_type="bar",
    show=False
)

plt.title("SHAP Feature Importance")

plt.tight_layout()

plt.savefig(
    "shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\n==========================================")
print("SHAP ANALYSIS COMPLETED SUCCESSFULLY!")
print("==========================================")
# ==========================================
# INDIVIDUAL MACHINE EXPLANATION
# ==========================================

sample_index = 0

sample = X.iloc[[sample_index]]

prediction = model.predict(sample)[0]

probability = model.predict_proba(sample)[0][1]

print("\n========== INDIVIDUAL MACHINE ==========")

if prediction == 1:
    print("Prediction: MACHINE FAILURE")
else:
    print("Prediction: NO MACHINE FAILURE")

print(f"Failure probability: {probability:.2%}")


# Create SHAP explanation for this machine

sample_shap_values = shap_values_for_failure[sample_index]

explanation = shap.Explanation(
    values=sample_shap_values,
    base_values=explainer.expected_value,
    data=sample.iloc[0].values,
    feature_names=X.columns
)

shap.plots.waterfall(
    explanation,
    max_display=6,
    show=False
)

plt.tight_layout()

plt.savefig(
    "shap_waterfall.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# ==========================================
# HUMAN-READABLE SHAP EXPLANATION
# ==========================================

print("\n========== SHAP EXPLANATION ==========")

feature_contributions = pd.DataFrame({
    "Feature": X.columns,
    "SHAP_Value": sample_shap_values,
    "Feature_Value": sample.iloc[0].values
})

# Sort by absolute SHAP impact
feature_contributions["Absolute_Impact"] = (
    feature_contributions["SHAP_Value"].abs()
)

feature_contributions = feature_contributions.sort_values(
    by="Absolute_Impact",
    ascending=False
)

for _, row in feature_contributions.iterrows():

    feature = row["Feature"]
    shap_value = row["SHAP_Value"]
    feature_value = row["Feature_Value"]

    if shap_value > 0:
        direction = "toward machine failure"
    else:
        direction = "away from machine failure"

    print(
        f"{feature}: value={feature_value}, "
        f"SHAP={shap_value:.3f} → {direction}"
    )
    # ==========================================
# FIND A HIGH-RISK MACHINE
# ==========================================

probabilities = model.predict_proba(X)[:, 1]

high_risk_index = probabilities.argmax()

high_risk_sample = X.iloc[[high_risk_index]]

high_risk_probability = probabilities[high_risk_index]

high_risk_prediction = model.predict(high_risk_sample)[0]

print("\n========== HIGH-RISK MACHINE ==========")

print("Machine index:", high_risk_index)
print("Prediction:", 
      "MACHINE FAILURE" if high_risk_prediction == 1
      else "NO MACHINE FAILURE")

print(f"Failure probability: {high_risk_probability:.2%}")

print("\nMachine values:")
print(high_risk_sample)
# ==========================================
# SHAP EXPLANATION FOR HIGH-RISK MACHINE
# ==========================================

high_risk_shap_values = shap_values_for_failure[high_risk_index]

high_risk_explanation = pd.DataFrame({
    "Feature": X.columns,
    "Feature_Value": high_risk_sample.iloc[0].values,
    "SHAP_Value": high_risk_shap_values
})

high_risk_explanation["Absolute_Impact"] = (
    high_risk_explanation["SHAP_Value"].abs()
)

high_risk_explanation = high_risk_explanation.sort_values(
    by="Absolute_Impact",
    ascending=False
)

print("\n========== HIGH-RISK SHAP EXPLANATION ==========")

for _, row in high_risk_explanation.iterrows():

    if row["SHAP_Value"] > 0:
        direction = "toward machine failure"
    else:
        direction = "away from machine failure"

    print(
        f'{row["Feature"]}: '
        f'value={row["Feature_Value"]}, '
        f'SHAP={row["SHAP_Value"]:.3f} '
        f'→ {direction}'
    )


# ==========================================
# HIGH-RISK WATERFALL PLOT
# ==========================================

high_risk_explanation_object = shap.Explanation(
    values=high_risk_shap_values,
    base_values=explainer.expected_value,
    data=high_risk_sample.iloc[0].values,
    feature_names=X.columns
)

shap.plots.waterfall(
    high_risk_explanation_object,
    max_display=6,
    show=False
)

plt.tight_layout()

plt.savefig(
    "high_risk_shap_waterfall.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()