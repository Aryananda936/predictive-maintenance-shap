import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. LOAD EXISTING RESULTS
# ==========================================

df = pd.read_csv("model_comparison_all_runs.csv")

print("========== COMPARISON RESULTS LOADED ==========")

# ==========================================
# 2. CALCULATE AVERAGE PERFORMANCE
# ==========================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

summary = df.groupby("Model")[metrics].mean()

# Convert scores to percentages for readability
summary_percent = summary * 100

print("\n========== SIX-FEATURE VS TWO-FEATURE MODEL ==========")
print(summary_percent.round(2).to_string())

# Save the comparison table
summary_percent.to_csv("model_comparison_percentages.csv")


# ==========================================
# 3. CREATE COMPARISON GRAPH
# ==========================================

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "ROC-AUC"
]

six_feature_scores = (
    summary.loc["Six Features", metrics].values * 100
)

two_feature_scores = (
    summary.loc["Two Features", metrics].values * 100
)

x = np.arange(len(metric_names))
width = 0.35

plt.figure(figsize=(11, 6))

plt.bar(
    x - width / 2,
    six_feature_scores,
    width,
    label="Six Features"
)

plt.bar(
    x + width / 2,
    two_feature_scores,
    width,
    label="Two Features"
)

plt.xticks(x, metric_names)
plt.ylabel("Performance (%)")
plt.xlabel("Evaluation Metrics")

plt.title("LightGBM: Six Features vs Two Features")

plt.ylim(0, 105)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "model_performance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nGraph saved as model_performance_comparison.png")
print("Table saved as model_comparison_percentages.csv")