import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("preprocessed_ai4i.csv")

print("========== DATASET LOADED ==========")
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ==========================================
# CORRELATION HEATMAP
# ==========================================

# Select numerical machine parameters and target
correlation_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure"
]

correlation_data = df[correlation_columns]

# Calculate correlation matrix
correlation_matrix = correlation_data.corr()

print("\n========== CORRELATION MATRIX ==========")
print(correlation_matrix)


# Plot heatmap
plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="Blues",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Machine Parameters")
plt.tight_layout()
plt.show()


# ==========================================
# BOX PLOT - TOOL WEAR VS MACHINE FAILURE
# ==========================================

plt.figure(figsize=(8, 6))

sns.boxplot(
    x="Machine failure",
    y="Tool wear [min]",
    data=df
)

plt.title("Tool Wear Distribution vs Machine Failure")
plt.xlabel("Machine Failure (0 = No Failure, 1 = Failure)")
plt.ylabel("Tool Wear [min]")

plt.tight_layout()
plt.show()