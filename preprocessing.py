# Import the pandas library
import pandas as pd

# Load the AI4I dataset
df = pd.read_csv("C:\\Users\\HP\\Downloads\\Sem 3\\AI-ML\\Project-AI&ML\\ai4i2020_raw.csv")

# Display the first 5 rows
print("========== FIRST 5 ROWS ==========")
print(df.head())

# Display dataset size
print("\n========== DATASET SHAPE ==========")
print(df.shape)
print("\n========== COLUMN NAMES ==========")
print(df.columns)

# Data types
print("\n========== DATA TYPES ==========")
print(df.dtypes)

# Missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())
# Remove unnecessary columns
df = df.drop(['UDI', 'Product ID'], axis=1)

print("\n========== AFTER REMOVING UDI AND PRODUCT ID ==========")
print(df.head())

# Convert Type into numerical values
df['Type'] = df['Type'].map({
    'L': 0,
    'M': 1,
    'H': 2
})

print("\n========== AFTER ENCODING TYPE ==========")
print(df.head())
print("\n========== DATASET SHAPE AFTER PREPROCESSING ==========")
print(df.shape)
print("\n========== DUPLICATE RECORDS ==========")
print(df.duplicated().sum())
X = df.drop("Machine failure", axis=1)
y = df["Machine failure"]
# Import train_test_split
from sklearn.model_selection import train_test_split

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n========== TRAINING DATA ==========")
print(X_train.shape)

print("\n========== TESTING DATA ==========")
print(X_test.shape)
print("\n========== MACHINE FAILURE COUNT ==========")
print(y.value_counts())
print("\n========== TRAINING FEATURES ==========")
print(X_train.head())

print("\n========== TRAINING TARGET ==========")
print(y_train.head())
# Save the preprocessed dataset
df.to_csv("preprocessed_ai4i.csv", index=False)

print("\n===================================")
print("Preprocessing completed successfully!")
print("Preprocessed dataset saved as: preprocessed_ai4i.csv")
print("===================================")