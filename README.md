# Explainable Predictive Maintenance with SHAP-Based Adaptive Maintenance Recommendation

## 1. Project Description

This project develops a machine learning-based predictive maintenance system to predict potential machine failures using machine operating conditions.

The system uses **LightGBM** for machine failure prediction and **SHAP (SHapley Additive exPlanations)** to explain the features influencing the model's predictions. Based on the predicted failure probability, the application assigns a risk category and provides a corresponding maintenance recommendation.

A Flask-based web application allows users to enter machine parameters and view predictions, explanations, and maintenance recommendations.

## 2. Methodology

The project follows these steps:

1. **Data preprocessing:** Prepare the dataset, select relevant input features, and encode categorical data.
2. **Data visualization:** Use correlation heatmaps and box plots to examine feature relationships and data distributions.
3. **Model training:** Train a LightGBM classifier using six machine operating features.
4. **Model comparison:** Compare the six-feature model with a simplified model using Torque and Tool Wear.
5. **Model evaluation:** Evaluate performance using accuracy, precision, recall, F1-score, and ROC-AUC.
6. **Explainable AI:** Use SHAP to understand how input features influence individual predictions.
7. **Risk classification:** Assign predefined risk categories based on predicted failure probability.
8. **Maintenance recommendation:** Generate a recommendation according to the assigned risk category.
9. **Web implementation:** Integrate prediction, explainability, and recommendation components into a Flask application.

## 3. Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

**Source:** UCI Machine Learning Repository

Dataset: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

The dataset contains approximately 10,000 machine operating records.

The six-feature model uses:

* Type
* Air Temperature
* Process Temperature
* Rotational Speed
* Torque
* Tool Wear

The target variable is `Machine failure`.

The identifier columns and failure-mode columns are excluded from the model inputs to avoid target leakage.

## 4. Implementation and File Guide

The repository contains the source code, trained models, dataset, visualizations, and experimental results.

### Application Files

| File                | Description                                                                                     |
| ------------------- | ----------------------------------------------------------------------------------------------- |
| `app.py`            | Runs the Flask web application and connects prediction, SHAP explanations, and recommendations. |
| `index.html`        | Defines the web interface.                                                                      |
| `style.css`         | Defines the appearance of the web interface.                                                    |
| `recommendation.py` | Implements risk classification and maintenance recommendation logic.                            |

### Data Processing and Model Training

| File                      | Description                                              |
| ------------------------- | -------------------------------------------------------- |
| `ai4i2020_raw.csv`        | Raw dataset.                                             |
| `preprocessed_ai4i.csv`   | Preprocessed dataset.                                    |
| `preprocessing.py`        | Prepares the dataset for analysis and modelling.         |
| `train_model.py`          | Trains the six-feature LightGBM model.                   |
| `train_top2_model.py`     | Trains the two-feature model using Torque and Tool Wear. |
| `lightgbm_model.pkl`      | Saved six-feature model.                                 |
| `lightgbm_top2_model.pkl` | Saved two-feature model.                                 |

### Explainability and Visualization

| File                                     | Description                                                  |
| ---------------------------------------- | ------------------------------------------------------------ |
| `visualization.py`                       | Generates exploratory data visualizations.                   |
| `shap_analysis.py`                       | Performs SHAP analysis for the main model.                   |
| `shap_top2_analysis.py`                  | Performs SHAP analysis for the two-feature model.            |
| `shap_summary.png`                       | SHAP summary visualization.                                  |
| `shap_feature_importance.png`            | SHAP feature importance visualization.                       |
| `shap_waterfall.png`                     | SHAP explanation for an individual prediction.               |
| `shap_top2_summary.png`                  | SHAP summary visualization for the two-feature model.        |
| `shap_top2_feature_importance.png`       | Feature importance visualization for the two-feature model.  |
| `shap_top2_waterfall.png`                | Individual prediction explanation for the two-feature model. |
| `high_risk_shap_waterfall.png`           | SHAP waterfall visualization for a high-risk prediction.     |
| `model_performance_comparison.png`       | Model performance comparison graph.                          |
| `feature_importance_rank_comparison.png` | Feature importance ranking comparison graph.                 |

### Model Evaluation and Validation

| File                                | Description                                                |
| ----------------------------------- | ---------------------------------------------------------- |
| `compare_models.py`                 | Compares model performance.                                |
| `plot_model_comparison.py`          | Generates model comparison graphs.                         |
| `check_model_overfitting.py`        | Examines training and testing performance.                 |
| `compare_feature_importance.py`     | Compares feature importance results.                       |
| `model_comparison_summary.csv`      | Summary of model comparison results.                       |
| `model_comparison_all_runs.csv`     | Results from individual model comparison runs.             |
| `model_comparison_percentages.csv`  | Model comparison metrics in percentage form.               |
| `feature_importance_comparison.csv` | Feature importance comparison results.                     |
| `overfitting_check_all_runs.csv`    | Results from individual overfitting-check runs.            |
| `overfitting_check_summary.csv`     | Summary of overfitting-check results.                      |
| `top2_model_all_runs.csv`           | Individual experimental results for the two-feature model. |
| `top2_model_summary.csv`            | Summary of two-feature model results.                      |

### Other Files

| File               | Description                                                          |
| ------------------ | -------------------------------------------------------------------- |
| `requirements.txt` | Lists the Python dependencies required by the project.               |
| `README.md`        | Documents the project and explains how to run it.                    |
| `settings.json`    | Configuration file; check whether it is required by the application. |

The file descriptions above are based on the repository filenames and intended roles. Refer to each script for its exact implementation.

## 5. Demonstration Instructions

### Prerequisites

* Python installed on the system.
* The required libraries listed in `requirements.txt`.
* The trained model files and any other files required by the application.

### Step 1: Download the Repository

Clone the repository:

```bash
git clone https://github.com/Aryananda936/predictive-maintenance-shap.git
```

Navigate to the project directory:

```bash
cd predictive-maintenance-shap
```

Alternatively, download the repository as a ZIP file from GitHub and extract it.

### Step 2: Install Dependencies

Run:

```bash
pip install -r requirements.txt
```

### Step 3: Start the Application

Run:

```bash
python app.py
```

Open the local address displayed in the terminal in a web browser.

### Step 4: Demonstrate the Application

1. Enter the machine operating parameters requested by the interface.
2. Submit the inputs to generate a machine failure prediction.
3. View the predicted failure probability.
4. Examine the SHAP contributions to understand the factors influencing the prediction.
5. View the risk category and corresponding maintenance recommendation.

### Risk Classification

The application uses predefined probability thresholds:

* **High Risk:** Probability ≥ 0.80
* **Medium Risk:** Probability ≥ 0.50 and < 0.80
* **Low Risk:** Probability < 0.50

These thresholds are predefined application rules.

## 6. Reproducing the Experiments

The repository includes scripts for preprocessing, training, visualization, and evaluation.

The following commands can be used to execute the corresponding scripts when their dependencies and input files are available:

```bash
python preprocessing.py
python visualization.py
python train_model.py
python train_top2_model.py
python shap_analysis.py
python shap_top2_analysis.py
python compare_models.py
python check_model_overfitting.py
python compare_feature_importance.py
python plot_model_comparison.py
```

**Note:** Check each script's input files and execution requirements before running it. Some scripts may generate files required by later steps or overwrite existing model and result files.

The saved CSV files and graphs can be used to inspect the results of the experiments.

## 7. Model Evaluation Results

The previously reported mean results across ten random train-test splits are shown below. Verify these values against the final experimental output files before submission.

| Metric    | Six-Feature Model | Two-Feature Model |
| --------- | ----------------: | ----------------: |
| Accuracy  |            98.04% |            92.99% |
| Precision |            69.04% |            26.11% |
| Recall    |            77.94% |            57.65% |
| F1-score  |            73.13% |            35.91% |
| ROC-AUC   |            97.26% |            89.35% |

The six-feature model performed better across the reported evaluation metrics. The two-feature model uses fewer inputs but has lower predictive performance on these experiments.

