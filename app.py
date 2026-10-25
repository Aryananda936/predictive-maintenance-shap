from flask import Flask, render_template, request
import pandas as pd
import joblib
import shap
import re

from recommendation import generate_recommendation


# ==========================================
# CREATE FLASK APP
# ==========================================

app = Flask(__name__)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = joblib.load("lightgbm_model.pkl")

# Create SHAP explainer
explainer = shap.TreeExplainer(model)


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        # -------------------------------
        # Get machine ID
        # -------------------------------

        machine_id = request.form["machine_id"]


        # -------------------------------
        # Get machine inputs
        # -------------------------------

        machine_data = {
            "Type": float(request.form["type"]),
            "Air temperature [K]": float(
                request.form["air_temperature"]
            ),
            "Process temperature [K]": float(
                request.form["process_temperature"]
            ),
            "Rotational speed [rpm]": float(
                request.form["rotational_speed"]
            ),
            "Torque [Nm]": float(
                request.form["torque"]
            ),
            "Tool wear [min]": float(
                request.form["tool_wear"]
            )
        }


        # -------------------------------
        # Convert to DataFrame
        # -------------------------------

        X = pd.DataFrame([machine_data])


        # -------------------------------
        # Clean feature names
        # -------------------------------

        X.columns = [
            re.sub(r'[^A-Za-z0-9_]+', '_', col).strip('_')
            for col in X.columns
        ]


        # -------------------------------
        # Prediction
        # -------------------------------

        prediction = model.predict(X)[0]

        probability = model.predict_proba(X)[0][1]


        # -------------------------------
        # SHAP explanation
        # -------------------------------

        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_values = shap_values[0]


        # -------------------------------
        # Create SHAP contribution dictionary
        # -------------------------------

        shap_contributions = {
            feature: float(value)
            for feature, value in zip(
                X.columns,
                shap_values
            )
        }


        # -------------------------------
        # Sort SHAP contributions
        # -------------------------------

        sorted_shap = sorted(
            shap_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )


        # -------------------------------
        # Generate recommendation
        # -------------------------------

        recommendation = generate_recommendation(
            prediction,
            probability,
            shap_contributions
        )


        # -------------------------------
        # Prepare result
        # -------------------------------

        result = {
            "machine_id": machine_id,
            "prediction": int(prediction),
            "probability": probability,
            "shap": sorted_shap,
            "recommendation": recommendation
        }


    return render_template(
        "index.html",
        result=result
    )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)