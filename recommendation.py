def generate_recommendation(prediction, probability, shap_contributions):
    """
    Generate a maintenance recommendation based on
    model prediction, failure probability, and SHAP contributions.
    """

    if prediction == 0:
        return (
            "No immediate maintenance action is required. "
            "Continue routine monitoring of the machine."
        )

    # Find features that increase failure prediction
    risk_factors = []

    for feature, shap_value in shap_contributions.items():
        if shap_value > 0:
            risk_factors.append(feature)

    if probability >= 0.80:
        risk_level = "HIGH"
        action = (
            "Schedule preventive maintenance soon and "
            "inspect the major contributing machine conditions."
        )

    elif probability >= 0.50:
        risk_level = "MEDIUM"
        action = (
            "Increase monitoring frequency and inspect "
            "the identified contributing conditions."
        )

    else:
        risk_level = "LOW"
        action = (
            "Continue routine monitoring and schedule "
            "maintenance according to the normal maintenance cycle."
        )

    return {
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "action": action
    }