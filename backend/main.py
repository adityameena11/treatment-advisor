"""
FastAPI backend for the AI-Based Personalized Treatment Advisor.

Endpoints:
  GET  /api/meta/{condition}     -> feature list + slider config for the frontend form
  POST /api/predict              -> risk score + SHAP-based explanation + recommendations

Run locally with:  uvicorn main:app --reload --port 8000
"""
import json
import os

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

app = FastAPI(title="AI-Based Personalized Treatment Advisor")

# Allow the frontend (served from anywhere - a local file, Netlify, Vercel, etc.)
# to call this API. In a production deployment you'd lock allow_origins down to
# your actual frontend domain instead of "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load both trained models once at startup
# ---------------------------------------------------------------------------
CONDITIONS = ["diabetes", "heart"]
_registry = {}

for cond in CONDITIONS:
    model = joblib.load(os.path.join(MODELS_DIR, f"{cond}_model.pkl"))
    explainer = joblib.load(os.path.join(MODELS_DIR, f"{cond}_explainer.pkl"))
    with open(os.path.join(MODELS_DIR, f"{cond}_meta.json")) as f:
        meta = json.load(f)
    _registry[cond] = {"model": model, "explainer": explainer, "meta": meta}


class PredictRequest(BaseModel):
    condition: str
    features: dict


def risk_tier(prob: float) -> str:
    if prob < 0.33:
        return "Low"
    elif prob < 0.66:
        return "Moderate"
    return "High"


# Confidence-aware escalation: when a prediction sits close to the decision
# boundary (near 50%), individual trees in the forest are more likely to
# disagree with each other, meaning the model itself is uncertain about this
# specific case. Rather than presenting a confident-looking percentage
# regardless, the system flags these cases for mandatory human review instead
# of automated interpretation - the same principle used in real clinical
# decision-support tools, which must not present low-confidence output as if
# it were reliable.
UNCERTAINTY_BAND = 0.12  # +/- this far from 0.5 counts as "uncertain"


def uncertainty_check(prob: float) -> dict:
    is_uncertain = abs(prob - 0.5) <= UNCERTAINTY_BAND
    return {
        "is_uncertain": is_uncertain,
        "message": (
            "Model confidence is low for this case (prediction sits near the "
            "decision boundary). Treat this output as inconclusive and rely on "
            "clinical judgement rather than the automated recommendation."
            if is_uncertain else None
        ),
    }


# ---------------------------------------------------------------------------
# Rule-based recommendation engine.
# Each rule fires when a feature is both (a) above/below a clinical threshold
# AND (b) among the factors SHAP identified as pushing risk up for this patient.
# This keeps advice grounded in *this patient's* drivers, not a generic checklist.
# ---------------------------------------------------------------------------
RULES = {
    "diabetes": [
        ("glucose", lambda v: v >= 140, "Blood glucose is in the pre-diabetic/diabetic range. Recommend an HbA1c test and consult a physician about glucose management."),
        ("bmi", lambda v: v >= 30, "BMI indicates obesity, a major modifiable diabetes risk factor. A structured weight-management plan (diet + exercise) is advised."),
        ("bmi", lambda v: 25 <= v < 30, "BMI is in the overweight range. Moderate lifestyle changes (diet, 150 min/week activity) can meaningfully lower risk."),
        ("blood_pressure", lambda v: v >= 90, "Blood pressure is elevated. Recommend monitoring alongside glucose management, as hypertension compounds diabetes risk."),
        ("age", lambda v: v >= 45, "Age is a non-modifiable risk factor; recommend routine annual screening."),
        ("diabetes_pedigree", lambda v: v >= 0.8, "Strong family history detected. Recommend earlier and more frequent screening."),
    ],
    "heart": [
        ("chol", lambda v: v >= 240, "Cholesterol is high. Recommend a lipid panel review and dietary fat reduction; consider statin therapy discussion with a physician."),
        ("trestbps", lambda v: v >= 140, "Resting blood pressure indicates hypertension. Recommend blood pressure monitoring and sodium-reduction advice."),
        ("thalach", lambda v: v <= 120, "Maximum heart rate achieved is low for typical exercise tolerance; recommend a cardiac stress evaluation."),
        ("oldpeak", lambda v: v >= 2, "Significant ST depression under exercise suggests possible ischemia; recommend cardiology referral."),
        ("exang", lambda v: v == 1, "Exercise-induced angina present. This is a strong indicator warranting prompt cardiology follow-up."),
        ("fbs", lambda v: v == 1, "Fasting blood sugar is elevated, adding cardiometabolic risk; recommend coordinated diabetes screening."),
    ],
}

GENERIC_RECOMMENDATION = "No major risk drivers crossed clinical thresholds. Recommend maintaining current healthy habits and routine annual checkups."


def generate_recommendations(condition: str, features: dict, top_shap_features: list) -> list:
    fired = []
    top_names = {f["feature"] for f in top_shap_features if f["contribution"] > 0}
    for feature_name, condition_fn, message in RULES.get(condition, []):
        value = features.get(feature_name)
        if value is None:
            continue
        if condition_fn(value) and feature_name in top_names:
            fired.append(message)
    if not fired:
        return [GENERIC_RECOMMENDATION]
    return fired


@app.get("/api/meta/{condition}")
def get_meta(condition: str):
    if condition not in _registry:
        raise HTTPException(status_code=404, detail="Unknown condition")
    return _registry[condition]["meta"]


@app.post("/api/predict")
def predict(req: PredictRequest):
    if req.condition not in _registry:
        raise HTTPException(status_code=404, detail="Unknown condition")

    entry = _registry[req.condition]
    feature_names = entry["meta"]["feature_names"]

    try:
        row = [float(req.features[name]) for name in feature_names]
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing feature: {e}")

    X = np.array([row])
    model = entry["model"]
    explainer = entry["explainer"]

    prob = float(model.predict_proba(X)[0][1])

    # SHAP values for the positive class (index 1) for this single prediction
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        contributions = shap_values[1][0]
    else:
        # newer SHAP versions return a single array with a class axis
        contributions = shap_values[0][:, 1] if shap_values.ndim == 3 else shap_values[0]

    factor_list = [
        {
            "feature": feature_names[i],
            "label": entry["meta"]["feature_meta"][feature_names[i]]["label"],
            "value": row[i],
            "contribution": round(float(contributions[i]), 4),
        }
        for i in range(len(feature_names))
    ]
    factor_list.sort(key=lambda f: abs(f["contribution"]), reverse=True)
    top_factors = factor_list[:5]

    recommendations = generate_recommendations(req.condition, req.features, top_factors)
    uncertainty = uncertainty_check(prob)

    return {
        "condition": req.condition,
        "risk_probability": round(prob, 4),
        "risk_tier": risk_tier(prob),
        "top_factors": top_factors,
        "recommendations": recommendations,
        "model_metrics": entry["meta"]["metrics"],
        "production_model": entry["meta"]["production_model"],
        "uncertainty": uncertainty,
        "disclaimer": "This is a decision-support demo, not a medical diagnosis. Always consult a qualified clinician.",
    }


@app.get("/api/model-comparison/{condition}")
def model_comparison(condition: str):
    """Returns the 5-fold cross-validation comparison across all algorithms
    tried for this condition, plus which one was selected for production and why."""
    if condition not in _registry:
        raise HTTPException(status_code=404, detail="Unknown condition")
    meta = _registry[condition]["meta"]
    return {
        "production_model": meta["production_model"],
        "comparison": meta["model_comparison"],
        "selection_rule": "Best ROC-AUC among tree-based models (SHAP TreeExplainer requires a tree model for fast, exact explanations).",
    }


@app.get("/")
def root():
    return {"status": "ok", "message": "AI-Based Personalized Treatment Advisor API is running."}
