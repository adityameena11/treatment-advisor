"""
Trains and compares THREE algorithms per condition (Random Forest, XGBoost,
Logistic Regression) using 5-fold cross-validation, so the project reports a
justified model choice instead of a single untested pick.

The best tree-based model (RF or XGBoost) is kept as the production model,
since SHAP's TreeExplainer gives fast, exact explanations for tree models.
Logistic Regression is trained purely as a comparison baseline.

Run once with: python train_model.py
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import joblib
import shap
import json
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def cross_validate_model(model, X, y):
    scores = cross_validate(model, X, y, cv=CV, scoring=SCORING)
    return {
        metric: round(float(np.mean(scores[f"test_{metric}"])), 3)
        for metric in SCORING
    }


def train_and_save(name, df, feature_names, label_col, feature_meta):
    X = df[feature_names]
    y = df[label_col]

    # ---- 1. Compare three algorithms with 5-fold cross-validation ----
    candidates = {
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=3,
            random_state=42, class_weight="balanced",
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            eval_metric="logloss", random_state=42,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced",
        ),
    }

    comparison = {}
    for model_name, model in candidates.items():
        if model_name == "logistic_regression":
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            comparison[model_name] = cross_validate_model(model, X_scaled, y)
        else:
            comparison[model_name] = cross_validate_model(model, X, y)
        print(f"[{name}/{model_name}] 5-fold CV: {comparison[model_name]}")

    # ---- 2. Pick the winner among tree-based models by ROC-AUC ----
    tree_candidates = {k: v for k, v in comparison.items() if k != "logistic_regression"}
    winner_name = max(tree_candidates, key=lambda k: tree_candidates[k]["roc_auc"])
    print(f"[{name}] selected production model: {winner_name}")

    # ---- 3. Train the winner on a held-out split for a final honest report ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    final_model = candidates[winner_name]
    final_model.fit(X_train, y_train)
    preds = final_model.predict(X_test)
    probs = final_model.predict_proba(X_test)[:, 1]

    holdout_metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 3),
        "precision": round(precision_score(y_test, preds), 3),
        "recall": round(recall_score(y_test, preds), 3),
        "f1": round(f1_score(y_test, preds), 3),
        "roc_auc": round(roc_auc_score(y_test, probs), 3),
    }
    print(f"[{name}] held-out test metrics ({winner_name}): {holdout_metrics}")

    # ---- 4. SHAP explainer for the chosen production model ----
    explainer = shap.TreeExplainer(final_model)

    joblib.dump(final_model, os.path.join(MODELS_DIR, f"{name}_model.pkl"))
    joblib.dump(explainer, os.path.join(MODELS_DIR, f"{name}_explainer.pkl"))

    with open(os.path.join(MODELS_DIR, f"{name}_meta.json"), "w") as f:
        json.dump(
            {
                "feature_names": feature_names,
                "feature_meta": feature_meta,
                "metrics": holdout_metrics,
                "production_model": winner_name,
                "model_comparison": comparison,
            },
            f,
            indent=2,
        )


# ---------------------------------------------------------------------------
# 1. DIABETES  (Pima Indians Diabetes dataset)
# ---------------------------------------------------------------------------
diabetes_cols = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age", "outcome",
]
diabetes_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "diabetes.csv"),
                           names=diabetes_cols)

zero_as_missing = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]
for col in zero_as_missing:
    diabetes_df[col] = diabetes_df[col].replace(0, np.nan)
    diabetes_df[col] = diabetes_df[col].fillna(diabetes_df[col].median())

diabetes_features = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age",
]

diabetes_feature_meta = {
    "pregnancies": {"label": "Pregnancies", "min": 0, "max": 15, "step": 1, "default": 1},
    "glucose": {"label": "Glucose (mg/dL)", "min": 60, "max": 250, "step": 1, "default": 110},
    "blood_pressure": {"label": "Blood Pressure (mm Hg)", "min": 40, "max": 140, "step": 1, "default": 72},
    "skin_thickness": {"label": "Skin Thickness (mm)", "min": 5, "max": 60, "step": 1, "default": 25},
    "insulin": {"label": "Insulin (mu U/mL)", "min": 0, "max": 400, "step": 1, "default": 80},
    "bmi": {"label": "BMI", "min": 15, "max": 55, "step": 0.1, "default": 26.5},
    "diabetes_pedigree": {"label": "Family History Score", "min": 0.05, "max": 2.5, "step": 0.01, "default": 0.4},
    "age": {"label": "Age", "min": 15, "max": 90, "step": 1, "default": 33},
}

train_and_save("diabetes", diabetes_df, diabetes_features, "outcome", diabetes_feature_meta)

# ---------------------------------------------------------------------------
# 2. HEART DISEASE (UCI Cleveland dataset)
# ---------------------------------------------------------------------------
heart_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "heart.csv"), encoding="utf-8-sig")
heart_df.columns = [c.strip() for c in heart_df.columns]

# NOTE ON LABEL DIRECTION:
# In this widely-used public mirror of the Cleveland heart disease dataset,
# target=1 correlates with LOWER cholesterol, LOWER blood pressure, LOWER
# oldpeak, and LOWER exercise-induced angina than target=0 - the reverse of
# standard clinical risk direction. This label-direction ambiguity is a known,
# documented issue with this specific popular CSV mirror. To keep the demo's
# risk score aligned with clinically intuitive direction (higher risk factors
# -> higher predicted risk), we flip the label here. This is stated explicitly
# for transparency/reproducibility - a real deployment would resolve this
# against the original UCI codebook rather than by inspection.
heart_df["target"] = 1 - heart_df["target"]

heart_features = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

heart_feature_meta = {
    "age": {"label": "Age", "min": 20, "max": 90, "step": 1, "default": 50},
    "sex": {"label": "Sex (1=Male, 0=Female)", "min": 0, "max": 1, "step": 1, "default": 1},
    "cp": {"label": "Chest Pain Type (0-3)", "min": 0, "max": 3, "step": 1, "default": 1},
    "trestbps": {"label": "Resting Blood Pressure", "min": 90, "max": 200, "step": 1, "default": 130},
    "chol": {"label": "Cholesterol (mg/dL)", "min": 100, "max": 400, "step": 1, "default": 240},
    "fbs": {"label": "Fasting Blood Sugar > 120 (1=Yes)", "min": 0, "max": 1, "step": 1, "default": 0},
    "restecg": {"label": "Resting ECG Result (0-2)", "min": 0, "max": 2, "step": 1, "default": 1},
    "thalach": {"label": "Max Heart Rate Achieved", "min": 70, "max": 210, "step": 1, "default": 150},
    "exang": {"label": "Exercise-Induced Angina (1=Yes)", "min": 0, "max": 1, "step": 1, "default": 0},
    "oldpeak": {"label": "ST Depression (Exercise)", "min": 0, "max": 6.2, "step": 0.1, "default": 1.0},
    "slope": {"label": "ST Segment Slope (0-2)", "min": 0, "max": 2, "step": 1, "default": 1},
    "ca": {"label": "Major Vessels Colored (0-3)", "min": 0, "max": 3, "step": 1, "default": 0},
    "thal": {"label": "Thalassemia (1-3)", "min": 1, "max": 3, "step": 1, "default": 2},
}

train_and_save("heart", heart_df, heart_features, "target", heart_feature_meta)

print("\nAll models trained and saved to:", MODELS_DIR)
