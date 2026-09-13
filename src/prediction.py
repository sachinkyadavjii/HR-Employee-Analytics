"""
prediction.py
--------------
Trains a machine-learning model to predict employee attrition, evaluates
it honestly, and saves the best pipeline (preprocessing + model) with
Joblib so the Streamlit dashboard can load it and score new employees.

IMPORTANT / ETHICAL NOTE:
Gender and MaritalStatus are intentionally EXCLUDED from the model
features. They are protected/sensitive attributes and using them as
direct predictive inputs could lead to unfair, discriminatory decisions.
This is a portfolio/educational demonstration, not a real HR decision
engine - see reports/business_insights.md and the dashboard "About" page
for full limitations.

Run directly to (re)train the model:
    python src/prediction.py
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
CLEAN_CSV = os.path.join(BASE_DIR, "data", "hr_employee_data_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "attrition_model.pkl")

# Sensitive attributes deliberately excluded from the feature set.
EXCLUDED_FEATURES = ["EmployeeID", "Attrition", "Gender", "MaritalStatus"]

NUMERIC_FEATURES = [
    "Age", "Education", "JobLevel", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "TotalWorkingYears", "MonthlyIncome", "HourlyRate",
    "JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction",
    "WorkLifeBalance", "PerformanceRating", "TrainingTimesLastYear",
    "NumCompaniesWorked", "DistanceFromHome", "JobInvolvement",
    "StockOptionLevel", "PromotionLast5Years",
]

CATEGORICAL_FEATURES = [
    "Department", "JobRole", "EducationField", "OverTime",
    "BusinessTravel", "EmploymentType",
]


def load_training_data(csv_path: str = CLEAN_CSV) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def build_preprocessor() -> ColumnTransformer:
    """Builds the shared preprocessing step (scaling + one-hot encoding)."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def evaluate_model(name, pipeline, X_test, y_test) -> dict:
    """Evaluates a trained pipeline and returns a metrics dictionary."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics


def train_and_select_model():
    """Trains Logistic Regression and Random Forest, picks the better one
    (by ROC-AUC, a good metric for imbalanced attrition data), and returns
    the winning pipeline plus evaluation metrics for BOTH models."""

    df = load_training_data()

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[features]
    y = (df["Attrition"] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=42, class_weight="balanced"
        ),
    }

    results = {}
    fitted_pipelines = {}

    for name, model in candidates.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", model),
        ])
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(name, pipeline, X_test, y_test)
        results[name] = metrics
        fitted_pipelines[name] = pipeline

    # Pick the model with the higher ROC-AUC score on the held-out test set
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_pipeline = fitted_pipelines[best_name]

    return best_name, best_pipeline, results, (X_test, y_test)


def save_model(pipeline, best_name: str, results: dict, model_path: str = MODEL_PATH):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "model_name": best_name,
        "metrics": results,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
    joblib.dump(bundle, model_path)
    print(f"Saved model bundle ({best_name}) -> {model_path}")


def load_model(model_path: str = MODEL_PATH) -> dict:
    """Loads the saved model bundle (pipeline + metadata) from disk."""
    return joblib.load(model_path)


def risk_level_from_probability(probability: float) -> str:
    """Converts a 0-1 probability into a LOW / MEDIUM / HIGH risk label."""
    pct = probability * 100
    if pct <= 30:
        return "LOW"
    elif pct <= 60:
        return "MEDIUM"
    else:
        return "HIGH"


def predict_single_employee(employee: dict, model_path: str = MODEL_PATH) -> dict:
    """
    Scores a single employee dict (must contain all NUMERIC_FEATURES and
    CATEGORICAL_FEATURES keys) and returns probability + risk level.
    """
    bundle = load_model(model_path)
    pipeline = bundle["pipeline"]

    features = bundle["numeric_features"] + bundle["categorical_features"]
    missing = [f for f in features if f not in employee or employee.get(f) is None]
    if missing:
        raise ValueError(f"Missing required employee fields for prediction: {missing}")

    row = pd.DataFrame([{k: employee.get(k) for k in features}])

    probability = float(pipeline.predict_proba(row)[0, 1])
    risk_level = risk_level_from_probability(probability)

    # Feature importance (only meaningful for tree-based models)
    top_factors = get_top_factors(bundle)

    return {
        "attrition_probability": round(probability * 100, 1),
        "risk_level": risk_level,
        "top_factors": top_factors,
    }


def get_top_factors(bundle: dict, top_n: int = 5):
    """Returns the top N most influential features, if the model supports it."""
    pipeline = bundle["pipeline"]
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return []

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        return []

    order = np.argsort(importances)[::-1][:top_n]
    return [(feature_names[i].split("__")[-1], round(float(importances[i]), 4)) for i in order]


def main():
    print("Training attrition prediction models...\n")
    best_name, best_pipeline, results, _ = train_and_select_model()

    for name, metrics in results.items():
        print(f"--- {name} ---")
        print(f"  Accuracy : {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall   : {metrics['recall']:.3f}")
        print(f"  F1 Score : {metrics['f1']:.3f}")
        print(f"  ROC-AUC  : {metrics['roc_auc']:.3f}")
        print(f"  Confusion Matrix: {metrics['confusion_matrix']}")
        print()

    print(f"Selected model (best ROC-AUC): {best_name}")
    save_model(best_pipeline, best_name, results)


if __name__ == "__main__":
    main()
