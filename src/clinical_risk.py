import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *


def generate_clinical_risk_plot():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)

    # Load data
    df = load_raw_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Load trained XGBoost model
    model = joblib.load(OUTPUT_MODEL_PATH + "xgboost_model.pkl")

    # Predict probabilities
    probs = model.predict_proba(X_test)[:, 1]

    # Risk stratification
    low_risk = np.sum(probs < 0.30)
    medium_risk = np.sum((probs >= 0.30) & (probs <= 0.60))
    high_risk = np.sum(probs > 0.60)

    labels = ["Low Risk", "Medium Risk", "High Risk"]
    values = [low_risk, medium_risk, high_risk]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values)
    plt.xlabel("Clinical Risk Category")
    plt.ylabel("Number of Patients")
    plt.title("Clinical Heart Disease Risk Stratification")
    plt.tight_layout()

    plt.savefig(OUTPUT_PLOT_PATH + "clinical_risk_distribution.png")
    plt.close()

    print("Clinical risk distribution plot generated successfully.")
